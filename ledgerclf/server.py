"""HTTP API for the dashboard (FastAPI), plus the built dashboard files.

Run with ``python -m ledgerclf serve``. Uploads are predicted in memory and
kept for explanations in a small bounded cache; nothing is written to disk.
Every error reaches the browser as JSON with a plain-English message.
"""

from __future__ import annotations

import logging
import mimetypes
import socket
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from ledgerclf import __version__
from ledgerclf.agent import explain_with_agent, resolve_connection
from ledgerclf.agent_tools import ToolContext, run_tool
from ledgerclf.chart_explainer import ChartAnswerCache, explain_chart
from ledgerclf.config import HOLDOUT_FEATURES_STEM
from ledgerclf.exceptions import ArtifactError, ConfigError, LedgerClfError
from ledgerclf.jsonsafe import to_jsonable
from ledgerclf.predict import PredictionResult
from ledgerclf.service import PredictionService
from ledgerclf.table_views import RowQuery, SortOrder, filtered_rows, insights, page

logger = logging.getLogger(__name__)

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
INTERNAL_ERROR = {
    "code": "INTERNAL",
    "message": "Something went wrong on the server. Restart it with --verbose and try again.",
}
PLACEHOLDER = """<!doctype html><html><head><meta charset="utf-8">
<title>Ledger Classifier</title></head>
<body style="font-family:system-ui;margin:3rem;max-width:40rem">
<h1>Ledger Classifier API is running</h1>
<p>The dashboard build was not found in <code>frontend/dist</code>. The API is available at
<a href="/docs">/docs</a>.</p></body></html>"""


class ResultCache:
    """Bounded, thread-safe store of recent prediction results (for explanations)."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._items: OrderedDict[str, PredictionResult] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, result: PredictionResult) -> str:
        """Store a result and return its id, evicting the oldest beyond capacity."""
        upload_id = uuid.uuid4().hex
        with self._lock:
            self._items[upload_id] = result
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return upload_id

    def get(self, upload_id: str) -> PredictionResult:
        """Return a stored result.

        Raises:
            HTTPException: 404 when the id is unknown or expired.
        """
        with self._lock:
            result = self._items.get(upload_id)
        if result is None:
            raise HTTPException(404, "This upload has expired. Upload the file again.")
        return result


class WhatIfRequest(BaseModel):
    """Body of POST /api/whatif: re-score one row with one input changed."""

    upload_id: str
    row_number: int = Field(ge=1)
    column: str
    value: str = Field(max_length=1000)


class ChartExplainRequest(BaseModel):
    """Body of POST /api/explain-chart."""

    chart: str = Field(max_length=64)
    view: str = Field(default="", max_length=500)
    upload_id: str | None = None
    use_llm: bool = True
    api_key: str | None = None


class ExplainRequest(BaseModel):
    """Body of POST /api/explain."""

    upload_id: str
    row_number: int = Field(ge=1)
    use_llm: bool = True
    api_key: str | None = None


def _sample_path(service: PredictionService) -> Path:
    path = service.settings.paths.holdout / f"{HOLDOUT_FEATURES_STEM}.xlsx"
    if not path.exists():
        raise ArtifactError(
            f"The sample file {path.name} is missing. Recreate it with: python -m ledgerclf split",
            code="SAMPLE_MISSING",
        )
    return path


def create_app(service: PredictionService) -> FastAPI:
    """Build the API around a loaded prediction service.

    Args:
        service: Loaded PredictionService.

    Returns:
        The FastAPI application.
    """
    app = FastAPI(title="Ledger Classifier", version=__version__)
    _add_error_handlers(app)
    _add_info_routes(app, service)
    _add_prediction_routes(app, service, ResultCache(service.settings.server.cached_uploads))
    _mount_frontend(app)
    return app


def _add_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(LedgerClfError)
    async def ledger_error(_: Request, error: LedgerClfError) -> JSONResponse:
        status = 503 if isinstance(error, ArtifactError) else 422
        if error.code == "REPORT_MISSING":
            status = 404
        return JSONResponse(status_code=status, content=error.to_dict())

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, error: StarletteHTTPException) -> JSONResponse:
        code = "NOT_FOUND" if error.status_code == 404 else f"HTTP_{error.status_code}"
        return JSONResponse(
            status_code=error.status_code, content={"code": code, "message": str(error.detail)}
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, error: RequestValidationError) -> JSONResponse:
        if any(tuple(item["loc"]) == ("body", "file") for item in error.errors()):
            problems = "Choose an Excel (.xlsx, .xls) or CSV file to upload."
        else:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in item['loc'])}: {item['msg']}" for item in error.errors()
            )
        return JSONResponse(
            status_code=422, content={"code": "INVALID_REQUEST", "message": problems}
        )

    @app.exception_handler(Exception)
    async def unexpected_error(_: Request, error: Exception) -> JSONResponse:
        logger.debug("Unexpected server error", exc_info=error)
        return JSONResponse(status_code=500, content=INTERNAL_ERROR)


def _add_info_routes(app: FastAPI, service: PredictionService) -> None:
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return service.health()

    @app.get("/api/model")
    def model() -> dict[str, Any]:
        return service.model_info()

    @app.get("/api/reports")
    def reports() -> list[dict[str, Any]]:
        return service.list_reports()

    @app.get("/api/reports/{file_path:path}")
    def report(file_path: str) -> Any:
        loaded = service.load_report(file_path)
        if isinstance(loaded, pd.DataFrame):
            return {
                "columns": list(loaded.columns),
                "rows": to_jsonable(loaded.to_dict(orient="records")),
            }
        return to_jsonable(loaded)

    @app.get("/api/agent/status")
    def agent_status() -> dict[str, Any]:
        connection = resolve_connection(service.settings)
        route = connection.route if connection else "offline"
        return {"route": route, "model": service.settings.agent.model if connection else None}

    @app.get("/api/sample/holdout")
    def sample_file() -> FileResponse:
        path = _sample_path(service)
        return FileResponse(path, filename=path.name)


def _add_prediction_routes(app: FastAPI, service: PredictionService, cache: ResultCache) -> None:
    params = service.settings.server

    threshold = service.bundle.policy.threshold

    def predicted(result: PredictionResult) -> dict[str, Any]:
        total = len(result.predictions)
        return {
            "upload_id": cache.add(result),
            "summary": result.summary(),
            "insights": insights(result, threshold, params.histogram_bins),
            "page": page(result.predictions, 0, params.default_page_rows, total),
        }

    @app.post("/api/predict")
    def predict(file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008 - FastAPI idiom
        # A plain def runs in the worker pool, so a large upload never blocks other requests.
        return predicted(service.predict_file(file.file.read(), file.filename))

    @app.post("/api/predict/sample")
    def predict_sample() -> dict[str, Any]:
        path = _sample_path(service)
        return predicted(service.predict_file(path.read_bytes(), path.name))

    @app.get("/api/predict/{upload_id}/rows")
    def rows(
        upload_id: str,
        offset: int = 0,
        limit: int = params.default_page_rows,
        q: str = "",
        label: str | None = None,
        review: bool = False,
        flag: str | None = None,
        sort: SortOrder = "row",
    ) -> Any:
        predictions = cache.get(upload_id).predictions
        query = RowQuery(search=q, label=label, review_only=review, flag=flag, sort=sort)
        size = max(1, min(limit, params.max_page_rows))
        return page(filtered_rows(predictions, query), offset, size, len(predictions))

    @app.get("/api/predict/{upload_id}/download")
    def download(upload_id: str) -> Response:
        csv = cache.get(upload_id).to_csv().encode("utf-8-sig")
        headers = {"Content-Disposition": 'attachment; filename="predictions.csv"'}
        return Response(content=csv, media_type="text/csv", headers=headers)

    @app.get("/api/explain/{upload_id}/{row_number}")
    def explanation_payload(upload_id: str, row_number: int) -> dict[str, Any]:
        return service.explain(cache.get(upload_id), row_number)

    @app.post("/api/whatif")
    def what_if_route(request: WhatIfRequest) -> dict[str, Any]:
        context = ToolContext(service, cache.get(request.upload_id), request.row_number)
        return run_tool(context, "what_if", {"column": request.column, "value": request.value})

    chart_cache = ChartAnswerCache(service.settings.agent.chart_cache_size)

    @app.post("/api/explain-chart")
    def explain_chart_route(request: ChartExplainRequest) -> dict[str, Any]:
        result = cache.get(request.upload_id) if request.upload_id else None
        key = (request.chart, " ".join(request.view.split()), request.upload_id or "")
        cached = chart_cache.get(key) if request.use_llm else None
        if cached is not None:
            return cached.to_dict() | {"cached": True}
        answer = explain_chart(
            service, request.chart, request.view, result, request.use_llm, request.api_key
        )
        chart_cache.put(key, answer)
        return answer.to_dict() | {"cached": False}

    @app.post("/api/explain")
    def explain(request: ExplainRequest) -> dict[str, Any]:
        result = cache.get(request.upload_id)
        answer = explain_with_agent(
            service, result, request.row_number, request.use_llm, request.api_key
        )
        payload = service.explain(result, request.row_number)
        return {"answer": answer.to_dict(), "payload": payload}


# Windows can map .js to text/plain in its registry, and browsers refuse module
# scripts served that way; pin the types the dashboard build uses.
STATIC_TYPES = (
    ("text/javascript", ".js"),
    ("text/css", ".css"),
    ("image/svg+xml", ".svg"),
    ("font/woff2", ".woff2"),
)


def _mount_frontend(app: FastAPI) -> None:
    for media_type, extension in STATIC_TYPES:
        mimetypes.add_type(media_type, extension)
    if (FRONTEND_DIST / "index.html").exists():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="dashboard")
        return

    @app.get("/", response_class=HTMLResponse)
    def placeholder() -> str:
        return PLACEHOLDER


def _require_free_port(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError as error:
            raise ConfigError(
                f"Port {port} on {host} is in use or not allowed. Start on another port, "
                f"for example: python -m ledgerclf serve --port {port + 10}",
                code="PORT_IN_USE",
            ) from error


def serve(host: str, port: int, open_browser: bool) -> None:
    """Load the service and run the API server until interrupted.

    Args:
        host: Interface to bind (127.0.0.1 keeps it local).
        port: TCP port.
        open_browser: Open the dashboard in the default browser.

    Raises:
        ConfigError: When the port is already in use.
    """
    import uvicorn

    _require_free_port(host, port)
    service = PredictionService.load(auto_train=True)
    app = create_app(service)
    if open_browser:
        import webbrowser

        delay = service.settings.server.browser_delay_seconds
        threading.Timer(delay, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
