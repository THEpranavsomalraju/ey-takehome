"""HTTP API for the dashboard (FastAPI), plus the built dashboard files.

Run with ``python -m ledgerclf serve``. Uploads are predicted in memory and
kept for explanations in a small bounded cache; nothing is written to disk.
Every error reaches the browser as JSON with a plain-English message.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ledgerclf import __version__
from ledgerclf.agent import explain_with_agent, resolve_connection
from ledgerclf.config import HOLDOUT_FEATURES_STEM
from ledgerclf.exceptions import ArtifactError, LedgerClfError
from ledgerclf.jsonsafe import to_jsonable
from ledgerclf.predict import PredictionResult
from ledgerclf.service import PredictionService

logger = logging.getLogger(__name__)

MAX_CACHED_UPLOADS = 20
DEFAULT_PAGE_ROWS = 25
MAX_PAGE_ROWS = 500
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
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


class ExplainRequest(BaseModel):
    """Body of POST /api/explain."""

    upload_id: str
    row_number: int = Field(ge=1)
    use_llm: bool = True
    api_key: str | None = None


def _page(result: PredictionResult, offset: int, limit: int) -> dict[str, Any]:
    limit = max(1, min(limit, MAX_PAGE_ROWS))
    offset = max(0, offset)
    rows = result.predictions.iloc[offset : offset + limit]
    records: list[dict[str, Any]] = to_jsonable(rows.to_dict(orient="records"))
    return {"offset": offset, "limit": limit, "total": len(result.predictions), "rows": records}


def create_app(service: PredictionService) -> FastAPI:
    """Build the API around a loaded prediction service.

    Args:
        service: Loaded PredictionService.

    Returns:
        The FastAPI application.
    """
    app = FastAPI(title="Ledger Classifier", version=__version__)
    cache = ResultCache(MAX_CACHED_UPLOADS)

    @app.exception_handler(LedgerClfError)
    async def ledger_error(_: Request, error: LedgerClfError) -> JSONResponse:
        status = 503 if isinstance(error, ArtifactError) else 422
        return JSONResponse(status_code=status, content=error.to_dict())

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

    def _predicted(result: PredictionResult) -> dict[str, Any]:
        upload_id = cache.add(result)
        return {
            "upload_id": upload_id,
            "summary": result.summary(),
            "page": _page(result, 0, DEFAULT_PAGE_ROWS),
        }

    @app.post("/api/predict")
    async def predict(file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008 - FastAPI idiom
        data = await file.read()
        return _predicted(service.predict_file(data, file.filename or "upload"))

    @app.post("/api/predict/sample")
    def predict_sample() -> dict[str, Any]:
        path = service.settings.paths.holdout / f"{HOLDOUT_FEATURES_STEM}.xlsx"
        return _predicted(service.predict_file(path.read_bytes(), path.name))

    @app.get("/api/sample/holdout")
    def sample_file() -> FileResponse:
        path = service.settings.paths.holdout / f"{HOLDOUT_FEATURES_STEM}.xlsx"
        return FileResponse(path, filename=path.name)

    @app.get("/api/predict/{upload_id}/rows")
    def rows(upload_id: str, offset: int = 0, limit: int = DEFAULT_PAGE_ROWS) -> dict[str, Any]:
        return _page(cache.get(upload_id), offset, limit)

    @app.get("/api/predict/{upload_id}/download")
    def download(upload_id: str) -> Response:
        csv = cache.get(upload_id).predictions.to_csv(index=False).encode("utf-8-sig")
        headers = {"Content-Disposition": 'attachment; filename="predictions.csv"'}
        return Response(content=csv, media_type="text/csv", headers=headers)

    @app.post("/api/explain")
    def explain(request: ExplainRequest) -> dict[str, Any]:
        result = cache.get(request.upload_id)
        answer = explain_with_agent(
            service, result, request.row_number, request.use_llm, request.api_key
        )
        payload = service.explain(result, request.row_number)
        return {"answer": answer.to_dict(), "payload": payload}

    @app.get("/api/agent/status")
    def agent_status() -> dict[str, Any]:
        connection = resolve_connection(service.settings)
        route = connection.route if connection else "offline"
        return {"route": route, "model": service.settings.agent.model if connection else None}

    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    if (FRONTEND_DIST / "index.html").exists():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="dashboard")
        return

    @app.get("/", response_class=HTMLResponse)
    def placeholder() -> str:
        return PLACEHOLDER


def serve(host: str, port: int, open_browser: bool) -> None:
    """Load the service and run the API server until interrupted.

    Args:
        host: Interface to bind (127.0.0.1 keeps it local).
        port: TCP port.
        open_browser: Open the dashboard in the default browser.
    """
    import uvicorn

    app = create_app(PredictionService.load(auto_train=True))
    if open_browser:
        import webbrowser

        threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
