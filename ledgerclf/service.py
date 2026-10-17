"""PredictionService: the facade the dashboard and the agent use (docs/02 section 8).

Stateless per call: every prediction stands alone, and the loaded bundle is
read-only, so dashboard reruns and concurrent requests are safe.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, BinaryIO

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from ledgerclf.artifacts import FALLBACK_BUNDLE, ModelBundle, load_best_available, read_metadata
from ledgerclf.config import Settings, get_settings
from ledgerclf.exceptions import ArtifactError, FileReadError, LedgerClfError
from ledgerclf.explain import explain_row
from ledgerclf.holdout_eval import EVALUATION_FILE
from ledgerclf.io import read_table
from ledgerclf.jsonsafe import to_jsonable
from ledgerclf.overlap import matches, row_fingerprints
from ledgerclf.predict import PredictionResult, predict_file
from ledgerclf.reports import load_report, read_manifest
from ledgerclf.schema import (
    LABEL_COLUMN,
    Severity,
    ValidationIssue,
    coerce_values,
    make_issue,
    resolve_columns,
)
from ledgerclf.splitting import HOLDOUT_WITH_LABELS_FILE, load_training_frame

logger = logging.getLogger(__name__)

DEFAULT_UPLOAD_NAME = "upload"


@dataclass(frozen=True)
class TrainingIndex:
    """Training rows with L2-normalized feature vectors for cosine similarity."""

    frame: pd.DataFrame
    y: np.ndarray
    matrix: sp.csr_matrix


@dataclass(frozen=True)
class KnownRows:
    """Fingerprints of the unreleased holdout rows and of the training rows."""

    holdout: frozenset[str]
    training: frozenset[str]


def upload_bytes(data: object) -> bytes:
    """Return the content of an upload given as bytes or as a binary file object.

    Raises:
        FileReadError: When ``data`` is neither (for example a path string or None).
    """
    if isinstance(data, bytes | bytearray | memoryview):
        return bytes(data)
    read = getattr(data, "read", None)
    content = read() if callable(read) else None
    if isinstance(content, bytes | bytearray):
        return bytes(content)
    raise FileReadError(
        "The upload is not file content. Pass the file's bytes (for example "
        "Path(...).read_bytes() or the upload widget's file object).",
        code="UPLOAD_NOT_BYTES",
    )


def _holdout_notice(rows: int, total: int) -> ValidationIssue:
    return make_issue(
        Severity.INFO,
        "HOLDOUT_METRICS_WITHHELD",
        f"{rows} of {total} row(s) belong to the labeled holdout. They were predicted, but "
        "their labels stay unread until the single, logged holdout evaluation has run "
        "(python -m ledgerclf evaluate-holdout). Accuracy figures leave them out.",
    )


def _in_sample_notice(rows: int, total: int) -> ValidationIssue:
    return make_issue(
        Severity.WARNING,
        "TRAINING_ROWS_IN_UPLOAD",
        f"{rows} of {total} row(s) were part of the training data, so accuracy on them is "
        "in-sample and overstates performance on new data. Upload rows the model has not "
        "seen (for example the holdout features file) for an honest estimate.",
    )


class PredictionService:
    """Load the best available model once; predict and explain uploaded files.

    Args:
        bundle: Loaded model bundle.
        fallback_reason: Why the fallback model is active, if it is.
        settings: Project settings.
    """

    def __init__(
        self, bundle: ModelBundle, fallback_reason: str | None, settings: Settings
    ) -> None:
        self.bundle = bundle
        self.fallback_reason = fallback_reason
        self.settings = settings
        self._training_index: TrainingIndex | None = None
        self._known: KnownRows | None = None
        self._lock = threading.Lock()

    @classmethod
    def load(cls, auto_train: bool = True, settings: Settings | None = None) -> PredictionService:
        """Load the champion, or the fallback when LightGBM cannot load.

        Args:
            auto_train: Retrain from the raw file and the saved split when no
                artifacts exist (needs a writable project folder).
            settings: Project settings (defaults to this project).

        Returns:
            The service.

        Raises:
            ArtifactError: When no model can load and retraining is off or fails.
        """
        settings = settings or get_settings()
        if not (settings.paths.artifacts / FALLBACK_BUNDLE).exists():
            if not auto_train:
                raise ArtifactError(
                    "No trained model found. Run: python -m ledgerclf train",
                    code="ARTIFACT_MISSING",
                )
            _retrain(settings)
        bundle, reason = load_best_available(settings.paths)
        return cls(bundle, reason, settings)

    def predict_file(self, data: bytes | BinaryIO, file_name: str | None) -> PredictionResult:
        """Validate and predict an uploaded file.

        Before the single holdout evaluation has run, holdout rows get
        predictions but their labels stay unread (CLAUDE.md rule 1), wherever
        they appear. Labeled training rows are scored with an in-sample caveat.

        Args:
            data: File bytes, or a binary file object such as an upload widget's file.
            file_name: Original file name (used for the format hint).

        Returns:
            The prediction result.

        Raises:
            FileReadError: When the upload is not file content or cannot be read.
            SchemaError: When the file structure cannot be mapped.
        """
        content = upload_bytes(data)
        limits, reason = self.settings.limits, self.fallback_reason
        name = file_name or DEFAULT_UPLOAD_NAME
        return predict_file(self.bundle, content, name, limits, reason, self._screen_labels)

    def _screen_labels(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[ValidationIssue]]:
        withheld = matches(frame, self._known_rows().holdout)
        in_training = matches(frame, self._known_rows().training) & ~withheld
        issues = []
        if withheld.any():
            issues.append(_holdout_notice(int(withheld.sum()), len(frame)))
        if in_training.any():
            issues.append(_in_sample_notice(int(in_training.sum()), len(frame)))
        return withheld, issues

    def _known_rows(self) -> KnownRows:
        with self._lock:
            if self._known is None:
                self._known = KnownRows(self._holdout_fingerprints(), self._training_fingerprints())
            return self._known

    def _holdout_fingerprints(self) -> frozenset[str]:
        paths = self.settings.paths
        labeled = paths.holdout / HOLDOUT_WITH_LABELS_FILE
        if (paths.reports / EVALUATION_FILE).exists() or not labeled.exists():
            return frozenset()
        raw, _, _ = read_table(labeled, labeled.name, self.settings.limits)
        resolved, _, _ = resolve_columns(raw, self.settings.limits)
        inputs = resolved.drop(columns=[LABEL_COLUMN], errors="ignore")
        return frozenset(row_fingerprints(coerce_values(inputs, self.settings.limits)[0]))

    def _training_fingerprints(self) -> frozenset[str]:
        try:
            return frozenset(row_fingerprints(load_training_frame(self.settings).frame))
        except LedgerClfError as error:
            logger.info("Training rows unavailable for the in-sample check: %s", error.message)
            return frozenset()

    def explain(self, result: PredictionResult, row_number: int) -> dict[str, Any]:
        """Return the exact explanation payload of one predicted row.

        Args:
            result: A result from ``predict_file``.
            row_number: 1-based row number as shown in the predictions table.

        Returns:
            The JSON-safe payload.
        """
        return explain_row(self.bundle, result, row_number)

    def model_info(self) -> dict[str, Any]:
        """Return the active model, fallback status, training metadata, and review policy."""
        metadata = read_metadata(self.settings.paths)
        return to_jsonable(
            {
                "model_used": self.bundle.name,
                "fallback_active": self.bundle.is_fallback,
                "fallback_reason": self.fallback_reason,
                "champion": metadata["champion_config"],
                "fallback": metadata["fallback_config"],
                "training_rows": metadata["training_rows"],
                "created_utc": metadata["created_utc"],
                "review_policy": self.bundle.policy.to_dict(),
                "cv_summary": metadata["cv_summary"],
                "library_versions": metadata["library_versions"],
            }
        )

    def list_reports(self) -> list[dict[str, Any]]:
        """Return the report manifest entries."""
        return read_manifest(self.settings)

    def load_report(self, file: str) -> pd.DataFrame | dict[str, Any]:
        """Load one report listed in the manifest."""
        return load_report(self.settings, file)

    def training_index(self) -> TrainingIndex:
        """Return training rows with normalized feature vectors, built once (similar rows).

        Raises:
            ArtifactError: When the training data is not available in this copy.
        """
        if self._training_index is None:
            try:
                data = load_training_frame(self.settings)
            except LedgerClfError as error:
                raise ArtifactError(
                    f"Similar training rows are unavailable: {error.message}",
                    code="TRAINING_DATA_MISSING",
                ) from error
            features = self.bundle.pipeline.named_steps["features"]
            matrix = normalize(sp.csr_matrix(features.transform(data.frame)))
            self._training_index = TrainingIndex(data.frame, data.y, matrix)
        return self._training_index

    def health(self) -> dict[str, Any]:
        """Return the doctor checks as JSON."""
        from ledgerclf.doctor import run_doctor

        return run_doctor(self.settings).to_dict()


def _retrain(settings: Settings) -> None:
    from ledgerclf.training import run_training

    logger.warning("No trained model found; retraining from the raw data and the saved split.")
    try:
        run_training(settings)
    except OSError as error:
        raise ArtifactError(
            f"Retraining failed: the project folder is not writable ({type(error).__name__}). "
            "Copy the project to a writable folder, or restore the artifacts/ folder.",
            code="RETRAIN_FAILED",
        ) from error
