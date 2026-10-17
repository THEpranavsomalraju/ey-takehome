"""PredictionService: the facade the dashboard and the agent use (docs/02 section 8).

Stateless per call: every prediction stands alone, and the loaded bundle is
read-only, so dashboard reruns and concurrent requests are safe.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ledgerclf.artifacts import FALLBACK_BUNDLE, ModelBundle, load_best_available, read_metadata
from ledgerclf.config import Settings, get_settings
from ledgerclf.exceptions import ArtifactError
from ledgerclf.explain import explain_row
from ledgerclf.predict import PredictionResult, predict_file
from ledgerclf.reports import load_report, read_manifest, to_jsonable

logger = logging.getLogger(__name__)


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

    def predict_file(self, data: bytes, file_name: str) -> PredictionResult:
        """Validate and predict an uploaded file.

        Args:
            data: File bytes.
            file_name: Original file name.

        Returns:
            The prediction result.
        """
        return predict_file(
            self.bundle, data, file_name, self.settings.limits, self.fallback_reason
        )

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
