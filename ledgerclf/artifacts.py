"""Model bundles: save, load with version checks, and the best-available choice.

The champion bundle may contain LightGBM objects; the fallback bundle never
does, so it loads where LightGBM's native library is missing.
``load_best_available`` loads the champion when LightGBM works and the
fallback otherwise, and reports which one is active.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import sklearn

from ledgerclf.config import Paths
from ledgerclf.environment import lightgbm_status
from ledgerclf.exceptions import ArtifactError
from ledgerclf.flags import ReviewPolicy
from ledgerclf.reference import ReferenceStats

logger = logging.getLogger(__name__)

CHAMPION_BUNDLE = "model_champion.joblib"
FALLBACK_BUNDLE = "model_fallback.joblib"
METADATA_FILE = "metadata.json"
REFERENCE_FILE = "reference.json"
GOLDEN_FILE = "golden_holdout_predictions.csv"
SAMPLE_PREDICTIONS_FILE = "sample_predictions.csv"
SAMPLE_ROWS_FILE = "sample_rows.csv"
RETRAIN_FIX = "Run: python -m ledgerclf train"


@dataclass
class ModelBundle:
    """A fitted pipeline with everything needed to predict and explain."""

    pipeline: Any
    reference: ReferenceStats
    policy: ReviewPolicy
    metadata: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    is_fallback: bool = False


def save_bundle(bundle: ModelBundle, file_name: str, paths: Paths) -> Path:
    """Write a bundle with joblib (plain dicts for reference and policy).

    Args:
        bundle: The fitted bundle.
        file_name: File name inside ``artifacts/``.
        paths: Project paths.

    Returns:
        The written path.
    """
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": bundle.pipeline,
        "reference": bundle.reference.to_dict(),
        "policy": bundle.policy.to_dict(),
        "metadata": bundle.metadata,
        "name": bundle.name,
        "is_fallback": bundle.is_fallback,
        "sklearn_version": sklearn.__version__,
    }
    path = paths.artifacts / file_name
    joblib.dump(payload, path, compress=3)
    return path


def load_bundle(file_name: str, paths: Paths) -> ModelBundle:
    """Load one bundle from ``artifacts/``.

    Args:
        file_name: Bundle file name.
        paths: Project paths.

    Returns:
        The bundle.

    Raises:
        ArtifactError: When the file is missing, corrupt, or needs a library
            that cannot load (the message names the fix).
    """
    path = paths.artifacts / file_name
    if not path.exists():
        raise ArtifactError(
            f"Model file {file_name} is missing. {RETRAIN_FIX}", code="ARTIFACT_MISSING"
        )
    try:
        payload = joblib.load(path)
    except (ImportError, OSError) as error:
        raise ArtifactError(
            f"Model file {file_name} needs a library that failed to load ({type(error).__name__}).",
            code="ARTIFACT_LIBRARY_MISSING",
        ) from error
    except (
        pickle.UnpicklingError,
        EOFError,
        ValueError,
        AttributeError,
        KeyError,
        TypeError,
    ) as error:
        raise ArtifactError(
            f"Model file {file_name} is damaged or was saved by incompatible library versions. "
            f"{RETRAIN_FIX}",
            code="ARTIFACT_CORRUPT",
        ) from error
    _warn_on_version_change(payload, file_name)
    return ModelBundle(
        pipeline=payload["pipeline"],
        reference=ReferenceStats.from_dict(payload["reference"]),
        policy=ReviewPolicy.from_dict(payload["policy"]),
        metadata=payload["metadata"],
        name=payload["name"],
        is_fallback=bool(payload["is_fallback"]),
    )


def _major_minor(version: str) -> str:
    return ".".join(version.split(".")[:2])


def _warn_on_version_change(payload: dict[str, Any], file_name: str) -> None:
    saved = str(payload.get("sklearn_version", "unknown"))
    if _major_minor(saved) != _major_minor(sklearn.__version__):
        logger.warning(
            "%s was saved with scikit-learn %s; %s is installed. If predictions fail, %s",
            file_name,
            saved,
            sklearn.__version__,
            RETRAIN_FIX,
        )


def load_best_available(paths: Paths) -> tuple[ModelBundle, str | None]:
    """Load the champion when LightGBM works, else the fallback.

    Args:
        paths: Project paths.

    Returns:
        Tuple of (bundle, fallback reason or None when the champion loaded).

    Raises:
        ArtifactError: When neither bundle can load.
    """
    available, reason = lightgbm_status()
    if available:
        try:
            return load_bundle(CHAMPION_BUNDLE, paths), None
        except ArtifactError as error:
            reason = error.message
            logger.warning("Champion model unavailable, using the fallback: %s", reason)
    return load_bundle(FALLBACK_BUNDLE, paths), reason


def read_metadata(paths: Paths) -> dict[str, Any]:
    """Return ``artifacts/metadata.json``.

    Raises:
        ArtifactError: When the file is missing or unreadable.
    """
    path = paths.artifacts / METADATA_FILE
    try:
        metadata: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(
            f"{METADATA_FILE} is missing or unreadable. {RETRAIN_FIX}", code="METADATA_MISSING"
        ) from error
    return metadata
