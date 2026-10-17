"""Final training: fit the champion and the fallback on the training portion only.

``train`` reads ``artifacts/champion.json`` (written by ``select``), never ad
hoc arguments, so the trained model always matches the recorded decision.
It also writes reference statistics, metadata, the doctor's sample rows, and
golden predictions on the holdout features (no labels are read).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import platform
from dataclasses import replace
from importlib import metadata as importlib_metadata
from typing import Any

import numpy as np
import pandas as pd

from ledgerclf import __version__
from ledgerclf.artifacts import (
    CHAMPION_BUNDLE,
    FALLBACK_BUNDLE,
    GOLDEN_FILE,
    METADATA_FILE,
    REFERENCE_FILE,
    SAMPLE_PREDICTIONS_FILE,
    SAMPLE_ROWS_FILE,
    ModelBundle,
    save_bundle,
)
from ledgerclf.config import HOLDOUT_FEATURES_STEM, Settings
from ledgerclf.exceptions import ArtifactError, ConfigError
from ledgerclf.flags import ReviewPolicy
from ledgerclf.io import sha256_file
from ledgerclf.labels import CANONICAL_CLASSES, FIVE_CLASSES
from ledgerclf.models import make_blend, make_calibrated_logistic
from ledgerclf.predict import PROBABILITY_COLUMNS, predict_file
from ledgerclf.reference import build_reference
from ledgerclf.schema import INPUT_COLUMNS, ROW_NUMBER
from ledgerclf.selection import CHAMPION_FILE
from ledgerclf.splitting import MANIFEST_FILE, TrainingData, load_split, load_training_frame

logger = logging.getLogger(__name__)

CHAMPION_NAME = "champion"
FALLBACK_NAME = "fallback_logistic"
LIBRARIES = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "joblib")


def read_champion(settings: Settings) -> dict[str, Any]:
    """Return ``artifacts/champion.json``.

    Raises:
        ConfigError: When ``select`` has not run yet.
    """
    path = settings.paths.artifacts / CHAMPION_FILE
    if not path.exists():
        raise ConfigError(
            "No champion configuration yet. Run: python -m ledgerclf select",
            code="CHAMPION_MISSING",
        )
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def build_pipeline(settings: Settings, config: dict[str, Any]) -> Any:
    """Build the unfitted pipeline described by one champion.json entry."""
    power = float(config["class_weight_power"])
    linear = replace(settings.linear, C=float(config["linear"]["C"]), class_weight_power=power)
    features = replace(settings.feature, ngram_range=tuple(config["features"]["ngram_range"]))
    temperature = float(config["temperature"])
    if config["family"] == "logistic":
        return make_calibrated_logistic(linear, temperature, features)
    tree = replace(settings.tree, **config["tree"], class_weight_power=power)
    blend = replace(
        settings.blend, linear_weight=float(config["linear_weight"]), temperature=temperature
    )
    return make_blend(linear, tree, blend, int(config["random_state"]), features)


def fallback_config(champion: dict[str, Any]) -> dict[str, Any]:
    """Return the fallback entry as a pure logistic configuration (no tree member)."""
    config = dict(champion["fallback"])
    config["family"] = "logistic"
    return config


def _library_versions() -> dict[str, str]:
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def build_metadata(
    settings: Settings, champion: dict[str, Any], data: TrainingData
) -> dict[str, Any]:
    """Return metadata.json content."""
    split = load_split(settings.paths)
    return {
        "schema_version": "1.0",
        "created_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "package_version": __version__,
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "library_versions": _library_versions(),
        "raw_data_sha256": split.manifest["raw_data_sha256"],
        "split_manifest_sha256": sha256_file(settings.paths.splits / MANIFEST_FILE),
        "holdout_seed": split.manifest["seed"],
        "training_rows": len(data.y),
        "classes": list(CANONICAL_CLASSES),
        "champion_config": champion["champion"],
        "fallback_config": champion["fallback"],
        "temperature": champion["champion"]["temperature"],
        "thresholds": champion["review_policy"],
        "cv_summary": champion["champion"]["scores"],
        "oot_summary": {
            "accuracy": champion["champion"]["scores"]["oot_accuracy"],
            "macro_f1_5": champion["champion"]["scores"]["oot_macro_f1_5"],
        },
        "fallback_cv_summary": champion["fallback"]["scores"],
        "tuning_quick_mode": champion.get("tuning_quick_mode", False),
    }


def sample_rows(data: TrainingData) -> pd.DataFrame:
    """Pick one training row per common and rare class for the doctor's self-test."""
    picks = []
    for cls in FIVE_CLASSES:
        picks.append(int(np.flatnonzero(data.y == cls)[0]))
    rows = data.raw.iloc[picks][list(INPUT_COLUMNS)].copy()
    return rows.map(lambda v: v.isoformat(sep=" ") if hasattr(v, "isoformat") else v)


def _require_writable(settings: Settings) -> None:
    for folder in (settings.paths.artifacts, settings.paths.resources):
        folder.mkdir(parents=True, exist_ok=True)
        if not os.access(folder, os.W_OK):
            raise ArtifactError(
                f"Cannot retrain: {folder.name}/ is read-only. Predictions still work with the "
                "existing model. Copy the project to a writable folder to retrain.",
                code="ARTIFACTS_READ_ONLY",
            )


def run_training(settings: Settings) -> dict[str, Any]:
    """Fit both bundles on the training portion and write every artifact.

    Args:
        settings: Project settings.

    Returns:
        A short summary for the CLI.
    """
    _require_writable(settings)
    champion = read_champion(settings)
    data = load_training_frame(settings)
    reference = build_reference(data.frame, data.y)
    policy = ReviewPolicy.from_dict(champion["review_policy"])
    metadata = build_metadata(settings, champion, data)
    bundles = {}
    for name, file, config, is_fallback in (
        (CHAMPION_NAME, CHAMPION_BUNDLE, champion["champion"], False),
        (FALLBACK_NAME, FALLBACK_BUNDLE, fallback_config(champion), True),
    ):
        pipeline = build_pipeline(settings, config).fit(data.frame, data.y)
        bundle = ModelBundle(pipeline, reference, policy, metadata, name, is_fallback)
        save_bundle(bundle, file, settings.paths)
        bundles[name] = bundle
        logger.info("trained %s (%s)", name, config["name"])
    artifacts = settings.paths.artifacts
    (artifacts / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    (artifacts / REFERENCE_FILE).write_text(
        json.dumps(reference.to_dict(), sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    _write_samples(settings, data, bundles)
    _write_golden(settings, bundles)
    return {
        "champion": champion["champion"]["name"],
        "fallback": champion["fallback"]["name"],
        "training_rows": len(data.y),
    }


def _write_samples(settings: Settings, data: TrainingData, bundles: dict[str, ModelBundle]) -> None:
    resources = settings.paths.resources
    resources.mkdir(parents=True, exist_ok=True)
    path = resources / SAMPLE_ROWS_FILE
    sample_rows(data).to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    expected = pd.DataFrame({ROW_NUMBER: np.arange(1, len(FIVE_CLASSES) + 1)})
    for name, bundle in bundles.items():
        result = predict_file(bundle, path, path.name, settings.limits)
        expected[f"predicted_label_{name}"] = result.predictions["predicted_label"].to_numpy()
    expected.to_csv(
        settings.paths.artifacts / SAMPLE_PREDICTIONS_FILE,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )


def _write_golden(settings: Settings, bundles: dict[str, ModelBundle]) -> None:
    source = settings.paths.holdout / f"{HOLDOUT_FEATURES_STEM}.xlsx"
    tables = []
    for name, bundle in bundles.items():
        result = predict_file(bundle, source, source.name, settings.limits)
        columns = [
            ROW_NUMBER,
            "predicted_label",
            "confidence",
            *PROBABILITY_COLUMNS,
            "review_required",
            "flags",
        ]
        tables.append(result.predictions[columns].assign(model=name))
    golden = pd.concat(tables, ignore_index=True)
    golden.to_csv(
        settings.paths.artifacts / GOLDEN_FILE,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.10f",
    )
