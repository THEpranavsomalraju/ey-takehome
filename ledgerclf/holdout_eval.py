"""The single holdout evaluation (docs/04 section 11, CLAUDE.md rule 1).

Runs only from ``python -m ledgerclf evaluate-holdout --confirm``. Refuses to
run twice unless a reason is given, logs every access, and scores both
models with CV estimates beside every holdout number.
"""

from __future__ import annotations

import datetime as dt
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from ledgerclf.artifacts import CHAMPION_BUNDLE, FALLBACK_BUNDLE, load_bundle, read_metadata
from ledgerclf.config import (
    HOLDOUT_DEFAULT_REASON,
    HOLDOUT_EVALUATION_REPORT,
    HOLDOUT_FEATURES_STEM,
    Settings,
)
from ledgerclf.evaluation import compute_metrics, confusion_table, per_class_report
from ledgerclf.exceptions import ArtifactError, HoldoutAccessError, SchemaError
from ledgerclf.io import sha256_file
from ledgerclf.labels import normalize_labels
from ledgerclf.predict import PROBABILITY_COLUMNS, predict_file
from ledgerclf.reports import write_json, write_table
from ledgerclf.schema import LABEL_COLUMN, ROW_NUMBER
from ledgerclf.splitting import HoldoutAccessToken, read_holdout_labels
from ledgerclf.training import CHAMPION_NAME, FALLBACK_NAME

PRODUCER = "evaluate-holdout"
EVALUATION_FILE = HOLDOUT_EVALUATION_REPORT
ACCESS_LOG = "holdout/access_log.csv"


def _git_commit(settings: Settings) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=settings.paths.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _log_access(settings: Settings, token: HoldoutAccessToken) -> None:
    path = settings.paths.reports / ACCESS_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = pd.DataFrame(
        [
            {
                "utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
                "git_commit": _git_commit(settings),
                "reason": token.reason,
                "champion_sha256": sha256_file(settings.paths.artifacts / CHAMPION_BUNDLE),
                "fallback_sha256": sha256_file(settings.paths.artifacts / FALLBACK_BUNDLE),
            }
        ]
    )
    previous = pd.read_csv(path, encoding="utf-8") if path.exists() else None
    table = entry if previous is None else pd.concat([previous, entry], ignore_index=True)
    table.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def run_holdout_evaluation(
    settings: Settings, token: HoldoutAccessToken, rerun: bool
) -> dict[str, Any]:
    """Score both models once on the holdout and write ``reports/holdout``.

    Args:
        settings: Project settings.
        token: Access token issued by the CLI after ``--confirm``.
        rerun: Allow a second evaluation (requires a reason in the token).

    Returns:
        Headline metrics for the CLI.

    Raises:
        HoldoutAccessError: When an evaluation already exists and ``rerun`` is False.
    """
    already = (settings.paths.reports / EVALUATION_FILE).exists()
    if already and not rerun:
        raise HoldoutAccessError(
            "The holdout was already evaluated (reports/holdout/holdout_evaluation.json). A second "
            'look needs --rerun --reason "...", which is logged.',
            code="HOLDOUT_ALREADY_EVALUATED",
        )
    if already and token.reason.strip() in ("", HOLDOUT_DEFAULT_REASON):
        raise HoldoutAccessError(
            'A second holdout look needs a written reason: --rerun --reason "why".',
            code="HOLDOUT_RERUN_REASON_MISSING",
        )
    _log_access(settings, token)
    labels = read_holdout_labels(settings.paths, token)
    y_true, _ = normalize_labels(labels[LABEL_COLUMN])
    if y_true.isna().any():
        raise SchemaError(
            "Some holdout labels do not map to the six classes; the evaluation stopped. "
            "Check data/holdout against the split manifest.",
            code="UNMAPPED_HOLDOUT_LABEL",
        )
    source = settings.paths.holdout / f"{HOLDOUT_FEATURES_STEM}.xlsx"
    metadata = read_metadata(settings.paths)
    summary: dict[str, Any] = {
        "rows": len(labels),
        "cv_estimate_champion": metadata["cv_summary"],
        "cv_estimate_fallback": metadata["fallback_cv_summary"],
    }
    for name, file in ((CHAMPION_NAME, CHAMPION_BUNDLE), (FALLBACK_NAME, FALLBACK_BUNDLE)):
        try:
            bundle = load_bundle(file, settings.paths)
        except ArtifactError as error:
            summary[name] = {"skipped": error.message}
            continue
        result = predict_file(bundle, source, source.name, settings.limits)
        aligned = _aligned_labels(result.predictions, labels, y_true)
        summary[name] = _score_model(settings, name, result.predictions, aligned)
    write_json(
        settings,
        EVALUATION_FILE,
        summary,
        "Holdout evaluation",
        PRODUCER,
        "Single evaluation of both models on the 590-row holdout, with the CV estimate beside "
        "each number.",
    )
    return summary


def _aligned_labels(
    predictions: pd.DataFrame, labels: pd.DataFrame, y_true: pd.Series
) -> np.ndarray:
    """Join labels to predictions by row number; refuse if any row was dropped or added."""
    by_row = pd.Series(y_true.to_numpy(dtype=object), index=labels[ROW_NUMBER].astype(int))
    rows = predictions[ROW_NUMBER].astype(int)
    if len(predictions) != len(labels) or not rows.isin(by_row.index).all():
        raise HoldoutAccessError(
            "Holdout predictions and labels do not line up row for row; the evaluation stopped. "
            "Regenerate the holdout files with: python -m ledgerclf split",
            code="HOLDOUT_MISALIGNED",
        )
    return by_row.loc[rows].to_numpy(dtype=object)


def _score_model(
    settings: Settings, name: str, predictions: pd.DataFrame, y_true: Any
) -> dict[str, Any]:
    y_pred = predictions["predicted_label"].to_numpy(dtype=object)
    proba = predictions[list(PROBABILITY_COLUMNS)].to_numpy()
    metrics = compute_metrics(y_true, y_pred, proba)
    correct = y_pred == y_true
    by_flag = _accuracy_by_flag(predictions, correct)
    review = predictions["review_required"].to_numpy(dtype=bool)
    per_class = per_class_report(y_true, y_pred)
    if name == CHAMPION_NAME:
        _write_champion_tables(settings, predictions, y_true, per_class, by_flag)
    return metrics | {
        "review_share": float(review.mean()),
        "accuracy_auto": float(correct[~review].mean()) if (~review).any() else None,
        "accuracy_reviewed": float(correct[review].mean()) if review.any() else None,
        "accuracy_by_flag": by_flag,
        "per_class": per_class.to_dict(orient="records"),
        "support": {str(k): int(v) for k, v in pd.Series(y_true).value_counts().items()},
    }


def _accuracy_by_flag(predictions: pd.DataFrame, correct: np.ndarray) -> list[dict[str, Any]]:
    """Accuracy of the rows carrying each flag code (exact codes, not substrings)."""
    flag_sets = [set(str(f).split("|")) - {""} for f in predictions["flags"].fillna("")]
    by_flag = []
    for code in sorted(set().union(*flag_sets)):
        mask = np.array([code in flags for flags in flag_sets], dtype=bool)
        by_flag.append(
            {"flag": code, "rows": int(mask.sum()), "accuracy": float(correct[mask].mean())}
        )
    return by_flag


def _write_champion_tables(
    settings: Settings,
    predictions: pd.DataFrame,
    y_true: Any,
    per_class: pd.DataFrame,
    by_flag: list[dict[str, Any]],
) -> None:
    y_pred = predictions["predicted_label"].to_numpy(dtype=object)
    tables = {
        "holdout/holdout_classification_report.csv": (
            per_class,
            "Holdout classification report",
            "Champion on the holdout: precision, recall, F1, support per class.",
        ),
        "holdout/holdout_confusion_matrix.csv": (
            confusion_table(y_true, y_pred),
            "Holdout confusion matrix",
            "Champion on the holdout: true class (rows) versus predicted class (columns).",
        ),
        "holdout/holdout_predictions.csv": (
            predictions.assign(true_label=y_true, correct=y_pred == y_true),
            "Holdout predictions",
            "Champion predictions on every holdout row with the true label.",
        ),
        "holdout/holdout_accuracy_by_flag.csv": (
            pd.DataFrame(by_flag),
            "Holdout accuracy by flag",
            "Champion accuracy on holdout rows carrying each flag.",
        ),
    }
    for file, (frame, title, description) in tables.items():
        write_table(settings, file, frame, title, PRODUCER, description)
