"""The single holdout evaluation (docs/04 section 11, CLAUDE.md rule 1).

Runs only from ``python -m ledgerclf evaluate-holdout --confirm``. Refuses to
run twice unless a reason is given, logs every access, and scores both
models with CV estimates beside every holdout number.
"""

from __future__ import annotations

import datetime as dt
import subprocess
from typing import Any

import pandas as pd

from ledgerclf.artifacts import CHAMPION_BUNDLE, FALLBACK_BUNDLE, load_bundle, read_metadata
from ledgerclf.config import HOLDOUT_FEATURES_STEM, Settings
from ledgerclf.evaluation import compute_metrics, confusion_table, per_class_report
from ledgerclf.exceptions import ArtifactError, HoldoutAccessError
from ledgerclf.io import sha256_file
from ledgerclf.labels import normalize_labels
from ledgerclf.predict import PROBABILITY_COLUMNS, predict_file
from ledgerclf.reports import write_json, write_table
from ledgerclf.splitting import HoldoutAccessToken, read_holdout_labels
from ledgerclf.training import CHAMPION_NAME, FALLBACK_NAME

PRODUCER = "evaluate-holdout"
EVALUATION_FILE = "holdout/holdout_evaluation.json"
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
    if (settings.paths.reports / EVALUATION_FILE).exists() and not rerun:
        raise HoldoutAccessError(
            "The holdout was already evaluated (reports/holdout/holdout_evaluation.json). A second "
            'look needs --rerun --reason "...", which is logged.',
            code="HOLDOUT_ALREADY_EVALUATED",
        )
    _log_access(settings, token)
    labels = read_holdout_labels(settings.paths, token)
    y_true, _ = normalize_labels(labels["ClassificationLabel"])
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
        summary[name] = _score_model(
            settings, name, result.predictions, y_true.to_numpy(dtype=object)
        )
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


def _score_model(
    settings: Settings, name: str, predictions: pd.DataFrame, y_true: Any
) -> dict[str, Any]:
    y_pred = predictions["predicted_label"].to_numpy(dtype=object)
    proba = predictions[list(PROBABILITY_COLUMNS)].to_numpy()
    metrics = compute_metrics(y_true, y_pred, proba)
    flags = predictions["flags"].fillna("")
    correct = y_pred == y_true
    by_flag = [
        {
            "flag": code,
            "rows": int(mask.sum()),
            "accuracy": float(correct[mask].mean()) if mask.any() else None,
        }
        for code in sorted({c for f in flags for c in str(f).split("|") if c})
        for mask in [flags.str.contains(code, regex=False).to_numpy()]
    ]
    review = predictions["review_required"].to_numpy(dtype=bool)
    if name == CHAMPION_NAME:
        write_table(
            settings,
            "holdout/holdout_classification_report.csv",
            per_class_report(y_true, y_pred),
            "Holdout classification report",
            PRODUCER,
            "Champion on the holdout: precision, recall, F1, support per class.",
        )
        write_table(
            settings,
            "holdout/holdout_confusion_matrix.csv",
            confusion_table(y_true, y_pred),
            "Holdout confusion matrix",
            PRODUCER,
            "Champion on the holdout: true class (rows) versus predicted class (columns).",
        )
        write_table(
            settings,
            "holdout/holdout_predictions.csv",
            predictions.assign(true_label=y_true, correct=correct),
            "Holdout predictions",
            PRODUCER,
            "Champion predictions on every holdout row with the true label.",
        )
        write_table(
            settings,
            "holdout/holdout_accuracy_by_flag.csv",
            pd.DataFrame(by_flag),
            "Holdout accuracy by flag",
            PRODUCER,
            "Champion accuracy on holdout rows carrying each flag.",
        )
    return metrics | {
        "review_share": float(review.mean()),
        "accuracy_auto": float(correct[~review].mean()) if (~review).any() else None,
        "accuracy_reviewed": float(correct[review].mean()) if review.any() else None,
        "accuracy_by_flag": by_flag,
        "support": {str(k): int(v) for k, v in pd.Series(y_true).value_counts().items()},
    }
