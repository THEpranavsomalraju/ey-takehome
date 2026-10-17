"""Benchmark command: leaderboard, ablations, class weight dial, leaf size, validation schemes.

Every run uses the training portion only, grouped CV with the group key
from ``splitting.py``, and the model factories in ``models.py``.
``quick`` mode cuts seeds and variants for development; final reports come
from a full run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ledgerclf.config import FeatureParams, Settings
from ledgerclf.evaluation import (
    CVResult,
    HoldOutResult,
    accuracy_by_pair_seen,
    grouped_splits,
    random_splits,
    run_cv,
    run_train_test,
)
from ledgerclf.features import DEFAULT_TRANSFORMERS
from ledgerclf.labels import CANONICAL_CLASSES, MAJORITY_CLASS
from ledgerclf.models import (
    logistic_with_features,
    make_blend,
    make_complement_nb,
    make_dummy,
    make_lightgbm,
    make_linear_svc,
    make_logistic,
    make_lookup,
    make_probability_blend,
)
from ledgerclf.reports import write_reports
from ledgerclf.splitting import TrainingData, load_training_frame, out_of_time_mask, vendor_groups

logger = logging.getLogger(__name__)

PRODUCER = "benchmark"
RESEARCH_CATBOOST = Path("research") / "catboost.json"


@dataclass(frozen=True)
class BenchmarkPlan:
    """Seeds and variant lists for one benchmark run."""

    seeds: tuple[int, ...]
    dial_powers: tuple[float, ...]
    leaf_sizes: tuple[int, ...]
    jobs: int


def plan_for(settings: Settings, quick: bool) -> BenchmarkPlan:
    """Return the full plan, or the reduced plan for ``--quick`` development runs."""
    params = settings.benchmark
    if quick:
        return BenchmarkPlan(
            settings.seeds[:1], params.quick_dial_powers, params.quick_leaf_sizes, params.cv_jobs
        )
    return BenchmarkPlan(settings.seeds, params.dial_powers, params.leaf_sizes, params.cv_jobs)


def model_zoo(settings: Settings) -> dict[str, Any]:
    """Return every leaderboard model, unfitted."""
    seed = settings.random_state
    linear0 = replace(settings.linear, class_weight_power=0.0)
    linear05 = replace(settings.linear, class_weight_power=0.5)
    tree0 = replace(settings.tree, class_weight_power=0.0)
    blend = settings.blend
    bench = settings.benchmark
    return {
        "dummy_majority": make_dummy(),
        "lookup_vendor_account": make_lookup(),
        "complement_nb": make_complement_nb(bench.nb_alpha),
        "linear_svc": make_linear_svc(bench.svc_C, bench.svc_power, seed, bench.svc_max_iter),
        "logistic_p0": make_logistic(linear0),
        "logistic_p05": make_logistic(linear05),
        "lightgbm_leaf20": make_lightgbm(bench.leaf20_tree, seed),
        "lightgbm_p0": make_lightgbm(tree0, seed),
        "probability_blend_p0": make_probability_blend(linear0, tree0, seed),
        "logodds_blend_p0": make_blend(linear0, tree0, blend, seed),
        "logodds_blend_p025": make_blend(
            replace(linear0, class_weight_power=0.25),
            replace(tree0, class_weight_power=0.25),
            blend,
            seed,
        ),
    }


def _leaderboard_row(result: CVResult) -> dict[str, Any]:
    summary = result.summary
    row: dict[str, Any] = {
        "model": result.model_name,
        "accuracy_mean": summary["accuracy_mean"],
        "accuracy_std": summary["accuracy_std"],
        "macro_f1_5_mean": summary["macro_f1_5_seed_mean"],
        "macro_f1_5_std": summary["macro_f1_5_seed_std"],
        "macro_f1_6_mean": summary["macro_f1_6_seed_mean"],
        "balanced_accuracy_mean": summary["balanced_accuracy_seed_mean"],
    }
    for cls in CANONICAL_CLASSES:
        row[f"recall_{cls}"] = summary[f"recall_{cls}_seed_mean"]
    row["log_loss_mean"] = summary.get("log_loss_seed_mean", np.nan)
    row["fit_seconds"] = summary["fit_seconds"]
    row["folds"] = summary["folds"]
    return row


def run_leaderboard(
    settings: Settings, data: TrainingData, plan: BenchmarkPlan
) -> dict[str, CVResult]:
    """Score every model of the zoo on primary grouped CV."""
    splits = grouped_splits(data.y, data.groups, plan.seeds, settings.cv_folds)
    results = {}
    for name, estimator in model_zoo(settings).items():
        logger.info("leaderboard: %s", name)
        results[name] = run_cv(name, estimator, data.frame, data.y, splits, plan.jobs)
    return results


def leaderboard_tables(
    results: dict[str, CVResult], settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the leaderboard (sorted by accuracy) and the per-seed table."""
    board = pd.DataFrame([_leaderboard_row(r) for r in results.values()])
    board["note"] = ""
    research = settings.paths.reports / RESEARCH_CATBOOST
    if research.exists():
        board = pd.concat([board, _catboost_row(research)], ignore_index=True)
    board = board.sort_values("accuracy_mean", ascending=False, kind="stable").reset_index(
        drop=True
    )
    per_seed = pd.concat(
        [r.per_seed.assign(model=name) for name, r in results.items()], ignore_index=True
    )
    return board, per_seed[["model", *[c for c in per_seed.columns if c != "model"]]]


def _catboost_row(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(
        [
            {
                "model": "catboost",
                "accuracy_mean": payload["accuracy"],
                "macro_f1_5_mean": payload["macro_f1_5"],
                "fit_seconds": payload["fit_seconds"],
                "folds": 1,
                "note": "research only: 1 fold, see research/catboost_benchmark.py",
            }
        ]
    )


ABLATION_ADDITIONS: dict[str, tuple[tuple[str, ...], FeatureParams]] = {
    "add col2 format features": ((*DEFAULT_TRANSFORMERS, "col2_format"), FeatureParams()),
    "add token bigrams": (DEFAULT_TRANSFORMERS, FeatureParams(ngram_range=(1, 2))),
}


def run_ablation(settings: Settings, data: TrainingData, plan: BenchmarkPlan) -> pd.DataFrame:
    """Drop each feature group in turn (and try two additions) on ``logistic_p0``."""
    splits = grouped_splits(data.y, data.groups, plan.seeds, settings.cv_folds)
    linear0 = replace(settings.linear, class_weight_power=0.0)
    variants: dict[str, tuple[tuple[str, ...], FeatureParams]] = {
        "full feature set": (DEFAULT_TRANSFORMERS, FeatureParams())
    }
    for name in DEFAULT_TRANSFORMERS:
        variants[f"drop {name}"] = (
            tuple(t for t in DEFAULT_TRANSFORMERS if t != name),
            FeatureParams(),
        )
    variants.update(ABLATION_ADDITIONS)
    rows = []
    for variant, (transformers, features) in variants.items():
        estimator = logistic_with_features(linear0, transformers, features)
        result = run_cv(variant, estimator, data.frame, data.y, splits, plan.jobs)
        rows.append(_leaderboard_row(result) | {"variant": variant})
    table = pd.DataFrame(rows)
    base = table.loc[table["variant"] == "full feature set"].iloc[0]
    table["delta_accuracy"] = table["accuracy_mean"] - base["accuracy_mean"]
    table["delta_macro_f1_5"] = table["macro_f1_5_mean"] - base["macro_f1_5_mean"]
    columns = [
        "variant",
        "accuracy_mean",
        "accuracy_std",
        "macro_f1_5_mean",
        "delta_accuracy",
        "delta_macro_f1_5",
    ]
    return table[columns + [f"recall_{c}" for c in CANONICAL_CLASSES]]


def run_class_weight_dial(
    settings: Settings, data: TrainingData, plan: BenchmarkPlan
) -> pd.DataFrame:
    """Score logistic and blend models across class weight powers, in CV and out of time."""
    splits = grouped_splits(data.y, data.groups, plan.seeds, settings.cv_folds)
    oot = out_of_time_mask(data.frame, settings.oot_cutoff)
    rows = []
    for power in plan.dial_powers:
        linear = replace(settings.linear, class_weight_power=power)
        tree = replace(settings.tree, class_weight_power=power)
        family = {
            "logistic": make_logistic(linear),
            "logodds_blend": make_blend(linear, tree, settings.blend, settings.random_state),
        }
        for name, estimator in family.items():
            cv = run_cv(f"{name}_p{power}", estimator, data.frame, data.y, splits, plan.jobs)
            held = run_train_test(estimator, data.frame, data.y, oot)
            rows.append(
                {
                    "model": name,
                    "power": power,
                    "cv_accuracy": cv.summary["accuracy_mean"],
                    "cv_macro_f1_5": cv.summary["macro_f1_5_seed_mean"],
                    "oot_accuracy": held.metrics["accuracy"],
                    "oot_macro_f1_5": held.metrics["macro_f1_5"],
                    "oot_majority_rows_predicted_rare": held.rare_alarms,
                }
            )
    return pd.DataFrame(rows)


def run_leaf_size(settings: Settings, data: TrainingData, plan: BenchmarkPlan) -> pd.DataFrame:
    """Score LightGBM across minimum leaf sizes (rare classes need tiny leaves)."""
    splits = grouped_splits(data.y, data.groups, plan.seeds, settings.cv_folds)
    rows = []
    for size in plan.leaf_sizes:
        tree = replace(settings.tree, min_child_samples=size, class_weight_power=0.0)
        result = run_cv(
            f"min_child_samples={size}",
            make_lightgbm(tree, settings.random_state),
            data.frame,
            data.y,
            splits,
            plan.jobs,
        )
        rows.append(_leaderboard_row(result) | {"min_child_samples": size})
    columns = ["min_child_samples", "accuracy_mean", "accuracy_std", "macro_f1_5_mean"]
    return pd.DataFrame(rows)[columns + [f"recall_{c}" for c in CANONICAL_CLASSES]]


def run_validation_schemes(
    settings: Settings, data: TrainingData, plan: BenchmarkPlan
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Leakage demonstration, out-of-time, unseen vendor, and seen versus unseen pair.

    Returns:
        Tuple of (validation scheme table, out-of-time detail, pair accuracy table).
    """
    linear0 = replace(settings.linear, class_weight_power=0.0)
    tree0 = replace(settings.tree, class_weight_power=0.0)
    models = {
        "logistic_p0": make_logistic(linear0),
        "logodds_blend_p0": make_blend(linear0, tree0, settings.blend, settings.random_state),
        # Balanced weights reproduce the exploration configuration behind the unseen-vendor
        # reference accuracy, so the reference check compares like with like.
        "logistic_p1": make_logistic(replace(settings.linear, class_weight_power=1.0)),
    }
    seed = settings.seeds[0]
    grouped = grouped_splits(data.y, data.groups, (seed,), settings.cv_folds)
    schemes = {
        "grouped CV (primary)": grouped,
        "random CV (leakage demo)": random_splits(data.y, seed, settings.cv_folds),
        "unseen-vendor CV": grouped_splits(
            data.y, vendor_groups(data.frame), (seed,), settings.cv_folds
        ),
    }
    rows, pair_tables, oot_detail = [], [], {}
    oot_mask = out_of_time_mask(data.frame, settings.oot_cutoff)
    for name, estimator in models.items():
        for scheme, splits in schemes.items():
            result = run_cv(name, estimator, data.frame, data.y, splits, plan.jobs)
            pooled = {str(k): float(v) for k, v in result.per_seed.iloc[0].items()}
            rows.append(_scheme_row(name, scheme, pooled, len(data.y)))
            if scheme == "grouped CV (primary)":
                pair_tables.append(accuracy_by_pair_seen(result, data.frame, data.y, splits, seed))
        held = run_train_test(estimator, data.frame, data.y, oot_mask)
        rows.append(_scheme_row(name, oot_scheme_name(settings), held.metrics, held.test_rows))
        oot_detail[name] = _oot_detail(held, data, oot_mask)
    return pd.DataFrame(rows), oot_detail, pd.concat(pair_tables, ignore_index=True)


def oot_scheme_name(settings: Settings) -> str:
    """Return the out-of-time scheme label, for example ``out-of-time (train <= 2018-09)``."""
    return f"out-of-time (train <= {settings.oot_cutoff[:7]})"


def _scheme_row(model: str, scheme: str, metrics: dict[str, float], rows: int) -> dict[str, Any]:
    return {
        "model": model,
        "scheme": scheme,
        "rows_scored": rows,
        "accuracy": metrics["accuracy"],
        "macro_f1_5": metrics["macro_f1_5"],
        "balanced_accuracy": metrics["balanced_accuracy"],
    }


def _oot_detail(held: HoldOutResult, data: TrainingData, train_mask: np.ndarray) -> dict[str, Any]:
    y_test = data.y[~train_mask]
    return {
        "train_rows": held.train_rows,
        "test_rows": held.test_rows,
        "test_majority_rate": float((y_test == MAJORITY_CLASS).mean()),
        "metrics": held.metrics,
        "majority_rows_predicted_rare": held.rare_alarms,
        "test_class_counts": {c: int((y_test == c).sum()) for c in CANONICAL_CLASSES},
    }


BENCHMARK_REPORTS: dict[str, tuple[str, str]] = {
    "benchmark/leaderboard.csv": (
        "Model leaderboard",
        "Every model on grouped 5-fold CV over 3 seeds (15 folds): accuracy mean and std over "
        "folds; other metrics on pooled out-of-fold predictions per seed, averaged over seeds.",
    ),
    "benchmark/leaderboard_per_seed.csv": (
        "Leaderboard per seed",
        "Pooled out-of-fold metrics of every model for each CV seed.",
    ),
    "benchmark/ablation.csv": (
        "Feature ablation (logistic, power 0)",
        "Drop each feature group in turn, or add Col2 format features or bigrams; deltas versus "
        "the full feature set.",
    ),
    "benchmark/class_weight_dial.csv": (
        "Class weight dial",
        "Logistic and log-odds blend at class weight powers 0 to 1: CV accuracy, macro F1, "
        "out-of-time accuracy, and Category_1 rows predicted as a rare class out of time.",
    ),
    "benchmark/leaf_size_effect.csv": (
        "LightGBM minimum leaf size",
        "LightGBM with min_child_samples from 1 to 20. Rare classes need very small leaves.",
    ),
    "benchmark/validation_schemes.csv": (
        "Validation schemes",
        "Grouped CV, random CV (leakage demonstration), unseen-vendor CV, and out-of-time "
        "evaluation for logistic and blend models.",
    ),
    "benchmark/oot_detail.json": (
        "Out-of-time detail",
        "Train on periods up to 2018-09, test on 2018-10 to 2018-12: metrics, majority rate, "
        "false rare-class alarms.",
    ),
    "benchmark/unseen_pair_accuracy.csv": (
        "Accuracy by seen versus unseen vendor-account pair",
        "Out-of-fold accuracy (grouped CV, seed 0) for rows whose vendor-account pair was or "
        "was not in the fold's training rows.",
    ),
}


def run_benchmark(settings: Settings, quick: bool = False) -> list[str]:
    """Run every benchmark stage and write the reports.

    Args:
        settings: Project settings.
        quick: Reduced plan for development runs.

    Returns:
        Report paths written, relative to ``reports/``.
    """
    data = load_training_frame(settings)
    plan = plan_for(settings, quick)
    results = run_leaderboard(settings, data, plan)
    board, per_seed = leaderboard_tables(results, settings)
    schemes, oot_detail, pairs = run_validation_schemes(settings, data, plan)
    outputs: dict[str, pd.DataFrame | dict[str, Any]] = {
        "benchmark/leaderboard.csv": board,
        "benchmark/leaderboard_per_seed.csv": per_seed,
        "benchmark/ablation.csv": run_ablation(settings, data, plan),
        "benchmark/class_weight_dial.csv": run_class_weight_dial(settings, data, plan),
        "benchmark/leaf_size_effect.csv": run_leaf_size(settings, data, plan),
        "benchmark/validation_schemes.csv": schemes,
        "benchmark/oot_detail.json": oot_detail | {"quick_mode": quick, "seeds": list(plan.seeds)},
        "benchmark/unseen_pair_accuracy.csv": pairs,
    }
    return write_reports(settings, BENCHMARK_REPORTS, outputs, PRODUCER)
