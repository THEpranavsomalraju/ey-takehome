"""Logged hyperparameter searches and nested CV (finishing item B, docs/04 section 7).

Every search uses grouped 5-fold CV with the groups passed to ``fit``,
scores accuracy (the rubric metric) and logs macro_f1_5 beside it, and writes
every trial. Nested CV re-runs a reduced search inside each outer fold so the
tuned procedure gets an unbiased accuracy estimate.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from ledgerclf.config import FeatureParams, LinearParams, Settings, TreeParams
from ledgerclf.evaluation import compute_metrics
from ledgerclf.labels import FIVE_CLASSES
from ledgerclf.models import make_blend, make_lightgbm, make_logistic
from ledgerclf.reports import write_reports
from ledgerclf.splitting import (
    TrainingData,
    grouped_cv,
    load_training_frame,
    small_class_warning_suppressed,
)

logger = logging.getLogger(__name__)

PRODUCER = "tune"
TOKEN_TRANSFORMERS = ("col1_tokens", "col4_tokens", "col6_tokens")
SCORING = {
    "accuracy": "accuracy",
    "macro_f1_5": make_scorer(
        f1_score, average="macro", labels=list(FIVE_CLASSES), zero_division=0
    ),
}
TREE_PARAM_NAMES = (
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "reg_lambda",
    "colsample_bytree",
    "subsample",
)


@dataclass(frozen=True)
class SearchSpace:
    """Search grids for one run (full or quick)."""

    linear_C: tuple[float, ...]  # noqa: N815 - sklearn parameter name
    linear_powers: tuple[float, ...]
    ngram_ranges: tuple[tuple[int, int], ...]
    tree_space: dict[str, tuple[float, ...]]
    tree_trials: int
    blend_weights: tuple[float, ...]
    blend_powers: tuple[float, ...]
    nested_linear_C: tuple[float, ...]  # noqa: N815
    nested_linear_powers: tuple[float, ...]
    nested_tree_trials: int
    nested_blend_weights: tuple[float, ...]

    def to_dict(self, settings: Settings) -> dict[str, Any]:
        """Return a JSON-safe description of the space and its CV schemes."""
        seed, folds, inner = settings.random_state, settings.cv_folds, settings.nested_inner_folds
        splitter = "StratifiedGroupKFold({n}, shuffle=True, random_state={s})"
        return {
            "linear": {
                "C": list(self.linear_C),
                "power": list(self.linear_powers),
                "ngram_range": [list(n) for n in self.ngram_ranges],
            },
            "tree": {
                "space": {k: list(v) for k, v in self.tree_space.items()},
                "random_trials": self.tree_trials,
                "random_state": seed,
            },
            "blend": {"linear_weight": list(self.blend_weights), "power": list(self.blend_powers)},
            "nested": {
                "outer": splitter.format(n=folds, s=seed),
                "inner": splitter.format(n=inner, s=seed),
                "linear_C": list(self.nested_linear_C),
                "linear_power": list(self.nested_linear_powers),
                "tree_random_trials": self.nested_tree_trials,
                "blend_weight": list(self.nested_blend_weights),
            },
            "metric": "accuracy (macro_f1_5 logged beside it); trial metrics are means over "
            "folds, while select reports macro_f1_5 on pooled out-of-fold predictions",
            "cv": splitter.format(n=folds, s=seed) + " with the group key",
        }


def search_space(settings: Settings, quick: bool) -> SearchSpace:
    """Return the full search space, or a reduced one (about a quarter) for ``--quick``."""
    tuning = settings.tuning
    if quick:
        return SearchSpace(
            tuning.quick_linear_C,
            tuning.quick_powers,
            tuning.quick_ngram_ranges,
            tuning.tree_space,
            tuning.quick_tree_trials,
            tuning.quick_blend_weights,
            tuning.blend_powers,
            tuning.quick_linear_C,
            tuning.quick_powers,
            tuning.quick_nested_tree_trials,
            tuning.quick_nested_blend_weights,
        )
    return SearchSpace(
        tuning.linear_C,
        tuning.powers,
        tuning.ngram_ranges,
        tuning.tree_space,
        tuning.tree_trials,
        tuning.blend_weights,
        tuning.blend_powers,
        tuning.nested_linear_C,
        tuning.nested_linear_powers,
        tuning.nested_tree_trials,
        tuning.nested_blend_weights,
    )


def _ngram_grid(ngram: tuple[int, int]) -> dict[str, list[tuple[int, int]]]:
    return {f"features__{name}__tfidf__ngram_range": [ngram] for name in TOKEN_TRANSFORMERS}


def linear_grid(
    C_values: tuple[float, ...], powers: tuple[float, ...], ngrams: tuple[tuple[int, int], ...]
) -> list[dict[str, Any]]:
    """Return the logistic regression grid; the three token vectorizers share one n-gram range."""
    return [
        {"model__estimator__C": list(C_values), "model__power": list(powers), **_ngram_grid(ngram)}
        for ngram in ngrams
    ]


def tree_distributions(space: dict[str, tuple[float, ...]]) -> dict[str, list[float]]:
    """Return the LightGBM random-search distributions in pipeline parameter names."""
    names = {name: f"model__estimator__{name}" for name in TREE_PARAM_NAMES}
    names["power"] = "model__power"
    return {names[key]: list(values) for key, values in space.items()}


def _search(search: Any, data: TrainingData) -> Any:
    with small_class_warning_suppressed():
        return search.fit(data.frame, data.y, groups=data.groups)


def _trials(search: Any) -> pd.DataFrame:
    results = pd.DataFrame(search.cv_results_)
    params = pd.DataFrame(list(results["params"])).rename(columns=_short_name)
    params = params.loc[:, ~params.columns.duplicated()]
    table = pd.concat(
        [
            pd.DataFrame({"trial_id": np.arange(1, len(results) + 1)}),
            params.map(lambda v: str(v) if isinstance(v, tuple) else v),
            pd.DataFrame(
                {
                    "mean_accuracy": results["mean_test_accuracy"],
                    "std_accuracy": results["std_test_accuracy"],
                    "mean_macro_f1_5": results["mean_test_macro_f1_5"],
                    "std_macro_f1_5": results["std_test_macro_f1_5"],
                    "mean_fit_seconds": results["mean_fit_time"],
                }
            ),
        ],
        axis=1,
    )
    table["rank_accuracy"] = table["mean_accuracy"].rank(ascending=False, method="min").astype(int)
    return table


def _short_name(parameter: str) -> str:
    if parameter.endswith("ngram_range"):
        return "ngram_range"
    return parameter.rsplit("__", 1)[-1]


def _cv(settings: Settings, n_splits: int | None = None) -> Any:
    return grouped_cv(settings.random_state, n_splits or settings.cv_folds)


def search_linear(
    settings: Settings, data: TrainingData, space: SearchSpace, jobs: int
) -> pd.DataFrame:
    """Grid search C, class weight power, and n-gram range for logistic regression."""
    search = GridSearchCV(
        make_logistic(settings.linear),
        linear_grid(space.linear_C, space.linear_powers, space.ngram_ranges),
        scoring=SCORING,
        refit=False,
        cv=_cv(settings),
        n_jobs=jobs,
        error_score="raise",
    )
    return _trials(_search(search, data))


def search_tree(
    settings: Settings, data: TrainingData, space: SearchSpace, jobs: int, trials: int | None = None
) -> pd.DataFrame:
    """Random search over LightGBM parameters and class weight power."""
    search = RandomizedSearchCV(
        make_lightgbm(settings.tree, settings.random_state),
        tree_distributions(space.tree_space),
        n_iter=trials or space.tree_trials,
        scoring=SCORING,
        refit=False,
        cv=_cv(settings),
        n_jobs=jobs,
        random_state=settings.random_state,
        error_score="raise",
    )
    return _trials(_search(search, data))


@dataclass(frozen=True)
class TunedParams:
    """Best parameters per class weight power, as plain config objects."""

    linear_by_power: dict[float, LinearParams]
    features_by_power: dict[float, FeatureParams]
    tree: TreeParams
    blend_weight_by_power: dict[float, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict (powers as string keys)."""
        return {
            "linear_by_power": {str(p): {"C": v.C} for p, v in self.linear_by_power.items()},
            "ngram_range_by_power": {
                str(p): list(v.ngram_range) for p, v in self.features_by_power.items()
            },
            "tree": {name: getattr(self.tree, name) for name in TREE_PARAM_NAMES},
            "blend_weight_by_power": {str(p): w for p, w in self.blend_weight_by_power.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], settings: Settings) -> TunedParams:
        """Rebuild tuned parameters from ``to_dict`` output."""
        linear = {
            float(p): replace(settings.linear, C=float(v["C"]), class_weight_power=float(p))
            for p, v in payload["linear_by_power"].items()
        }
        features = {
            float(p): replace(settings.feature, ngram_range=tuple(v))
            for p, v in payload["ngram_range_by_power"].items()
        }
        tree = replace(settings.tree, **payload["tree"])
        weights = {float(p): float(w) for p, w in payload["blend_weight_by_power"].items()}
        return cls(linear, features, tree, weights)


def _best_row(trials: pd.DataFrame) -> pd.Series:
    ordered = trials.sort_values(
        ["mean_accuracy", "mean_macro_f1_5", "trial_id"], ascending=[False, False, True]
    )
    return ordered.iloc[0]


def best_linear(
    settings: Settings, trials: pd.DataFrame, power: float
) -> tuple[LinearParams, FeatureParams]:
    """Return the most accurate logistic configuration at one class weight power."""
    row = _best_row(trials[np.isclose(trials["power"], power)])
    ngram = tuple(int(x) for x in str(row["ngram_range"]).strip("()").split(","))
    linear = replace(settings.linear, C=float(row["C"]), class_weight_power=power)
    return linear, replace(settings.feature, ngram_range=(ngram[0], ngram[1]))


def best_tree(settings: Settings, trials: pd.DataFrame) -> TreeParams:
    """Return the most accurate LightGBM configuration (power is chosen later, per candidate)."""
    row = _best_row(trials)
    values = {name: type(getattr(settings.tree, name))(row[name]) for name in TREE_PARAM_NAMES}
    return replace(settings.tree, **values)


def search_blend(
    settings: Settings,
    data: TrainingData,
    space: SearchSpace,
    linear_trials: pd.DataFrame,
    tree: TreeParams,
    jobs: int,
) -> pd.DataFrame:
    """Grid search the linear weight of the log-odds blend at each class weight power."""
    tables = []
    for power in space.blend_powers:
        linear, features = best_linear(settings, linear_trials, power)
        member_tree = replace(tree, class_weight_power=power)
        estimator = make_blend(linear, member_tree, settings.blend, settings.random_state, features)
        search = GridSearchCV(
            estimator,
            {"model__linear_weight": list(space.blend_weights)},
            scoring=SCORING,
            refit=False,
            cv=_cv(settings),
            n_jobs=jobs,
            error_score="raise",
        )
        tables.append(
            _trials(_search(search, data)).assign(
                power=power, C=linear.C, ngram_range=str(features.ngram_range)
            )
        )
    table = pd.concat(tables, ignore_index=True)
    table["trial_id"] = np.arange(1, len(table) + 1)
    table["rank_accuracy"] = table["mean_accuracy"].rank(ascending=False, method="min").astype(int)
    return table


def tuned_params(
    settings: Settings,
    linear_trials: pd.DataFrame,
    tree_trials: pd.DataFrame,
    blend_trials: pd.DataFrame,
) -> TunedParams:
    """Collect the best configuration per power from the three trial logs."""
    powers = sorted(set(blend_trials["power"]))
    linear, features = {}, {}
    for power in powers:
        linear[power], features[power] = best_linear(settings, linear_trials, power)
    weights = {
        p: float(_best_row(blend_trials[blend_trials["power"] == p])["linear_weight"])
        for p in powers
    }
    return TunedParams(linear, features, best_tree(settings, tree_trials), weights)


def nested_cv(
    settings: Settings, data: TrainingData, space: SearchSpace, jobs: int
) -> dict[str, Any]:
    """Estimate the tuned blend procedure without optimism: search inside each outer fold.

    Outer: primary grouped 5-fold (seed 0). Inner: grouped 3-fold searches for
    the logistic (C x power), LightGBM (random configs), and blend weight.
    """
    with small_class_warning_suppressed():
        outer = list(_cv(settings).split(np.zeros(len(data.y)), data.y, data.groups))
    rows = []
    for fold, (train_idx, test_idx) in enumerate(outer):
        started = time.perf_counter()
        inner = _subset(data, train_idx)
        inner_settings = replace(settings, cv_folds=settings.nested_inner_folds)
        linear_trials = search_linear(
            inner_settings,
            inner,
            replace(
                space,
                linear_C=space.nested_linear_C,
                linear_powers=space.nested_linear_powers,
                ngram_ranges=settings.tuning.nested_ngram_ranges,
            ),
            jobs,
        )
        power = float(_best_row(linear_trials)["power"])
        tree_trials = search_tree(
            inner_settings, inner, space, jobs, trials=space.nested_tree_trials
        )
        tree = best_tree(settings, tree_trials)
        blend_space = replace(
            space, blend_weights=space.nested_blend_weights, blend_powers=(power,)
        )
        blend_trials = search_blend(inner_settings, inner, blend_space, linear_trials, tree, jobs)
        best = _best_row(blend_trials)
        linear, features = best_linear(settings, linear_trials, power)
        blend = replace(settings.blend, linear_weight=float(best["linear_weight"]))
        model = make_blend(
            linear, replace(tree, class_weight_power=power), blend, settings.random_state, features
        )
        with small_class_warning_suppressed():
            model.fit(inner.frame, inner.y)
        metrics = compute_metrics(data.y[test_idx], model.predict(data.frame.iloc[test_idx]))
        rows.append(
            {
                "outer_fold": fold,
                "chosen_power": power,
                "chosen_C": linear.C,
                "chosen_weight": blend.linear_weight,
                "inner_best_accuracy": float(best["mean_accuracy"]),
                "outer_accuracy": metrics["accuracy"],
                "outer_macro_f1_5": metrics["macro_f1_5"],
                "seconds": time.perf_counter() - started,
            }
        )
        logger.info("nested fold %d: outer accuracy %.4f", fold, metrics["accuracy"])
    table = pd.DataFrame(rows)
    return {
        "outer_folds": rows,
        "outer_accuracy_mean": float(table["outer_accuracy"].mean()),
        "outer_accuracy_std": float(table["outer_accuracy"].std(ddof=1)),
        "outer_macro_f1_5_mean": float(table["outer_macro_f1_5"].mean()),
        "outer_macro_f1_5_std": float(table["outer_macro_f1_5"].std(ddof=1)),
    }


def _subset(data: TrainingData, rows: np.ndarray) -> TrainingData:
    return TrainingData(
        data.frame.iloc[rows].reset_index(drop=True),
        data.y[rows],
        data.groups[rows],
        data.raw.iloc[rows].reset_index(drop=True),
    )


TUNING_REPORTS: dict[str, tuple[str, str]] = {
    "tuning/search_space.json": (
        "Search space",
        "Every searched hyperparameter, its values, the trial budget, and the CV scheme.",
    ),
    "tuning/linear_trials.csv": (
        "Logistic regression trials",
        "Grid search over C, class weight power, and n-gram range (grouped 5-fold CV).",
    ),
    "tuning/tree_trials.csv": (
        "LightGBM trials",
        "Random search over LightGBM parameters and class weight power (grouped 5-fold CV).",
    ),
    "tuning/blend_trials.csv": (
        "Blend weight trials",
        "Linear weight of the log-odds blend at each class weight power, with the best members.",
    ),
    "tuning/nested_cv.json": (
        "Nested CV",
        "Unbiased accuracy of the whole tuning procedure: searches rerun inside each outer fold.",
    ),
    "tuning/best_params.json": (
        "Best parameters",
        "Best configuration per class weight power; input to the select command.",
    ),
}


def run_tuning(settings: Settings, quick: bool = False) -> list[str]:
    """Run every search and nested CV, then write the tuning reports.

    Args:
        settings: Project settings.
        quick: Reduced budgets for development; final reports come from a full run.

    Returns:
        Report paths written, relative to ``reports/``.
    """
    data = load_training_frame(settings)
    space = search_space(settings, quick)
    jobs = settings.benchmark.cv_jobs
    linear_trials = search_linear(settings, data, space, jobs)
    logger.info("linear search done")
    tree_trials = search_tree(settings, data, space, jobs)
    logger.info("tree search done")
    blend_trials = search_blend(
        settings, data, space, linear_trials, best_tree(settings, tree_trials), jobs
    )
    best = tuned_params(settings, linear_trials, tree_trials, blend_trials)
    nested = nested_cv(settings, data, space, jobs)
    non_nested = float(blend_trials["mean_accuracy"].max())
    nested |= {
        "non_nested_best_accuracy": non_nested,
        "optimism_gap": non_nested - nested["outer_accuracy_mean"],
        "quick_mode": quick,
    }
    outputs: dict[str, pd.DataFrame | dict[str, Any]] = {
        "tuning/search_space.json": space.to_dict(settings) | {"quick_mode": quick},
        "tuning/linear_trials.csv": linear_trials,
        "tuning/tree_trials.csv": tree_trials,
        "tuning/blend_trials.csv": blend_trials,
        "tuning/nested_cv.json": nested,
        "tuning/best_params.json": best.to_dict() | {"quick_mode": quick},
    }
    return write_reports(settings, TUNING_REPORTS, outputs, PRODUCER)
