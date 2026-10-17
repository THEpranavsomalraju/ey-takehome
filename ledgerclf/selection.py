"""Champion selection, calibration analysis, thresholds, and flag policy (items A and D).

``select`` scores the four pre-registered candidates with their tuned
parameters, applies the decision rule recorded in DECISIONS.md D11 before
this code ran, calibrates the champion and the fallback with temperature
scaling on out-of-fold logits, picks the review threshold by rule (D17),
measures accuracy per flag with fold-specific reference statistics (D18),
and writes ``artifacts/champion.json`` for ``train``.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from ledgerclf.calibration import (
    brier_score,
    choose_threshold,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    reliability_table,
    threshold_table,
)
from ledgerclf.config import DecisionRule, FeatureParams, Settings
from ledgerclf.evaluation import (
    CVResult,
    confusion_table,
    grouped_splits,
    per_class_report,
    run_cv,
    run_train_test,
)
from ledgerclf.exceptions import ConfigError
from ledgerclf.flags import (
    DEFAULT_SEVERITIES,
    LOW_CONFIDENCE,
    FlagSeverity,
    ReviewPolicy,
    apply_review_policy,
    compute_row_flags,
)
from ledgerclf.labels import CANONICAL_CLASSES
from ledgerclf.models import make_blend, make_logistic, make_probability_blend, softmax
from ledgerclf.reference import build_reference
from ledgerclf.reports import write_json, write_table
from ledgerclf.schema import AMOUNT, COL_ACCOUNT, COL_DESCRIPTION, COL_PERIOD, COL_VENDOR
from ledgerclf.splitting import ROW_ID, TrainingData, load_training_frame, out_of_time_mask
from ledgerclf.tuning import TunedParams

logger = logging.getLogger(__name__)

PRODUCER = "select"
CHAMPION_FILE = "champion.json"
CANDIDATES: tuple[tuple[str, float], ...] = (
    ("logistic", 0.0),
    ("logistic", 0.25),
    ("logodds_blend", 0.0),
    ("logodds_blend", 0.25),
)
FAMILY_ORDER = {"logistic": 0, "logodds_blend": 1}  # simpler first, for the final tie-break


@dataclass
class CandidateScore:
    """Scores of one candidate under the decision rule."""

    name: str
    family: str
    power: float
    cv_accuracy: float
    cv_accuracy_std: float
    macro_f1_5: float
    oot_accuracy: float
    oot_macro_f1_5: float
    oot_rare_alarms: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict."""
        return dict(self.__dict__)


@dataclass
class ChampionDecision:
    """Outcome of the decision rule with its step-by-step trail."""

    champion: str
    eligible: list[str]
    trail: list[str]


def candidate_name(family: str, power: float) -> str:
    """Return the report name of a candidate, for example ``logodds_blend_p0.25``."""
    return f"{family}_p{power:g}"


def build_candidate(settings: Settings, tuned: TunedParams, family: str, power: float) -> Any:
    """Return the unfitted pipeline of one candidate with its tuned parameters."""
    linear = replace(tuned.linear_by_power[power], class_weight_power=power)
    features = tuned.features_by_power[power]
    if family == "logistic":
        return make_logistic(linear, features)
    tree = replace(tuned.tree, class_weight_power=power)
    blend = replace(
        settings.blend, linear_weight=tuned.blend_weight_by_power[power], temperature=1.0
    )
    return make_blend(linear, tree, blend, settings.random_state, features)


def score_candidates(
    settings: Settings, data: TrainingData, tuned: TunedParams
) -> tuple[list[CandidateScore], dict[str, CVResult]]:
    """Score every candidate on primary CV (3 seeds) and out of time."""
    splits = grouped_splits(data.y, data.groups, settings.seeds, settings.cv_folds)
    oot = out_of_time_mask(data.frame, settings.oot_cutoff)
    scores, results = [], {}
    for family, power in CANDIDATES:
        name = candidate_name(family, power)
        estimator = build_candidate(settings, tuned, family, power)
        result = run_cv(name, estimator, data.frame, data.y, splits, settings.benchmark.cv_jobs)
        held = run_train_test(estimator, data.frame, data.y, oot)
        results[name] = result
        scores.append(
            CandidateScore(
                name=name,
                family=family,
                power=power,
                cv_accuracy=result.summary["accuracy_mean"],
                cv_accuracy_std=result.summary["accuracy_std"],
                macro_f1_5=result.summary["macro_f1_5_seed_mean"],
                oot_accuracy=held.metrics["accuracy"],
                oot_macro_f1_5=held.metrics["macro_f1_5"],
                oot_rare_alarms=held.rare_alarms,
            )
        )
        logger.info("candidate %s: cv %.4f", name, scores[-1].cv_accuracy)
    return scores, results


def apply_decision_rule(candidates: list[CandidateScore], rule: DecisionRule) -> ChampionDecision:
    """Apply the pre-registered champion rule (docs/04 section 8, DECISIONS.md D11).

    Args:
        candidates: Scored candidates.
        rule: Tolerances of the rule.

    Returns:
        The champion name, the eligible set, and the decision trail.
    """
    best_f1 = max(c.macro_f1_5 for c in candidates)
    best_oot = max(c.oot_accuracy for c in candidates)
    trail = [f"best macro_f1_5 = {best_f1:.4f}; best out-of-time accuracy = {best_oot:.4f}"]
    eligible = [
        c
        for c in candidates
        if c.macro_f1_5 >= best_f1 - rule.macro_f1_tolerance
        and c.oot_accuracy >= best_oot - rule.oot_tolerance
    ]
    for c in candidates:
        status = "eligible" if c in eligible else "excluded"
        trail.append(
            f"{c.name}: cv {c.cv_accuracy:.4f}, macro_f1_5 {c.macro_f1_5:.4f}, "
            f"oot {c.oot_accuracy:.4f} -> {status}"
        )
    ranked = sorted(eligible, key=lambda c: c.cv_accuracy, reverse=True)
    if len(ranked) == 1 or ranked[0].cv_accuracy - ranked[1].cv_accuracy >= rule.tie_margin:
        trail.append(
            f"{ranked[0].name} leads on CV accuracy by at least {rule.tie_margin} "
            "(or is the only eligible candidate): chosen"
        )
        return ChampionDecision(ranked[0].name, [c.name for c in eligible], trail)
    first, second = ranked[0], ranked[1]
    trail.append(
        f"{first.name} and {second.name} within {rule.tie_margin} on CV accuracy: "
        "compare out-of-time accuracy"
    )
    if abs(first.oot_accuracy - second.oot_accuracy) >= rule.tie_margin:
        winner = max((first, second), key=lambda c: c.oot_accuracy)
        trail.append(f"{winner.name} has the higher out-of-time accuracy: chosen")
    else:
        winner = min((first, second), key=lambda c: (FAMILY_ORDER[c.family], c.power))
        trail.append(
            f"out-of-time also within {rule.tie_margin}: the simpler {winner.name} is chosen"
        )
    return ChampionDecision(winner.name, [c.name for c in eligible], trail)


def fallback_name(candidates: list[CandidateScore]) -> str:
    """Return the best logistic candidate (the fallback when LightGBM cannot load)."""
    logistic = [c for c in candidates if c.family == "logistic"]
    return max(logistic, key=lambda c: (c.cv_accuracy, -c.power)).name


@dataclass
class CalibrationOutcome:
    """Temperature and before/after calibration metrics for one model."""

    temperature: float
    summary: dict[str, float]
    reliability: pd.DataFrame
    calibrated_proba: np.ndarray


def calibrate(result: CVResult, y: np.ndarray, settings: Settings, seed: int) -> CalibrationOutcome:
    """Fit the temperature on out-of-fold logits of one seed and report before and after."""
    if seed not in result.oof_logits:
        raise ConfigError(
            f"No out-of-fold logits for {result.model_name}; cannot calibrate.", code="NO_LOGITS"
        )
    logits = result.oof_logits[seed]
    params = settings.calibration
    temperature = fit_temperature(logits, y, params.temperature_bounds)
    before, after = softmax(logits), softmax(logits / temperature)
    summary = {"temperature": temperature}
    for stage, proba in (("before", before), ("after", after)):
        summary[f"ece_{stage}"] = expected_calibration_error(proba, y, params.n_bins)
        summary[f"brier_{stage}"] = brier_score(proba, y)
        summary[f"log_loss_{stage}"] = negative_log_likelihood(proba, y)
    tables = [
        reliability_table(proba, y, params.n_bins).assign(model=result.model_name, stage=stage)
        for stage, proba in (("before", before), ("after", after))
    ]
    return CalibrationOutcome(temperature, summary, pd.concat(tables, ignore_index=True), after)


def fold_flags(data: TrainingData, result: CVResult, seed: int) -> pd.DataFrame:
    """Compute row flags for out-of-fold rows against each fold's own training rows."""
    folds = result.fold_of_row[seed]
    parts = []
    for fold in sorted(set(folds)):
        test = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        reference = build_reference(data.frame.iloc[train], data.y[train])
        parts.append(compute_row_flags(data.frame.iloc[test], reference).set_index(test))
    return pd.concat(parts).sort_index()


def accuracy_by_flag(flags: pd.DataFrame, correct: np.ndarray) -> pd.DataFrame:
    """Return rows and out-of-fold accuracy for each flag, plus flagged and unflagged totals."""
    rows = [
        {
            "flag": code,
            "rows": int(flags[code].sum()),
            "accuracy": float(correct[flags[code].to_numpy()].mean())
            if flags[code].any()
            else np.nan,
        }
        for code in flags.columns
    ]
    any_flag = flags.to_numpy().any(axis=1)
    rows.append(
        {
            "flag": "(any flag)",
            "rows": int(any_flag.sum()),
            "accuracy": float(correct[any_flag].mean()) if any_flag.any() else np.nan,
        }
    )
    rows.append(
        {
            "flag": "(no flag)",
            "rows": int((~any_flag).sum()),
            "accuracy": float(correct[~any_flag].mean()),
        }
    )
    rows.append({"flag": "(all rows)", "rows": len(correct), "accuracy": float(correct.mean())})
    return pd.DataFrame(rows)


def evidence_policy(
    table: pd.DataFrame, threshold: float, settings: Settings
) -> tuple[ReviewPolicy, list[str]]:
    """Set flag severities from out-of-fold evidence (rule in config.FlagPolicyParams, D18)."""
    params = settings.flag_policy
    overall = float(table.loc[table["flag"] == "(all rows)", "accuracy"].item())
    severities, notes = dict(DEFAULT_SEVERITIES), []
    for _, row in table[~table["flag"].str.startswith("(")].iterrows():
        code = str(row["flag"])
        if code == LOW_CONFIDENCE or DEFAULT_SEVERITIES[code] is FlagSeverity.INFO:
            continue
        if row["rows"] < params.min_rows:
            notes.append(
                f"{code}: {int(row['rows'])} rows, below {params.min_rows}: "
                f"keeps default {DEFAULT_SEVERITIES[code]}"
            )
            continue
        critical = row["accuracy"] < overall - params.critical_gap
        severities[code] = FlagSeverity.CRITICAL if critical else FlagSeverity.WARNING
        notes.append(
            f"{code}: accuracy {row['accuracy']:.3f} on {int(row['rows'])} rows "
            f"vs {overall:.3f} overall -> {severities[code]}"
        )
    return ReviewPolicy(threshold=threshold, severities=severities), notes


def high_confidence_errors(data: TrainingData, proba: np.ndarray, threshold: float) -> pd.DataFrame:
    """Return out-of-fold errors made with calibrated confidence at or above ``threshold``."""
    predicted = np.asarray(CANONICAL_CLASSES, dtype=object)[proba.argmax(axis=1)]
    confidence = proba.max(axis=1)
    mask = (predicted != data.y) & (confidence >= threshold)
    columns = [ROW_ID, COL_VENDOR, COL_DESCRIPTION, COL_ACCOUNT, COL_PERIOD, AMOUNT]
    table = data.frame.loc[mask, columns].copy()
    table["true_label"] = data.y[mask]
    table["predicted_label"] = predicted[mask]
    table["confidence"] = confidence[mask]
    return table.sort_values("confidence", ascending=False).reset_index(drop=True)


def blend_comparison(
    settings: Settings, data: TrainingData, tuned: TunedParams, power: float
) -> pd.DataFrame:
    """Compare the log-odds blend with probability averaging under the same tuned members."""
    splits = grouped_splits(data.y, data.groups, settings.seeds, settings.cv_folds)
    linear = replace(tuned.linear_by_power[power], class_weight_power=power)
    tree = replace(tuned.tree, class_weight_power=power)
    models = {
        "logodds_blend": build_candidate(settings, tuned, "logodds_blend", power),
        "probability_blend": make_probability_blend(linear, tree, settings.random_state),
    }
    rows = []
    for name, estimator in models.items():
        result = run_cv(name, estimator, data.frame, data.y, splits, settings.benchmark.cv_jobs)
        rows.append(
            {
                "model": name,
                "power": power,
                "cv_accuracy": result.summary["accuracy_mean"],
                "cv_accuracy_std": result.summary["accuracy_std"],
                "macro_f1_5": result.summary["macro_f1_5_seed_mean"],
            }
        )
    table = pd.DataFrame(rows)
    table["accuracy_gap_vs_probability_blend"] = (
        table["cv_accuracy"]
        - table.loc[table["model"] == "probability_blend", "cv_accuracy"].item()
    )
    return table


def load_tuned(settings: Settings) -> tuple[TunedParams, bool]:
    """Read the best parameters written by ``tune``.

    Raises:
        ConfigError: When the tuning reports do not exist yet.
    """
    path = settings.paths.reports / "tuning" / "best_params.json"
    if not path.exists():
        raise ConfigError(
            "Tuning results are missing. Run: python -m ledgerclf tune", code="TUNING_MISSING"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return TunedParams.from_dict(payload, settings), bool(payload.get("quick_mode", False))


def champion_config(
    tuned: TunedParams, family: str, power: float, settings: Settings
) -> dict[str, Any]:
    """Return the full configuration of a candidate for champion.json."""
    linear = tuned.linear_by_power[power]
    features: FeatureParams = tuned.features_by_power[power]
    config: dict[str, Any] = {
        "family": family,
        "class_weight_power": power,
        "linear": {"C": linear.C, "max_iter": linear.max_iter, "solver": linear.solver},
        "features": {"ngram_range": list(features.ngram_range)},
    }
    if family == "logodds_blend":
        config["tree"] = tuned.to_dict()["tree"]
        config["linear_weight"] = tuned.blend_weight_by_power[power]
    config["random_state"] = settings.random_state
    return config


SELECTION_REPORTS: dict[str, tuple[str, str]] = {
    "selection/champion_candidates.csv": (
        "Champion candidates",
        "The four pre-registered candidates with tuned parameters: grouped CV accuracy (3 seeds, "
        "15 folds), macro F1 over five classes, and out-of-time accuracy.",
    ),
    "selection/champion_decision.json": (
        "Champion decision",
        "The decision rule of DECISIONS.md D11 applied step by step, with the chosen champion "
        "and fallback.",
    ),
    "selection/blend_comparison.csv": (
        "Log-odds versus probability blend",
        "The same tuned members blended in logit space (exact explanations) or by averaging "
        "probabilities.",
    ),
    "selection/oof_confusion_matrix.csv": (
        "Out-of-fold confusion matrix",
        "Champion, grouped CV seed 0: true class (rows) versus predicted class (columns).",
    ),
    "selection/oof_classification_report.csv": (
        "Out-of-fold classification report",
        "Champion, grouped CV seed 0: precision, recall, F1, and support per class.",
    ),
    "selection/calibration_reliability.csv": (
        "Reliability before and after temperature scaling",
        "Confidence bins with mean confidence and accuracy, for champion and fallback.",
    ),
    "selection/calibration_summary.json": (
        "Calibration summary",
        "Temperatures, ECE, Brier score, and log loss before and after temperature scaling.",
    ),
    "selection/threshold_table.csv": (
        "Confidence threshold table",
        "Coverage and accuracy when rows below each calibrated confidence threshold go to "
        "review; the chosen threshold follows D17.",
    ),
    "selection/accuracy_by_flag.csv": (
        "Accuracy by flag",
        "Out-of-fold accuracy of rows carrying each risk flag, with fold-specific reference "
        "statistics; sets flag severities (D18).",
    ),
    "selection/high_confidence_errors.csv": (
        "High-confidence errors",
        "Out-of-fold errors made with calibrated confidence of 0.9 or more: candidates for "
        "client label review (D21).",
    ),
}


def _write_reports(settings: Settings, outputs: dict[str, pd.DataFrame | dict[str, Any]]) -> None:
    for file, (title, description) in SELECTION_REPORTS.items():
        result = outputs[file]
        if isinstance(result, pd.DataFrame):
            write_table(settings, file, result, title, PRODUCER, description)
        else:
            write_json(settings, file, result, title, PRODUCER, description)


def _review_summary(
    flags: pd.DataFrame, confidence: np.ndarray, correct: np.ndarray, policy: ReviewPolicy
) -> dict[str, float]:
    _, review = apply_review_policy(flags, confidence, policy)
    return {
        "review_share": float(review.mean()),
        "accuracy_reviewed": float(correct[review].mean()) if review.any() else float("nan"),
        "accuracy_auto": float(correct[~review].mean()) if (~review).any() else float("nan"),
    }


def run_selection(settings: Settings) -> list[str]:
    """Run the champion decision, calibration, threshold, and flag policy; write champion.json.

    Args:
        settings: Project settings.

    Returns:
        Report paths written, relative to ``reports/``.
    """
    tuned, quick = load_tuned(settings)
    data = load_training_frame(settings)
    scores, results = score_candidates(settings, data, tuned)
    decision = apply_decision_rule(scores, settings.decision_rule)
    fallback = fallback_name(scores)
    seed = settings.seeds[0]
    champion_cal = calibrate(results[decision.champion], data.y, settings, seed)
    fallback_cal = calibrate(results[fallback], data.y, settings, seed)
    proba = champion_cal.calibrated_proba
    confidence = proba.max(axis=1)
    correct = np.asarray(CANONICAL_CLASSES, dtype=object)[proba.argmax(axis=1)] == data.y
    thresholds = threshold_table(proba, data.y, settings.calibration.threshold_candidates)
    threshold = choose_threshold(
        thresholds,
        settings.calibration.target_kept_accuracy,
        settings.calibration.default_threshold,
    )
    row_flags = fold_flags(data, results[decision.champion], seed)
    all_flags, _ = apply_review_policy(row_flags, confidence, ReviewPolicy(threshold=threshold))
    flag_table = accuracy_by_flag(all_flags, correct)
    policy, notes = evidence_policy(flag_table, threshold, settings)
    by_name = {s.name: s for s in scores}
    champion = by_name[decision.champion]
    comparison_power = champion.power if champion.family == "logodds_blend" else 0.0
    review = _review_summary(row_flags, confidence, correct, policy)
    write_champion(
        settings, tuned, decision, by_name, fallback, champion_cal, fallback_cal, policy, quick
    )
    _write_reports(
        settings,
        {
            "selection/champion_candidates.csv": pd.DataFrame([s.to_dict() for s in scores]),
            "selection/champion_decision.json": {
                "champion": decision.champion,
                "fallback": fallback,
                "eligible": decision.eligible,
                "trail": decision.trail,
                "rule": settings.decision_rule.__dict__,
                "tuning_quick_mode": quick,
            },
            "selection/blend_comparison.csv": blend_comparison(
                settings, data, tuned, comparison_power
            ),
            "selection/oof_confusion_matrix.csv": confusion_table(
                data.y, results[decision.champion].oof_pred[seed]
            ),
            "selection/oof_classification_report.csv": per_class_report(
                data.y, results[decision.champion].oof_pred[seed]
            ),
            "selection/calibration_reliability.csv": pd.concat(
                [champion_cal.reliability, fallback_cal.reliability], ignore_index=True
            ),
            "selection/calibration_summary.json": {
                "champion": decision.champion,
                "fallback": fallback,
                decision.champion: champion_cal.summary,
                fallback: fallback_cal.summary,
                "ece_target": settings.calibration.ece_target,
                "seed": seed,
            },
            "selection/threshold_table.csv": thresholds.assign(
                chosen=thresholds["threshold"] == threshold
            ),
            "selection/accuracy_by_flag.csv": flag_table.assign(
                severity=flag_table["flag"].map(lambda f: str(policy.severities.get(f, "")))
            ),
            "selection/high_confidence_errors.csv": high_confidence_errors(
                data, proba, settings.flag_policy.high_confidence_error
            ),
        },
    )
    logger.info("flag policy: %s; review share %.3f", "; ".join(notes), review["review_share"])
    return list(SELECTION_REPORTS)


def write_champion(
    settings: Settings,
    tuned: TunedParams,
    decision: ChampionDecision,
    scores: dict[str, CandidateScore],
    fallback: str,
    champion_cal: CalibrationOutcome,
    fallback_cal: CalibrationOutcome,
    policy: ReviewPolicy,
    quick: bool,
) -> None:
    """Write artifacts/champion.json: the configuration ``train`` must use."""
    champion, backup = scores[decision.champion], scores[fallback]
    payload = {
        "created_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "champion": {
            "name": champion.name,
            **champion_config(tuned, champion.family, champion.power, settings),
            "temperature": champion_cal.temperature,
            "scores": champion.to_dict(),
        },
        "fallback": {
            "name": backup.name,
            **champion_config(tuned, backup.family, backup.power, settings),
            "temperature": fallback_cal.temperature,
            "scores": backup.to_dict(),
        },
        "review_policy": policy.to_dict(),
        "candidates": [s.to_dict() for s in scores.values()],
        "decision_trail": decision.trail,
        "tuning_quick_mode": quick,
    }
    settings.paths.artifacts.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    (settings.paths.artifacts / CHAMPION_FILE).write_text(text + "\n", encoding="utf-8")
