"""Project configuration: paths, seeds, limits, and every tunable number.

All constants live here (CLAUDE.md rule 4). Values selected by the ``select``
command are written to ``artifacts/champion.json`` and override the model
defaults below at training time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

SUPPORTED_PYTHON = ((3, 11), (3, 14))
FORCE_FALLBACK_ENV = "LEDGERCLF_FORCE_FALLBACK"

# Exit codes shared by the CLI and the doctor (docs/02 section 10).
EXIT_OK = 0
EXIT_USER_ERROR = 1
EXIT_ENVIRONMENT_ERROR = 2
EXIT_INTERNAL_ERROR = 3

# Excel stores dates as day counts from 1899-12-30; numbers in this range read as dates.
EXCEL_SERIAL_RANGE = (20_000, 80_000)
VALID_YEARS = (1900, 2200)
MONTHS_PER_YEAR = 12
UTF16_SAMPLE_BYTES = 400  # BOM-less UTF-16 sniffing: bytes inspected
UTF16_NUL_SHARE = 0.9  # share of NUL bytes in the high-byte positions of ASCII text
UTF16_TEXT_SHARE = 0.1  # maximum NUL share in the character positions

HOLDOUT_FEATURES_STEM = "holdout_features"


@dataclass(frozen=True)
class Paths:
    """Filesystem locations, all resolved from the package location."""

    root: Path
    raw_data: Path
    splits: Path
    holdout: Path
    artifacts: Path
    reports: Path
    resources: Path

    @classmethod
    def from_root(cls, root: Path) -> Paths:
        """Build every project path from a project root directory."""
        return cls(
            root=root,
            raw_data=root / "data" / "raw" / "Challenge_Data.xls",
            splits=root / "data" / "splits",
            holdout=root / "data" / "holdout",
            artifacts=root / "artifacts",
            reports=root / "reports",
            resources=root / "ledgerclf" / "resources",
        )


@dataclass(frozen=True)
class Limits:
    """Input limits applied to every uploaded or training file."""

    max_file_bytes: int = 50 * 1024 * 1024
    max_rows: int = 200_000
    max_text_chars: int = 1_000
    header_scan_rows: int = 20
    header_min_matches: int = 4
    predict_chunk_rows: int = 20_000
    csv_sniff_bytes: int = 64 * 1024
    csv_sniff_lines: int = 50
    binary_sniff_bytes: int = 4096
    header_max_chars: int = 200
    message_value_chars: int = 40
    message_detail_chars: int = 160
    issue_rows_shown: int = 10


@dataclass(frozen=True)
class FeatureParams:
    """Feature pipeline parameters (docs/04 section 3)."""

    token_pattern: str = r"(?u)\S+"
    ngram_range: tuple[int, int] = (1, 1)
    sublinear_tf: bool = True
    min_df: int = 1
    whole_value_min_frequency: int = 2
    pair_min_frequency: int = 2
    pair_separator: str = " || "
    round_amount_modulus: float = 10.0


@dataclass(frozen=True)
class LinearParams:
    """Multinomial logistic regression parameters."""

    C: float = 10.0
    max_iter: int = 5_000
    solver: str = "lbfgs"
    class_weight_power: float = 0.0


@dataclass(frozen=True)
class TreeParams:
    """LightGBM parameters (docs/04 section 4, ``lightgbm_p0``)."""

    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = 8
    min_child_samples: int = 2
    reg_lambda: float = 2.0
    colsample_bytree: float = 0.5
    subsample: float = 0.8
    subsample_freq: int = 1
    class_weight_power: float = 0.0


@dataclass(frozen=True)
class BlendParams:
    """Log-odds blend parameters (docs/04 section 6)."""

    linear_weight: float = 0.5
    temperature: float = 1.0


@dataclass(frozen=True)
class HoldoutRule:
    """Written holdout seed rule (DECISIONS.md D03, docs/04 section 2.2)."""

    n_splits: int = 10
    max_seed: int = 100
    min_rows: int = 580
    max_rows: int = 600
    required_classes: tuple[str, ...] = ("Category_3", "Category_4", "Category_6")
    excluded_classes: tuple[str, ...] = ("Category_5",)
    share_classes: tuple[str, ...] = ("Category_1", "Category_2")
    share_tolerance: float = 0.01


@dataclass(frozen=True)
class DecisionRule:
    """Pre-registered champion decision rule (DECISIONS.md D11, docs/04 section 8)."""

    macro_f1_tolerance: float = 0.03
    oot_tolerance: float = 0.015
    tie_margin: float = 0.002


@dataclass(frozen=True)
class CalibrationParams:
    """Temperature scaling and confidence threshold settings (docs/04 section 9)."""

    temperature_bounds: tuple[float, float] = (0.05, 20.0)
    n_bins: int = 10
    ece_target: float = 0.03
    threshold_candidates: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99)
    target_kept_accuracy: float = 0.975
    default_threshold: float = 0.9
    probability_floor: float = 1e-15


@dataclass(frozen=True)
class EdaParams:
    """Sizes of the EDA and preprocessing report tables."""

    top_values: int = 20
    histogram_bins: int = 30
    iqr_multiplier: float = 1.5
    amount_deciles: int = 10
    sample_rows: int = 5


@dataclass(frozen=True)
class FlagPolicyParams:
    """Evidence rule for flag severities (DECISIONS.md D18).

    A flag becomes critical when out-of-fold accuracy on its rows falls more
    than ``critical_gap`` below the overall out-of-fold accuracy, measured on
    at least ``min_rows`` rows. Flags with fewer rows keep their default.
    """

    critical_gap: float = 0.05
    min_rows: int = 10
    high_confidence_error: float = 0.9


@dataclass(frozen=True)
class ExplainParams:
    """Explanation payload settings (docs/04 section 13)."""

    top_features: int = 8
    evidence_tokens: int = 3
    display_decimals: int = 4
    amount_quantile_points: int = 101
    absent_class_logit: float = -1e9


@dataclass(frozen=True)
class BenchmarkParams:
    """Benchmark and validation-scheme settings (docs/04 section 4)."""

    svc_C: float = 1.0  # noqa: N815 - sklearn parameter name
    svc_power: float = 0.5
    nb_alpha: float = 0.1
    leaf20_tree: TreeParams = field(
        default_factory=lambda: TreeParams(
            n_estimators=300,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            reg_lambda=0.0,
            colsample_bytree=1.0,
            subsample=1.0,
            subsample_freq=0,
        )
    )
    dial_powers: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    leaf_sizes: tuple[int, ...] = (1, 2, 5, 10, 20)
    quick_dial_powers: tuple[float, ...] = (0.0, 0.5, 1.0)
    quick_leaf_sizes: tuple[int, ...] = (2, 20)
    cv_jobs: int = 5
    svc_max_iter: int = 10_000


@dataclass(frozen=True)
class TuningParams:
    """Hyperparameter search spaces and budgets (docs/04 section 7)."""

    linear_C: tuple[float, ...] = (1.0, 3.0, 10.0, 30.0, 100.0)  # noqa: N815 - sklearn name
    powers: tuple[float, ...] = (0.0, 0.25, 0.5)
    ngram_ranges: tuple[tuple[int, int], ...] = ((1, 1), (1, 2))
    tree_space: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: {
            "n_estimators": (300, 500, 800),
            "learning_rate": (0.03, 0.05, 0.1),
            "num_leaves": (15, 31, 63),
            "max_depth": (-1, 6, 8, 12),
            "min_child_samples": (1, 2, 3, 5, 10, 20),
            # No 0: without L2 a one-row class gets exploding leaf values (D29).
            "reg_lambda": (1.0, 2.0, 5.0),
            "colsample_bytree": (0.3, 0.5, 0.8),
            "subsample": (0.7, 0.8, 1.0),
            "power": (0.0, 0.25),
        }
    )
    tree_trials: int = 40
    blend_weights: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
    blend_powers: tuple[float, ...] = (0.0, 0.25)
    nested_linear_C: tuple[float, ...] = (1.0, 3.0, 10.0, 30.0, 100.0)  # noqa: N815
    nested_linear_powers: tuple[float, ...] = (0.0, 0.25, 0.5)
    nested_tree_trials: int = 15
    nested_blend_weights: tuple[float, ...] = (0.4, 0.5, 0.6)
    quick_linear_C: tuple[float, ...] = (3.0, 10.0, 30.0)  # noqa: N815
    quick_powers: tuple[float, ...] = (0.0, 0.25)
    quick_tree_trials: int = 8
    quick_blend_weights: tuple[float, ...] = (0.4, 0.5, 0.6)
    quick_nested_tree_trials: int = 3
    quick_ngram_ranges: tuple[tuple[int, int], ...] = ((1, 1),)
    nested_ngram_ranges: tuple[tuple[int, int], ...] = ((1, 1),)
    quick_nested_blend_weights: tuple[float, ...] = (0.5,)


@dataclass(frozen=True)
class AgentParams:
    """Explanation agent settings (DECISIONS.md D25).

    The relay forwards requests to Claude with the project owner's key held
    server-side, so graders need no key. A grader's own ``ANTHROPIC_API_KEY``
    takes precedence; without either, the deterministic offline engine answers.
    """

    model: str = "claude-haiku-5-5"
    effort: str = "low"
    max_turns: int = 6
    max_tokens: int = 4000
    timeout_seconds: float = 25.0
    similar_rows: int = 5
    # The relay holds the key; this shared token only keeps casual scanners out.
    relay_url: str = "https://ledgerclf-agent-relay.vercel.app/api"
    relay_token: str = "69ffdcce74916b566ca3c2004eeab862"
    percent_tolerance: float = 1.0


@dataclass(frozen=True)
class Settings:
    """Complete configuration bundle passed explicitly to modules."""

    paths: Paths
    limits: Limits = field(default_factory=Limits)
    feature: FeatureParams = field(default_factory=FeatureParams)
    linear: LinearParams = field(default_factory=LinearParams)
    tree: TreeParams = field(default_factory=TreeParams)
    blend: BlendParams = field(default_factory=BlendParams)
    holdout_rule: HoldoutRule = field(default_factory=HoldoutRule)
    decision_rule: DecisionRule = field(default_factory=DecisionRule)
    calibration: CalibrationParams = field(default_factory=CalibrationParams)
    explain: ExplainParams = field(default_factory=ExplainParams)
    flag_policy: FlagPolicyParams = field(default_factory=FlagPolicyParams)
    agent: AgentParams = field(default_factory=AgentParams)
    benchmark: BenchmarkParams = field(default_factory=BenchmarkParams)
    tuning: TuningParams = field(default_factory=TuningParams)
    seeds: tuple[int, ...] = (0, 1, 2)
    random_state: int = 0
    cv_folds: int = 5
    nested_inner_folds: int = 3
    oot_cutoff: str = "2018-09-30"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the default settings rooted at this project.

    Returns:
        The process-wide default ``Settings``. Callers needing a different
        root (tests, clean-room copies) build ``Settings(paths=Paths.from_root(...))``.
    """
    return Settings(paths=Paths.from_root(PROJECT_ROOT))
