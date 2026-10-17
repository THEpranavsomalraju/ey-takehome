# 02. Architecture

## 1. Design principles

1. **One pipeline object.** Training and inference share one fitted sklearn `Pipeline` per model. Inference never reimplements a transformation.
2. **Validate at the edge.** Raw files pass through `io` then `schema` before touching the model. Everything downstream receives a clean, typed frame.
3. **Pure functions, explicit inputs.** Modules take data and config as arguments and return values. No hidden global state. The service holds loaded artifacts read-only.
4. **Stateless serving.** Every prediction call stands alone. Safe for dashboard reruns and concurrent calls.
5. **Fail loud on structure, tolerant on values.** A missing required column stops the run with a clear message. A bad cell becomes a flagged, imputed value.
6. **Graceful degradation.** If LightGBM fails to import, the service loads the fallback model and says so in every result.
7. **Reports as data.** Every table and chart the dashboard shows comes from a file in `reports/`, listed in `reports/manifest.json`.

## 2. Repository layout

```
ledgerclf/
  __init__.py          package version, public API re-exports (service only)
  __main__.py          enables `python -m ledgerclf`, delegates to cli.main()
  cli.py               argparse commands, ASCII output, exit codes
  config.py            frozen dataclasses: paths, seeds, CV, features, models, thresholds, limits
  exceptions.py        exception hierarchy (section 10)
  logging_setup.py     one function configuring logging, ASCII-safe formatter
  schema.py            column constants, column resolution, value coercion, ValidationReport
  io.py                file reading: xls, xlsx, csv (encoding and delimiter sniffing), size limits
  labels.py            label normalization, canonical class list
  splitting.py         group key, holdout seed rule, holdout split, CV splitter factories
  eda.py               EDA tables and statistics (pure functions returning DataFrames or dicts)
  features.py          custom transformers and build_feature_pipeline()
  models.py            model factories, PoweredClassWeight wrapper, LogOddsBlendClassifier, baselines
  evaluation.py        metrics, CV runner with out-of-fold capture, scorecard, validation schemes
  tuning.py            logged searches, nested CV
  calibration.py       temperature scaling, ECE, reliability tables, threshold tables
  selection.py         champion decision rule, champion.json
  reference.py         training reference statistics (known values, ranges, class frequencies)
  flags.py             row-level flag computation and review policy
  explain.py           exact contributions, aggregation, evidence, payload, fallback text
  artifacts.py         save and load bundles with metadata, version checks
  predict.py           predict_frame() and predict_file(): validation to PredictionResult
  service.py           PredictionService facade for Part 2 (section 8)
  doctor.py            environment and artifact diagnostics, self-test prediction
  reports.py           write reports and maintain reports/manifest.json
  resources/
    sample_rows.csv    5 training rows (never holdout) for the doctor self-test
tools/
  gate.py              quality gate runner (docs/06 section 2)
  check_wheels.py      wheel availability across OS and Python matrix
  clean_room.py        fresh venv, odd path, install, test, predict, compare golden
  package_submission.py  builds "Last, First_AI Technical Assessment.zip"
  reference_ranges.json  ranges from docs/01 section 15
  check_reference_ranges.py
  compare_golden.py    compares predictions with artifacts/golden_holdout_predictions.csv (same-machine or cross-platform tolerance)
tests/
  conftest.py          fixtures: raw frame, training frame, small fitted pipeline, file factories
  unit/                one test module per package module
  edge_cases/          one test per catalog ID in docs/05 section 4 (test names include the ID)
  integration/         end-to-end train-on-subsample, predict-from-disk, parity, golden
  portability/         static checks (paths, encodings, ASCII, lazy imports), subprocess runs
  adversarial/         tests written by the edge-case-breaker subagent
notebooks/
  01_eda.ipynb
  02_preprocessing.ipynb
  03_modeling.ipynb
  _style.py            shared chart style and class colors for notebooks
research/
  catboost_benchmark.py  optional, needs requirements-research.txt, writes reports/research_catboost.json
```

Flat layout (package at repo root) so `python -m ledgerclf` works from the repo root without installation. `pyproject.toml` still supports `pip install -e .` for tooling.

## 3. Data flow

```
raw file (path or bytes + filename)
  -> io.read_table()            DataFrame of strings/objects + ReadInfo (sheet, encoding, delimiter, header row)
  -> schema.resolve_columns()   canonical column names + issues (missing, extra, positional mapping)
  -> schema.coerce_values()     typed frame: text cleaned, Col3 float, Col5 period, missing indicators + issues (no imputation)
  -> flags.compute_row_flags()  per-row flags against reference statistics
  -> pipeline.predict_proba()   features.FeatureBuilder -> classifier (blend or fallback)
  -> calibration (temperature inside the classifier)
  -> flags.apply_review_policy()
  -> PredictionResult           predictions frame + ValidationReport + metrics (if labels present) + timings
```

Training follows the same path through `schema.coerce_values()`, so training and inference see identical cleaning.

## 4. Module contracts

Signatures below define the public surface. Keep private helpers private (leading underscore). Every public function gets a Google-style docstring with Args, Returns, Raises.

### config.py

```python
@dataclass(frozen=True)
class Paths:
    root: Path            # resolved from Path(__file__).resolve().parent.parent
    raw_data: Path        # data/raw/Challenge_Data.xls
    splits: Path
    holdout: Path
    artifacts: Path
    reports: Path

@dataclass(frozen=True)
class Limits:
    max_file_bytes: int = 50 * 1024 * 1024
    max_rows: int = 200_000
    max_text_chars: int = 1_000
    header_scan_rows: int = 20
    predict_chunk_rows: int = 20_000

@dataclass(frozen=True)
class Settings:
    paths: Paths
    limits: Limits
    seeds: tuple[int, ...] = (0, 1, 2)
    holdout_fraction_folds: int = 10
    cv_folds: int = 5
    oot_cutoff: str = "2018-09-30"
    feature: FeatureParams
    linear: LinearParams
    tree: TreeParams
    blend: BlendParams
    thresholds: Thresholds

def get_settings() -> Settings: ...
```

All tunable numbers live here, including defaults overwritten by `champion.json` at train time.

### io.py

```python
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".xls", ".xlsx", ".xlsm", ".csv", ".txt", ".tsv")

@dataclass(frozen=True)
class ReadInfo:
    file_name: str
    detected_format: str          # "xls" | "xlsx" | "csv"
    sheet_name: str | None
    encoding: str | None
    delimiter: str | None
    header_row_index: int         # 0-based row in the sheet where the header sits

def read_table(source: Path | bytes, file_name: str, limits: Limits) -> tuple[pd.DataFrame, ReadInfo, list[ValidationIssue]]:
    """Read any supported file into a raw frame with every cell as object. Sniffs format from magic bytes,
    not only the extension. Raises FileReadError with a plain-English message on fatal problems."""
```

### schema.py

```python
INPUT_COLUMNS: tuple[str, ...] = ("Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7")
MODEL_INPUT_COLUMNS: tuple[str, ...] = ("Col1", "Col3", "Col4", "Col5", "Col6", "Col7")
REQUIRED_COLUMNS: tuple[str, ...] = MODEL_INPUT_COLUMNS
OPTIONAL_COLUMNS: tuple[str, ...] = ("Col2",)
LABEL_COLUMN: str = "ClassificationLabel"
MISSING_TOKEN: str = "__missing__"

class Severity(StrEnum): ERROR, WARNING, INFO

@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    code: str                     # e.g. "MISSING_REQUIRED_COLUMN", "AMOUNT_UNPARSEABLE"
    message: str                  # plain English, includes the fix when one exists
    column: str | None = None
    row_numbers: tuple[int, ...] = ()   # file row numbers (1-based, header = row 1)

@dataclass
class ValidationReport:
    issues: list[ValidationIssue]
    rows_read: int
    rows_dropped_empty: int
    rows_predicted: int
    column_mapping: dict[str, str]      # source header -> canonical name
    def has_errors(self) -> bool: ...
    def to_dict(self) -> dict: ...      # JSON-safe

def resolve_columns(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], list[ValidationIssue]]: ...
def coerce_values(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[ValidationIssue]]:
    """Return a typed frame with columns: row_number, source_row, Col1..Col7 display values,
    amount (float, NaN when missing or unparseable), period (int index, NaN when missing or unparseable),
    amount_was_missing (bool), period_was_missing (bool). Coercion never imputes: the pipeline's
    TrainingMedianImputer fills NaN with training-fold medians, so training and inference impute
    identically and no statistic leaks across folds."""
```

### labels.py

```python
CANONICAL_CLASSES: tuple[str, ...] = ("Category_1", ..., "Category_6")
def normalize_label(raw: object) -> str | None: ...      # None for unrecognized labels
def normalize_labels(values: pd.Series) -> tuple[pd.Series, pd.DataFrame]:  # labels + mapping table
```

### splitting.py

```python
def build_group_key(frame: pd.DataFrame) -> np.ndarray: ...          # integer group ids, Col2 excluded
def select_holdout_seed(y: np.ndarray, groups: np.ndarray, rule: HoldoutRule) -> HoldoutSelection: ...
def make_holdout_split(...) -> SplitResult: ...                       # train/holdout row ids + manifest
def load_split(paths: Paths) -> SplitResult: ...                      # reads data/splits, never recomputes
def grouped_cv(seed: int, n_splits: int) -> StratifiedGroupKFold: ...
def vendor_groups(frame: pd.DataFrame) -> np.ndarray: ...
def out_of_time_mask(frame: pd.DataFrame, cutoff: str) -> np.ndarray: ...
def load_training_frame(settings: Settings) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Training rows only: coerced frame, labels, groups. The single entry point for all modeling code."""
```

### features.py

Custom transformers (sklearn API, `get_feature_names_out` implemented, `set_output` not required):

| Transformer | Input | Output |
|---|---|---|
| `TextNormalizer` | text columns | lowercased strings, missing or empty to `MISSING_TOKEN` |
| `SignedLogAmount` | amount | `sign(x) * log1p(abs(x))` |
| `AmountFlags` | amount | `is_negative`, `is_round` |
| `PeriodEncoder` | period | period index clipped to the training range learned at fit |
| `ColumnConcat` | Col1, Col6 | `vendor || account` key |
| `TrainingMedianImputer` | amount, period | fills NaN with training-fold median |

```python
def build_feature_pipeline(params: FeatureParams) -> ColumnTransformer: ...
def feature_groups(feature_names: Sequence[str]) -> dict[str, list[int]]:
    """Map each output feature index to a source group: Col1, Col1+Col6, Col3, Col4, Col5, Col6, Col7."""
```

### models.py

```python
def make_dummy() -> Pipeline: ...
def make_lookup() -> VendorAccountLookup: ...
def make_logistic(params: LinearParams, class_weight_power: float) -> Pipeline: ...
def make_linear_svc(...) -> Pipeline: ...
def make_complement_nb(...) -> Pipeline: ...
def make_lightgbm(params: TreeParams, class_weight_power: float) -> Pipeline: ...   # lazy import inside
def make_blend(linear: LinearParams, tree: TreeParams, blend: BlendParams, class_weight_power: float) -> Pipeline: ...
def make_probability_blend(...) -> Pipeline: ...   # comparison only

class PoweredClassWeight(BaseEstimator, ClassifierMixin):
    """Wraps an estimator with a class_weight parameter. At fit, sets class_weight to
    (n_max / n_c) ** power computed from the training labels of that fit."""

class LogOddsBlendClassifier(BaseEstimator, ClassifierMixin):
    """Blends a linear model and LightGBM in logit space. See docs/04 section 6."""
    def fit(self, X, y): ...
    def decision_function(self, X) -> np.ndarray: ...     # blended logits divided by temperature
    def predict_proba(self, X) -> np.ndarray: ...
    def predict(self, X) -> np.ndarray: ...
    def contributions(self, X) -> Contributions: ...       # exact, additive, per class
```

### evaluation.py

```python
@dataclass
class CVResult:
    model_name: str
    per_seed: pd.DataFrame          # one row per seed: accuracy, macro_f1_5, macro_f1_6, balanced_accuracy, per-class recall
    summary: pd.Series              # means and standard deviations
    oof_proba: np.ndarray | None    # last seed, aligned to training rows
    oof_logits: np.ndarray | None
    fit_seconds: float

def compute_metrics(y_true, y_pred, proba=None) -> dict[str, float]: ...
def run_grouped_cv(factory: Callable[[], BaseEstimator], X, y, groups, seeds, n_splits) -> CVResult: ...
def run_random_cv(...) -> CVResult: ...
def run_out_of_time(factory, X, y, mask_train, mask_test) -> dict: ...
def run_unseen_vendor_cv(factory, X, y, vendor_groups, seeds) -> CVResult: ...
def accuracy_by_flag(y_true, y_pred, flags: pd.DataFrame) -> pd.DataFrame: ...
```

### tuning.py, calibration.py, selection.py

```python
def search_linear(...) -> SearchResult: ...      # grid, logged trials
def search_tree(...) -> SearchResult: ...        # random search, logged trials
def search_blend_weight(...) -> SearchResult: ...
def nested_cv(...) -> dict: ...

def fit_temperature(logits: np.ndarray, y: np.ndarray, classes) -> float: ...
def expected_calibration_error(proba, y, n_bins=10) -> float: ...
def reliability_table(proba, y, n_bins=10) -> pd.DataFrame: ...
def threshold_table(proba, y, thresholds) -> pd.DataFrame: ...

def apply_decision_rule(candidates: list[CandidateScore], rule: DecisionRule) -> ChampionDecision: ...
```

### reference.py and flags.py

```python
@dataclass(frozen=True)
class ReferenceStats:
    known_vendors: frozenset[str]
    known_accounts: frozenset[str]
    known_pairs: frozenset[str]
    known_doc_types: frozenset[str]
    known_tokens: dict[str, frozenset[str]]     # per text column
    period_min: int
    period_max: int
    amount_min: float
    amount_max: float
    amount_median: float          # display and explanation context only (imputation lives in the pipeline)
    period_median: int
    value_class_counts: dict[str, dict[str, dict[str, int]]]   # column -> value -> class -> count
    token_class_counts: dict[str, dict[str, dict[str, int]]]   # column -> token -> class -> count
    class_counts: dict[str, int]

def build_reference(train_frame: pd.DataFrame, y: np.ndarray) -> ReferenceStats: ...
def compute_row_flags(frame: pd.DataFrame, reference: ReferenceStats) -> pd.DataFrame: ...
def apply_review_policy(flags: pd.DataFrame, confidence: np.ndarray, policy: ReviewPolicy) -> np.ndarray: ...
```

### explain.py

```python
def explain_rows(bundle: ModelBundle, frame: pd.DataFrame, row_numbers: Sequence[int]) -> list[ExplanationPayload]: ...
def render_fallback_text(payload: ExplanationPayload) -> str: ...
```

### artifacts.py

```python
@dataclass
class ModelBundle:
    pipeline: Pipeline
    reference: ReferenceStats
    metadata: dict
    classes: tuple[str, ...]
    is_fallback: bool

def save_bundle(bundle: ModelBundle, name: str, paths: Paths) -> None: ...
def load_bundle(name: str, paths: Paths) -> ModelBundle: ...      # raises ArtifactError with fix
def load_best_available(paths: Paths) -> ModelBundle: ...           # blend, else fallback, logs reason
```

### predict.py

```python
def predict_frame(bundle: ModelBundle, raw: pd.DataFrame, read_info: ReadInfo | None = None) -> PredictionResult: ...
def predict_file(bundle: ModelBundle, source: Path | bytes, file_name: str) -> PredictionResult: ...
```

## 5. Core data structures

### PredictionResult

```python
@dataclass
class PredictionResult:
    predictions: pd.DataFrame        # columns below
    report: ValidationReport
    metrics: dict | None             # present only when the upload holds usable labels
    model_used: str                  # "logodds_blend" | "fallback_logistic"
    fallback_reason: str | None
    timings_ms: dict[str, float]     # read, validate, predict, total
    def to_records(self) -> list[dict]: ...   # JSON-safe, native Python types only
```

Predictions columns, in order:

| Column | Type | Meaning |
|---|---|---|
| `row_number` | int | 1-based position among data rows as read |
| `source_row` | int | Spreadsheet row number in the original file (header counts as a row) |
| `Col1` to `Col7` | str | Cleaned display values |
| `predicted_label` | str | Canonical class |
| `confidence` | float | Calibrated probability of the predicted class |
| `runner_up_label` | str | Second most likely class |
| `runner_up_probability` | float | Its calibrated probability |
| `p_Category_1` to `p_Category_6` | float | Calibrated class probabilities |
| `review_required` | bool | Review policy outcome |
| `flags` | str | Pipe-separated flag codes, empty string when none |
| `true_label` | str or empty | Only when the upload holds labels |
| `correct` | bool or empty | Only when labels exist for the row |

### ExplanationPayload (JSON schema, version 1.0)

```json
{
  "schema_version": "1.0",
  "row_number": 17,
  "input": {"Col1": "...", "Col2": "...", "Col3": "...", "Col4": "...", "Col5": "2018-08", "Col6": "...", "Col7": "..."},
  "prediction": {
    "label": "Category_2",
    "probability": 0.91,
    "runner_up": {"label": "Category_1", "probability": 0.07},
    "logit_margin": 2.56,
    "review_required": false
  },
  "probabilities": {"Category_1": 0.07, "Category_2": 0.91, "...": 0.0},
  "model": {"name": "logodds_blend", "linear_weight": 0.5, "temperature": 1.12, "fallback_active": false},
  "contributions": {
    "target": "predicted_minus_runner_up",
    "total": 2.56,
    "bias": 0.31,
    "by_group": [{"group": "Col6", "contribution": 1.40}, {"group": "Col4", "contribution": 0.62}],
    "top_features": [
      {"group": "Col6", "feature": "token:word575", "present_in_row": true, "contribution": 1.12},
      {"group": "Col1", "feature": "token:word19", "present_in_row": true, "contribution": 0.41}
    ]
  },
  "evidence": [
    {"group": "Col6", "value": "Word563 Word575", "seen_in_training": true, "training_rows": 226,
     "class_counts": {"Category_1": 31, "Category_2": 195}, "share_predicted_class": 0.863}
  ],
  "flags": [{"code": "UNSEEN_VENDOR_ACCOUNT_PAIR", "message": "This vendor and account pair never appeared together in training."}],
  "context": {"training_rows": 5309, "class_prior": {"Category_1": 0.884}, "period_range": ["2016-10", "2018-12"], "amount_percentile": 0.62},
  "fallback_text": "Predicted Category_2 with 91% confidence ..."
}
```

Rules: native Python types only (test with `json.dumps`), probabilities rounded to 4 decimals for display fields but unrounded values used for additivity tests, `total` equals `bias` plus the sum over all groups within 1e-6.

## 6. Artifacts

```
artifacts/
  model_blend.joblib         ModelBundle (pipeline with LogOddsBlendClassifier)
  model_fallback.joblib      ModelBundle (pipeline with calibrated logistic regression, no LightGBM objects)
  metadata.json              see below
  reference.json             ReferenceStats as JSON (also embedded in bundles)
  champion.json              output of `select`: chosen config, decision trail
  golden_holdout_predictions.csv   predictions of both models on holdout features, for parity and clean-room checks
  sample_predictions.csv     doctor self-test expected output
```

`metadata.json` fields: `schema_version`, `created_utc`, `package_version`, `python_version`, `platform`, `library_versions` (numpy, pandas, scipy, scikit-learn, lightgbm, joblib), `raw_data_sha256`, `split_manifest_sha256`, `holdout_seed`, `training_rows`, `classes`, `feature_names_count`, `champion_config`, `temperature`, `thresholds`, `cv_summary`, `oot_summary`, `fallback_cv_summary`.

Load rule: compare `library_versions` with installed versions. Same major.minor for scikit-learn and lightgbm: load. Different: log a warning, try to load, and on failure raise `ArtifactError` telling the user to run `python -m ledgerclf train`. `load_service(auto_train=True)` retrains automatically from `data/raw` and the saved split when artifacts are missing or unloadable, provided the directory is writable.

## 7. Reports

Every report lives in `reports/` as CSV (tables) or JSON (structured). `reports/manifest.json` lists each file:

```json
{"reports": [{"file": "eda/class_distribution.csv", "title": "Class distribution", "producer": "eda", "description": "...", "columns": ["class", "rows", "share"]}]}
```

Full list of required reports: `docs/04_MODELING_SPEC.md` section 15. Reports never contain holdout labels except `reports/holdout/*`, written only by `evaluate-holdout`.

## 8. Service facade (contract for Part 2)

```python
class PredictionService:
    @classmethod
    def load(cls, auto_train: bool = True) -> "PredictionService": ...
    def predict_file(self, data: bytes, file_name: str) -> PredictionResult: ...
    def explain(self, result: PredictionResult, row_number: int) -> ExplanationPayload: ...
    def model_info(self) -> dict: ...            # metadata summary, fallback status, metrics
    def list_reports(self) -> list[dict]: ...    # from manifest
    def load_report(self, file: str) -> pd.DataFrame | dict: ...
    def health(self) -> dict: ...                # doctor summary, JSON-safe
```

The dashboard imports only `ledgerclf.service`. Explanations require the frame used for prediction, so `PredictionResult` keeps the coerced frame privately (`_frame`) for `explain`.

## 9. Configuration

`config.py` holds defaults. `artifacts/champion.json` holds the selected model configuration produced by `select`. `train` reads `champion.json`, never ad hoc arguments, so the trained model always matches the documented decision.

## 10. Errors and messages

```
LedgerClfError
  FileReadError        unreadable, unsupported, empty, too large, password protected
  SchemaError          missing required columns, no data rows, duplicate headers beyond recovery
  ArtifactError        missing, incompatible, or corrupt artifacts
  HoldoutAccessError   holdout labels requested outside evaluate-holdout
  ConfigError          invalid configuration or champion.json
```

Message style: what happened, why, how to fix. Example: `Missing required columns: Col4, Col6. Found columns: Col1, Col2, Col3, Col5, Col7. Rename the columns to match Col1 to Col7, or upload a file with all seven input columns.`

CLI exit codes: 0 success, 1 user input error, 2 artifact or environment error, 3 internal error.

## 11. Logging

`logging_setup.configure(level)` sets one handler with an ASCII-safe formatter (non-ASCII characters escaped). Library modules call `logging.getLogger(__name__)`. No `print` outside `cli.py` and `tools/`.

## 12. Dependencies

Runtime (`requirements.txt`, exact pins, see docs/05 section 5 for version selection): numpy, pandas, scipy, scikit-learn, lightgbm, joblib, openpyxl, xlrd.

Development (`requirements-dev.txt`, exact pins): pytest, pytest-cov, pytest-timeout, hypothesis, ruff, mypy, pandas-stubs, jupyter (nbclient, nbformat, ipykernel), matplotlib.

Research (`requirements-research.txt`, optional): catboost.

Adding any other dependency requires user approval (CLAUDE.md section 8).
