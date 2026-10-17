# 05. Robustness and Portability

Goal: the pipeline handles any reasonable upload without crashing, and the whole project runs on the grader's machine exactly as on the developer's machine. The worst outcome for this project: perfect behavior on the developer laptop and a break on the grader's laptop. Every rule here exists to prevent this outcome.

## 1. Principles

1. **Fail loud on structure.** Missing required columns, empty files, and unreadable files stop the run with a plain-English message and a fix.
2. **Tolerate values.** Bad cells become missing values, get imputed, and raise a row flag. One bad cell never stops a file.
3. **Never crash.** No stack trace reaches the user. Unexpected exceptions become `LedgerClfError` subclasses with a message, logged with the traceback at DEBUG level.
4. **Every non-empty row gets exactly one prediction**, in original order, traceable to its spreadsheet row.
5. **Same code path for every source.** Training rows, holdout files, and grader uploads all pass through `io.read_table`, `schema.resolve_columns`, and `schema.coerce_values`.

## 2. Input contract

### 2.1 Accepted files

| Format | Detection | Reader |
|---|---|---|
| `.xls` (Excel 97-2003) | magic bytes `D0 CF 11 E0 A1 B1 1A E1` | `pandas.read_excel(engine="xlrd", header=None, dtype=object)` |
| `.xlsx`, `.xlsm` | magic bytes `PK\x03\x04` plus `[Content_Types].xml` in the archive | `pandas.read_excel(engine="openpyxl", header=None, dtype=object)` |
| `.csv`, `.txt`, `.tsv` | any other file decoding as text | encoding sniffing then delimiter sniffing (2.3) |

Detect format from bytes first and extension second. A CSV named `.xlsx` still reads, with a warning.

Password-protected `.xlsx` files are OLE containers (same magic bytes as `.xls`). When the OLE reader reports encryption, raise `PASSWORD_PROTECTED` instead of a generic read error.

Limits (from `config.Limits`): file size 50 MB, 200,000 data rows, 1,000 characters per text cell, header search within the first 20 rows.

### 2.2 Sheet selection (Excel)

Pick the first sheet whose header row (2.4) contains the required columns. If none qualifies, pick the first non-empty sheet and continue to column resolution, which then reports what is missing. Record the chosen sheet in `ReadInfo` and add an INFO issue when the workbook holds more than one sheet.

### 2.3 CSV decoding

- Encodings, tried in order: BOM detection (`utf-8-sig`, `utf-16` with BOM), `utf-8`, `cp1252`, `latin-1`. Record the encoding used. `latin-1` always decodes, so add a WARNING when the reader falls back to `latin-1`.
- Delimiters: `csv.Sniffer` over the first 64 KB restricted to `, ; \t |`. Fallback: comma.
- Quoted fields with embedded delimiters and newlines must parse. Use pandas with `dtype=object`, `keep_default_na=False`, `header=None`.

### 2.4 Column resolution

1. Read with `header=None`. Scan the first 20 rows for a header row: the first row where at least 4 cells match a known column name after normalization (lowercase, strip, remove spaces, underscores, hyphens, and dots).
2. Known names: `col1` to `col7` map to `Col1` to `Col7`. Single letters `a` to `g` map to `Col1` to `Col7` (the brief names the inputs "Columns A to G"). Label aliases: `classificationlabel`, `label`, `class`, `target`, `h`.
3. Drop rows above the header row. Drop columns with an empty header and no values. Drop pandas index artifacts (`unnamed0` style headers whose values equal a running index).
4. When no header row exists and the frame has 7 or 8 columns: treat every row as data and map by position (first 7 to `Col1` to `Col7`, an 8th to the label). Add WARNING `POSITIONAL_MAPPING`.
5. Duplicate matches for one canonical name: keep the first, WARNING `DUPLICATE_COLUMN`.
6. Missing required columns (`Col1, Col3, Col4, Col5, Col6, Col7`): ERROR `MISSING_REQUIRED_COLUMNS`, listing missing and found columns. Missing `Col2`: WARNING only (display column).
7. Extra columns: dropped, WARNING `EXTRA_COLUMNS_IGNORED` naming them.

### 2.5 Value coercion

Applied in `schema.coerce_values`, identical for training and inference.

**All text cells (Col1, Col2, Col4, Col6, Col7):**

- Floats with integer value become integer strings (`10719.0` to `10719`). Other numbers become `str(value)`. Booleans become `True` or `False`.
- Unicode NFKC normalization. Replace non-breaking spaces and zero-width characters (`\u00a0 \u200b \u200c \u200d \ufeff`) with spaces or nothing. Collapse whitespace runs. Strip.
- Missing markers become missing: empty string, `nan`, `none`, `null`, `n/a`, `na`, `#n/a`, `-` (case-insensitive).
- Truncate beyond 1,000 characters, flag `TEXT_TRUNCATED`.
- Keep display case. Model normalization (lowercase) happens in the feature pipeline.

**Col7:** case-insensitive match against known document types maps to the canonical spelling (`nodoc` to `NoDoc`). Unknown values pass through and get flagged.

**Col3 (amount):**

- Numeric cells pass through. Booleans are invalid.
- Strings: strip, remove currency symbols and codes (`$ EUR USD GBP` and the euro and pound signs), remove thousands separators (commas, spaces, apostrophes), parentheses mean negative (`(500.00)` to `-500.00`), trailing minus means negative (`500-`), Unicode minus becomes `-`. European format `1.234,56` (regex `^\d{1,3}(\.\d{3})+(,\d+)?$`) converts to `1234.56`.
- Infinite or unparseable values become missing: WARNING `AMOUNT_UNPARSEABLE` with row numbers, row flag `AMOUNT_IMPUTED`.

**Col5 (period):**

- `datetime`, `date`, and `Timestamp` values pass. Timezone-aware values convert to naive.
- Numbers between 20,000 and 80,000 read as Excel serial dates (origin 1899-12-30).
- Strings: ISO 8601 first, then numeric `a/b/yyyy` and `a-b-yyyy` forms, then month-name forms. Infer day-first versus month-first per column: if any value has a first part above 12, the column reads day-first. Otherwise month-first, with one INFO issue `DATE_ORDER_ASSUMED_MONTH_FIRST`.
- Output: period index `year*12 + month` and display string `YYYY-MM`.
- Unparseable or missing: WARNING `DATE_UNPARSEABLE`, row flag `DATE_IMPUTED`.

**Rows:** drop rows with every input cell missing (WARNING with count and row numbers). Keep `source_row` (spreadsheet row number) for every remaining row.

### 2.6 Severity model

| Severity | Effect | Examples |
|---|---|---|
| ERROR | No predictions. Clear message and fix. | `FILE_EMPTY`, `FILE_TOO_LARGE`, `UNSUPPORTED_FORMAT`, `FILE_UNREADABLE`, `PASSWORD_PROTECTED`, `MISSING_REQUIRED_COLUMNS`, `NO_DATA_ROWS`, `TOO_MANY_ROWS` |
| WARNING | Predictions made. Data quality issue reported. | `POSITIONAL_MAPPING`, `EXTRA_COLUMNS_IGNORED`, `AMOUNT_UNPARSEABLE`, `DATE_UNPARSEABLE`, `EMPTY_ROWS_DROPPED`, `ENCODING_FALLBACK` |
| INFO | Context only. | `SHEET_SELECTED`, `DATE_ORDER_ASSUMED_MONTH_FIRST`, `LABELS_DETECTED` |

Row-level flags (docs/04 section 10) sit beside the file-level issues.

## 3. Output guarantees

- One row in `PredictionResult.predictions` per non-empty input row, same order.
- `predicted_label` always in the canonical class list. Probabilities in [0, 1], summing to 1 within 1e-9.
- `to_records()` and `ValidationReport.to_dict()` serialize with `json.dumps` (native Python types only, no NaN: use `None`).
- Exported CSV for users: UTF-8 with BOM (`utf-8-sig`) so Excel on Windows shows text correctly. Internal reports: plain UTF-8.

## 4. Edge-case catalog

Every ID below needs at least one test in `tests/edge_cases/` whose name starts with the lowercase ID (for example `test_f07_csv_utf16_with_bom`). The gate fails when an ID lacks a test. Generate fixture files inside tests with `tmp_path` (no committed binary fixtures), except the raw `.xls` file already in the repo.

### File level

| ID | Input | Expected |
|---|---|---|
| F01 | 0-byte file | ERROR `FILE_EMPTY` |
| F02 | `.pdf`, `.json`, `.docx` files | ERROR `UNSUPPORTED_FORMAT` listing supported types |
| F03 | CSV content named `.xlsx` | Reads as CSV, WARNING about the mismatch |
| F04 | Truncated or corrupted `.xlsx` | ERROR `FILE_UNREADABLE` |
| F05 | Password-protected `.xlsx` (simulate with an encrypted OLE container or mock the reader error) | ERROR `PASSWORD_PROTECTED` |
| F06 | File above 50 MB (use a sparse temp file or patch the limit) | ERROR `FILE_TOO_LARGE` |
| F07 | CSV in UTF-8, UTF-8 with BOM, UTF-16 with BOM, cp1252, latin-1 | Same predictions for all five |
| F08 | CSV with comma, semicolon, tab, and pipe delimiters | Same predictions for all four |
| F09 | CSV with quoted fields holding commas and line breaks | Parses, correct row count |
| F10 | CRLF, LF, and CR-only line endings | Same predictions |
| F11 | Workbook with an empty first sheet and data on the second | Picks the data sheet, INFO `SHEET_SELECTED` |
| F12 | Title rows above the header | Finds the header within 20 rows |
| F13 | The original `data/raw/Challenge_Data.xls` with the label column | Reads, scores, 5,899 predictions |
| F14 | `.xlsx` with formula cells lacking cached values | Treated as missing values, flagged, no crash |
| F15 | File name with spaces, commas, and non-ASCII characters | Reads normally |
| F16 | `.xlsm` file | Reads like `.xlsx` |

### Structure level

| ID | Input | Expected |
|---|---|---|
| S01 | Header only, no data rows | ERROR `NO_DATA_ROWS` |
| S02 | Missing Col4 and Col6 | ERROR `MISSING_REQUIRED_COLUMNS` naming both and listing found columns |
| S03 | Extra columns (`Notes`, `row_id`) | Ignored, WARNING naming them |
| S04 | Columns in shuffled order | Same predictions as canonical order |
| S05 | Header variants: `col1`, `COL1`, ` Col 1 `, `col_1`, `Col-1` | Resolved |
| S06 | No header, 7 columns | Positional mapping, WARNING |
| S07 | No header, 8 columns with labels in the last | Positional mapping, scores the labels |
| S08 | Duplicate `Col3` headers | First kept, WARNING |
| S09 | Label column present (raw variants) | Dropped from features, used for metrics |
| S10 | Col2 missing | WARNING only, predictions made |
| S11 | Empty rows between data rows | Dropped, WARNING with row numbers, `source_row` preserved |
| S12 | Exactly one data row | One prediction |
| S13 | Letter headers `A` to `G` | Resolved to Col1 to Col7 |
| S14 | Pandas index column (`Unnamed: 0`) | Ignored |

### Value level

| ID | Input | Expected |
|---|---|---|
| V01 | Numbers in Col1, Col4, Col6 | Cast to strings, prediction made |
| V02 | Missing Col1 or Col6 | `__missing__` token, flag `TEXT_MISSING`, review required |
| V03 | Missing Col4 | `__missing__` token, no critical flag (2.6% of training rows) |
| V04 | Extra spaces, tabs, non-breaking spaces, zero-width characters | Same prediction as the clean value |
| V05 | Lowercase tokens (`word563`) | Same prediction as `Word563` |
| V06 | Unseen vendor, account, pair, tokens | Prediction made, matching flags |
| V07 | Amount strings: `$1,234.50`, `1234.50 USD`, `(500.00)`, `500-`, `1.5e3`, ` 42 `, `1.234,56` | Parsed to the right floats |
| V08 | Amount `abc`, empty, `inf` | Imputed, flag `AMOUNT_IMPUTED`, WARNING |
| V09 | Amount 1e15 and -1e15 | Finite features, flag `AMOUNT_OUT_OF_RANGE` |
| V10 | Amount 0 and negatives | Valid, no flag |
| V11 | Dates: Timestamp, `2018-08-31`, `08/31/2018`, `31/08/2018`, `Aug 31, 2018`, `2018-08-31 13:45:00`, Excel serial `43343`, timezone-aware | All map to period 2018-08 |
| V12 | Ambiguous dates where every first part is 12 or less | Month-first, INFO issue |
| V13 | Date missing or `not a date` | Imputed, flag `DATE_IMPUTED` |
| V14 | Dates in 2015, 2019, 2030 | Clipped for features, flag `DATE_OUT_OF_RANGE` |
| V15 | Col7 `doc1`, ` NoDoc `, `Doc4` | First two canonicalized, `Doc4` flagged `UNSEEN_DOC_TYPE` |
| V16 | Col2 as scientific string, int, float | Kept as display strings, never used by the model |
| V17 | Mixed types within one column | Coerced per rules |
| V18 | 10,000-character text cell | Truncated, flag `TEXT_TRUNCATED` |
| V19 | Non-ASCII and emoji text | Handled, output JSON-safe |
| V20 | Boolean cells in text columns | Cast to strings |

### Label level (uploads with labels)

| ID | Input | Expected |
|---|---|---|
| L01 | All 11 raw label spellings | Normalized, metrics computed |
| L02 | Unknown labels (`Category_9`, `foo`, empty) | Rows excluded from scoring with WARNING, predictions still made |
| L03 | Label column entirely empty | No metrics, predictions made |
| L04 | Labels for only some rows | Metrics over labeled rows, count reported |

### Row level

| ID | Input | Expected |
|---|---|---|
| R01 | Duplicate rows | Identical predictions |
| R02 | Order preservation on a shuffled file | Output order equals input order |
| R03 | Training rows (non-holdout) re-uploaded | Predictions equal in-memory pipeline predictions |
| R04 | Holdout features uploaded as `.xlsx` and `.csv` | Identical predictions, equal to golden file |

### Environment level

| ID | Condition | Expected |
|---|---|---|
| E01 | `lightgbm` import fails (`ImportError` or `OSError`) | Fallback model loads, `model_used="fallback_logistic"`, reason recorded, doctor shows the OS-specific fix |
| E02 | Artifacts missing | `PredictionService.load(auto_train=True)` retrains from raw data and the saved split. With `auto_train=False`: `ArtifactError` naming the fix command |
| E03 | Artifact saved with a different scikit-learn minor version | Warning, load attempt, retrain or clear error on failure |
| E04 | Commands run from a different working directory | Work unchanged |
| E05 | Read-only project directory | Prediction works. Retraining fails with a clear message |
| E06 | Project path with spaces, commas, non-ASCII characters | Works |
| E07 | Console encoding cp1252 (`PYTHONIOENCODING=cp1252`) | CLI output prints without errors |
| E08 | No network | Everything works |
| E09 | Unsupported Python version | Doctor warns with the supported range |
| E10 | Concurrent predictions from threads | Same results as sequential calls |
| E11 | Corrupt artifact file | `ArtifactError` or auto-retrain, never a raw pickle traceback |
| E12 | `LEDGERCLF_FORCE_FALLBACK=1` | Fallback path, used by tests and CI |

### Explanation level

| ID | Input | Expected |
|---|---|---|
| X01 | Row number out of range | Clear error |
| X02 | Row with every token unseen | Valid payload, flags present |
| X03 | Any payload | Passes `json.dumps`, matches schema version 1.0 |
| X04 | No LLM available | `fallback_text` present and deterministic |
| X05 | Any explained row | Additivity holds within 1e-6 |

### Performance level

| ID | Input | Expected |
|---|---|---|
| P01 | 590-row holdout | Under 2 s end to end |
| P02 | 10,000 rows | Under 15 s |
| P03 | 100,000 rows | Under 120 s, peak memory under 2 GB |
| P04 | One explanation | Under 300 ms |

Test timeouts sit at three times the budget to avoid flaky failures on slower machines. Record measured times in `reports/robustness/performance.json`.

## 5. Portability rules

### 5.1 Supported platforms

Windows 10 and 11 (x64), macOS 13 or newer (Apple Silicon and Intel), Linux x86_64 with glibc 2.28 or newer. Python 3.11 to 3.14 supported, 3.12 recommended. State this in README and enforce in `doctor`.

### 5.2 Dependency pinning and wheel verification

- Exact `==` pins in `requirements.txt`, `requirements-dev.txt`, `requirements-research.txt`.
- Pick versions released at least 3 months before the build date, with binary wheels for CPython 3.11, 3.12, 3.13, and 3.14 on `win_amd64`, macOS arm64, macOS x86_64, and manylinux x86_64. Starting candidates to verify, not to trust blindly: numpy 2.3.x, pandas 2.3.x, scipy 1.16.x, scikit-learn 1.7.x, lightgbm 4.6.0, joblib 1.5.x, openpyxl 3.1.5, xlrd 2.0.x. Pick the newest patch of each line which passes the check.
- `tools/check_wheels.py` verifies wheel availability without installing: for every pin, Python version, and platform, run `pip download --only-binary=:all: --no-deps --python-version <v> --implementation cp --platform <tag> [--platform <tag> ...] -d <tmp> <name>==<version>`. Pass several platform tags per OS (for example `macosx_11_0_arm64`, `macosx_12_0_arm64`, `macosx_13_0_arm64`, `macosx_14_0_arm64`, and `manylinux2014_x86_64`, `manylinux_2_17_x86_64`, `manylinux_2_28_x86_64`). Print a matrix. Any gap fails the check.
- If no pin set covers both 3.11 and 3.14, drop 3.11 support and record the decision.
- Never use `>=` or unpinned names. Never install from Git URLs.

### 5.3 LightGBM fallback

- `import lightgbm` only inside functions (`models.make_lightgbm`, unpickling happens inside `artifacts.load_bundle`).
- `artifacts.load_best_available` tries `model_blend.joblib`. On `ImportError`, `OSError`, `ModuleNotFoundError`, or an unpickling error tied to LightGBM, log the reason and load `model_fallback.joblib`. The fallback bundle must hold zero LightGBM objects (test by unpickling the fallback bundle in a subprocess with LightGBM blocked).
- `PredictionResult.model_used` and `fallback_reason` surface the state. `doctor` prints the fix per OS: macOS `brew install libomp`, Debian or Ubuntu `sudo apt-get install -y libgomp1`, Windows "install the latest Microsoft Visual C++ Redistributable (x64)".
- `LEDGERCLF_FORCE_FALLBACK=1` forces the fallback for tests and the CI fallback job.

### 5.4 File system

- `pathlib.Path` everywhere. Resolve the project root from `Path(__file__).resolve()`. Never depend on the current working directory.
- No absolute paths in code or config.
- Paths with spaces, commas, and non-ASCII characters must work (the submission ZIP name itself contains a comma and spaces).
- Keep repo-internal paths short (under 100 characters relative to the root) to stay clear of Windows path limits.
- Prediction writes nothing inside the project. Temporary files go through `tempfile`. Close every file handle (context managers) to avoid Windows file locks.
- Match file name case exactly in imports and paths (macOS is case-insensitive by default, Linux is not).

### 5.5 Text, encoding, console

- Every text-mode `open()` passes `encoding="utf-8"` (or the detected encoding when reading uploads). Never rely on the platform default (cp1252 on Windows).
- CLI output and log messages use ASCII only. No emoji, no check marks, no box-drawing characters. The logging formatter escapes non-ASCII with `backslashreplace`.
- No non-ASCII characters in `.py` source files. Use escapes in tests.

### 5.6 Processes, threads, determinism

- No multiprocessing in the runtime path. `n_jobs=1` in production models. Scripts guard entry points with `if __name__ == "__main__":` (Windows spawn safety).
- No global mutable state in the serving path. Loaded bundles stay read-only.
- Seeds from config. LightGBM `deterministic=True`, `force_col_wise=True`, `n_jobs=1`.
- Golden comparisons: same machine, labels identical and probabilities within 1e-6. Cross-platform, labels agree on at least 99.5% of rows and probabilities within 1e-4. Report disagreeing rows.

### 5.7 Library versions

- Code runs on pandas 2.2 or newer, including pandas 3. Avoid chained assignment, `inplace=True`, and dtype-dependent string logic. Use `.loc`.
- Artifacts record library versions (docs/02 section 6). Mismatch handling follows E03.
- Load only artifacts from the project's own `artifacts/` directory. Never unpickle uploaded content.

### 5.8 Other rules

- No network access at runtime or in tests.
- No locale-dependent parsing (`locale` module banned).
- No shell-only tooling. Every tool runs as `python tools/<name>.py` on all three operating systems.

### 5.9 Optional Docker

Provide a `Dockerfile`: `python:3.12-slim`, `apt-get install -y --no-install-recommends libgomp1`, copy the repo, `pip install -r requirements.txt`, default command `python -m ledgerclf doctor`. Document build and run commands in README. Docker stays optional: the primary path remains a plain virtual environment.

## 6. Test strategy

| Layer | Location | Content |
|---|---|---|
| Unit | `tests/unit/` | Every public function: happy path, boundary, error path |
| Edge cases | `tests/edge_cases/` | Every catalog ID (section 4) |
| Property-based | `tests/unit/test_schema_fuzz.py` | Hypothesis generates frames with garbage values per column: `predict_frame` never raises anything except `SchemaError` for structural problems, returns one prediction per non-empty row, probabilities sum to 1. `max_examples=200`, `deadline=None` |
| Integration | `tests/integration/` | Train on a stratified subsample (quick config), save, load, predict from disk |
| Parity | `tests/integration/test_parity.py` | In-memory pipeline predictions on coerced holdout frame equal `predict_file` on `.xlsx` and on `.csv` |
| Golden | `tests/integration/test_golden.py` | `predict_file(holdout_features.xlsx)` equals `artifacts/golden_holdout_predictions.csv` |
| Reproducibility | `tests/integration/test_reproducibility.py` | Two quick trainings in temp dirs produce identical predictions and feature names |
| Leakage | `tests/integration/test_leakage.py` | Holdout row ids absent from training rows. No group spans both sides. Vectorizer vocabulary excludes tokens seen only in holdout rows. Reference statistics exclude holdout-only values. Holdout label file opened only by `evaluate-holdout` (guard function plus a test). |
| Portability (static) | `tests/portability/` | AST scan for text `open()` without encoding, `shell=True`, `os.system`, absolute path patterns, non-ASCII in `.py` files, top-level `import lightgbm` in `ledgerclf/` |
| Portability (subprocess) | `tests/portability/` | `python -m ledgerclf doctor` from another working directory. Import and predict with LightGBM blocked. CLI under `PYTHONIOENCODING=cp1252`. |

Markers: `slow` (training-heavy), `edge`, `portability`, `golden`. The default gate excludes `slow`. `gate.py --full` includes everything.

Coverage: at least 90% line coverage on `ledgerclf/`. `# pragma: no cover` allowed only on `if __name__ == "__main__":` lines.

## 7. Clean-room test (`tools/clean_room.py`)

1. Build the submission ZIP with `tools/package_submission.py` (test names).
2. Extract into a temp directory named `clean room, Unicode \u00fcml\u00e4ut test` (real characters on disk).
3. Create a fresh virtual environment with the current interpreter (and any extra interpreters passed with `--python`).
4. Install `requirements.txt` only. Run `doctor`, `predict` on `data/holdout/holdout_features.xlsx`, compare with the golden file, run `explain` on row 1.
5. Install `requirements-dev.txt`. Run `pytest -m "not slow"` inside the extracted copy.
6. Move artifacts aside, run `train`, compare predictions with the golden file.
7. Uninstall `lightgbm` in the venv. Run `doctor` and `predict`. Expect the fallback path and a clear message.
8. Write `reports/robustness/clean_room.json` with every step, duration, and result. Exit non-zero on any failure.

## 8. Continuous integration

Create `.github/workflows/ci.yml` (use the current major versions of `actions/checkout` and `actions/setup-python`):

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python: ["3.11", "3.12", "3.13", "3.14"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - if: runner.os == 'macOS'
        run: brew install libomp
      - run: python -m pip install -r requirements.txt -r requirements-dev.txt
      - run: python -m ruff format --check .
      - run: python -m ruff check .
      - run: python -m mypy ledgerclf
      - run: python -m pytest -m "not slow" -q
      - run: python -m ledgerclf doctor
      - run: python tools/compare_golden.py --tolerance cross-platform
  fallback:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: python -m pip install -r requirements.txt -r requirements-dev.txt
      - run: python -m pip uninstall -y lightgbm
      - run: python -m ledgerclf doctor
      - run: python -m pytest tests/edge_cases -k "e01 or e12" -q
  canary-latest:
    runs-on: ubuntu-latest
    continue-on-error: true
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.14"
      - run: python -m pip install numpy pandas scipy scikit-learn lightgbm joblib openpyxl xlrd pytest hypothesis
      - run: python -m pytest -m "not slow" -q
```

`tools/compare_golden.py` prints SKIP and exits 0 while `artifacts/golden_holdout_predictions.csv` does not exist yet (before Phase 6), so CI stays green during early phases.

Tell the user: push to a private GitHub repository to run this matrix, then paste failures back into Claude Code. Keep the repository private because the repository holds client-provided data.

## 9. Performance budgets

| Operation | Budget |
|---|---|
| `import ledgerclf.service` | under 2 s |
| Load best bundle | under 3 s |
| Predict 590 rows (read to result) | under 2 s |
| Predict 10,000 rows | under 15 s |
| Predict 100,000 rows | under 120 s, peak memory under 2 GB, chunked by `predict_chunk_rows` |
| One explanation | under 300 ms |
| `doctor` | under 10 s |

## 10. Doctor command

`python -m ledgerclf doctor` prints one ASCII line per check with `PASS`, `WARN`, or `FAIL`, then a summary and the fix for each problem:

1. Python version within 3.11 to 3.14.
2. Operating system and architecture.
3. Each runtime package importable, version compared with the pin (WARN on mismatch).
4. LightGBM import (FAIL means fallback active, print the OS-specific fix).
5. Artifacts present, metadata readable, library versions compatible.
6. Reference statistics present.
7. Holdout files present.
8. Write permission for `artifacts/` (WARN only).
9. Self-test: predict `ledgerclf/resources/sample_rows.csv` and compare with `artifacts/sample_predictions.csv`.

Exit codes: 0 when the environment checks pass and the service works (blend or fallback), or when artifacts do not exist yet (WARN with the fix `python -m ledgerclf train`). 2 when an environment check fails or artifacts exist but no model loads. With `--require-model`, missing artifacts also exit 2. The gate passes `--require-model` from Phase 6 onward. `PredictionService.health()` returns the same checks as JSON.
