# Ledger Classifier: EY AI Technical Assessment

This project suggests the classification label for lines of an accounts payable ledger. It is trained on an anonymized ledger of 5,899 invoice lines (inputs `Col1` to `Col7`, six categories), and it also includes:

- a dashboard with an AI agent that explains any single prediction in plain language;
- a prediction service that accepts messy real-world Excel and CSV files.

| | Result |
|---|---|
| Holdout accuracy (590 rows never seen in development, scored once) | **97.1%** |
| Grouped cross-validation accuracy (the expected accuracy on new data) | **95.4%** (always guessing the most common category: 88.4%) |
| Rows sent to a person for review / accuracy of the rest (holdout) | 8.8% / 98.9% |

## A 10-minute tour for graders

1. **Run the dashboard** (Quick start below). Tab 1 tells the data and modeling story. In Tab 2, upload `data/holdout/holdout_features.xlsx` and click **Explain** on any row.
2. **Read the notebooks** in order. Each one stands on its own:
   - `notebooks/01_eda.ipynb`: what is in the data and what it changed in the model.
   - `notebooks/02_preprocessing.ipynb`: every step from a raw spreadsheet cell to the feature matrix.
   - `notebooks/03_modeling.ipynb`: validation design, model comparison, tuning, the final choice, calibration, and the holdout result.
3. **Skim `MODEL_CARD.md`**: results, key design choices, and limitations on one page.

## Requirements

- Python 3.11 to 3.14 (3.12 recommended).
- Windows 10/11, macOS 13+, or Linux x86_64.
- No internet access at runtime, except for the optional AI agent.
- macOS only: run `brew install libomp` so LightGBM can load. Without it, the app runs on the fallback model and `doctor` says so.

## Install

macOS / Linux:

```
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Windows PowerShell:

```
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The last line registers the package so that `python -m ledgerclf` works from any folder. The trained models, the holdout files, and every report ship with the project, so nothing needs to be retrained. For the tests and notebooks, also run `python -m pip install -r requirements-dev.txt`.

## Quick start

```
python -m ledgerclf doctor                                   check the environment and model files
python -m ledgerclf serve                                    dashboard at http://127.0.0.1:8000
python -m ledgerclf predict data/holdout/holdout_features.xlsx --out predictions.csv
python -m ledgerclf explain data/holdout/holdout_features.xlsx --row 1
```

The API documentation is at http://127.0.0.1:8000/docs while the server runs.

## How it works

1. **Validation.** Any Excel (.xlsx, .xls) or CSV file is read: odd encodings, delimiters, title rows above the header, shuffled or renamed columns, currency strings, European numbers, and mixed date formats are all handled. Problems become clear warnings, and **every non-empty row gets a prediction**.
2. **Features.** Each text column (vendor, description, GL account) gets its own TF-IDF vocabulary. Recurring vendors, accounts, and vendor-account pairs get one-hot indicators. The amount enters as a signed log plus flags for credits and round amounts, and the posting date as a month index. Col2, the invoice number, is an identifier and is not used.
3. **Model.** A blend of logistic regression and LightGBM, combined in log-odds space. Every prediction can be split exactly into contributions from each input, which the explanations use. Confidence is calibrated, so "90%" means right about 9 times out of 10.
4. **Review routing.** Rows below 80% confidence, or with a new vendor or GL account, are flagged for a person. Flags never block a prediction.
5. **Explanations.** On request, the agent explains one row. It gives a plain-language summary, then the exact facts: the probability, each input's contribution, and how past invoices with the same vendor and account were labeled. Then it says what a reviewer should check.

## The AI agent

The agent explains one row when asked, never in advance. It answers only from the model's own figures for that row and from the training history, and every number it quotes is checked against those figures. It reaches Claude (`claude-haiku-5-5`) by the first route that works:

1. **Your own Anthropic key**, typed into the dashboard for the session, or set before `serve`:
   - macOS / Linux: `export ANTHROPIC_API_KEY=...`
   - PowerShell: `$env:ANTHROPIC_API_KEY="..."`
   - or a line `ANTHROPIC_API_KEY=...` in a `.env` file in the project folder
2. **The project relay (no setup needed).** A small hosted function (source in `relay/`) holds the project's key on the server and forwards only this agent's explanation requests.
3. **Offline explanation.** If neither route answers (no internet, a firewall, an outage), you get the same structured explanation built directly from the model's figures. It is labeled as offline, and the app keeps working.

`GET /api/agent/status` shows which route is active. Corporate networks that inspect TLS traffic work, because the client uses the operating system's certificate store.

## All commands

```
python -m ledgerclf doctor [--require-model]     environment and model health check
python -m ledgerclf split [--force]              label cleanup, duplicate groups, holdout split
python -m ledgerclf eda                          exploration reports (training rows only)
python -m ledgerclf benchmark [--quick]          model comparison, ablations, validation schemes
python -m ledgerclf tune [--quick]               logged hyperparameter search and nested CV
python -m ledgerclf select                       final model choice, calibration, review threshold
python -m ledgerclf train                        final fit on the training rows, model files
python -m ledgerclf evaluate-holdout --confirm   the single holdout evaluation (already run once)
python -m ledgerclf predict FILE [--out CSV]
python -m ledgerclf explain FILE --row N [--out JSON]
python -m ledgerclf serve [--host H] [--port P] [--no-browser]
```

Exit codes: 0 success, 1 problem with the input (the message says how to fix it), 2 environment or model-file problem, 3 internal error (rerun with `--verbose`).

## Reproduce everything from the raw file

```
python -m ledgerclf split
python -m ledgerclf eda
python -m ledgerclf benchmark
python -m ledgerclf tune
python -m ledgerclf select
python -m ledgerclf train
python tools/compare_golden.py --tolerance same-machine
```

`split` reproduces the shipped holdout split byte for byte, and it refuses to change it silently. Every step is deterministic: fixed seeds, and single-threaded, deterministic LightGBM. `compare_golden.py` checks that a retrained model gives the same predictions as the shipped one. `benchmark` and `tune` take the longest; `--quick` cuts each to a few minutes.

## Tests

```
python -m pytest                     about 500 tests: units, 79 catalogued edge cases,
                                     72 hostile-input tests, API and agent, portability
python tools/run_notebooks.py        re-execute the three notebooks
```

## Project structure

```
ledgerclf/        the Python package
  io, schema, values, labels       reading and validating any uploaded file
  splitting                        duplicate groups and the holdout split
  features                         the feature pipeline (single source of feature logic)
  models, evaluation, calibration  models, grouped cross-validation, calibration
  benchmark, tuning, selection     model comparison, search, final choice
  training, predict, explain       final fit, predictions, exact explanations
  flags, reference, overlap        review flags and training-history evidence
  service, server                  the facade used by the dashboard, and the HTTP API
  agent, agent_tools               the Claude explanation agent and its read-only tools
  cli                              python -m ledgerclf ...
frontend/         the dashboard (built files in frontend/dist are served by `serve`)
notebooks/        01 exploration, 02 preprocessing, 03 modeling
data/raw/         Challenge_Data.xls (original, never modified)
data/splits/      holdout row ids and the split record
data/holdout/     holdout_features.xlsx/.csv (upload these), holdout labels
artifacts/        trained models, training statistics, reference predictions
reports/          every table behind the notebooks and the dashboard (reports/manifest.json)
relay/            the hosted relay that lets the agent run without a key
tests/            the test suite
```

## Troubleshooting

- **`doctor` says LightGBM cannot load (macOS).** Run `brew install libomp`. Until then, the fallback model (95.3% holdout accuracy) is used and everything else works.
- **`No module named ledgerclf`.** Run `python -m pip install -e . --no-deps` inside the activated virtual environment.
- **Wrong Python version.** Use Python 3.11 to 3.14; on Windows, `py -3.12`.
- **The agent answers "offline".** No key is set and the relay cannot be reached (no internet or a firewall). The offline explanation uses the same exact figures.
- **Port 8000 is in use.** `python -m ledgerclf serve --port 8010`.
- **Guaranteed environment.** Build and run the Docker image:
  ```
  docker build -t ledgerclf .
  docker run --rm -p 8000:8000 ledgerclf python -m ledgerclf serve --host 0.0.0.0 --no-browser
  ```
