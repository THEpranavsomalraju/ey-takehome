# Ledger Classifier (EY AI Technical Assessment)

This project classifies lines of an anonymized accounts payable ledger: 5,899 rows, inputs `Col1` to `Col7`, and six classes. It includes:

- a validated prediction pipeline;
- a calibrated log-odds blend of logistic regression and LightGBM;
- exact per-row explanations;
- an on-demand AI agent (Claude) that explains any predicted row.

Results, limitations, and the fallback model are in [MODEL_CARD.md](MODEL_CARD.md). Every design choice is recorded with its evidence in [DECISIONS.md](DECISIONS.md).

## Requirements

- Python 3.11 to 3.14 (3.12 recommended).
- Windows 10/11, macOS 13+, or Linux x86_64.
- No internet connection at runtime, except the optional AI agent.
- macOS only: run `brew install libomp` so LightGBM can load. Without it the app still runs, on the fallback model, and `doctor` says so.

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

The last line registers the package in the virtual environment, so `python -m ledgerclf` works from any folder. The trained model, the holdout files and all reports ship with the project, so nothing needs to be retrained.

For the tests and notebooks, also run: `python -m pip install -r requirements-dev.txt`.

## Quick start

```
python -m ledgerclf doctor                                   check environment and artifacts
python -m ledgerclf serve                                    dashboard at http://127.0.0.1:8000
python -m ledgerclf predict data/holdout/holdout_features.xlsx --out predictions.csv
python -m ledgerclf explain data/holdout/holdout_features.xlsx --row 1
```

`serve` opens the dashboard in your browser. It has two tabs:

- **Tab 1** tells the data and modeling story.
- **Tab 2** accepts an Excel or CSV upload (for example `data/holdout/holdout_features.xlsx`). It predicts every row in real time and explains any row on demand.

The API documentation is at http://127.0.0.1:8000/docs.

## The AI agent

The agent explains one row when you ask. It answers from the exact explanation payload of that row and from evidence in the training data, and every number it quotes is checked against that evidence. It reaches Claude (`claude-haiku-5-5`) through the first route that is available:

1. **Your own key**, typed into the dashboard for the session, or set before `serve`:
   - macOS / Linux: `export ANTHROPIC_API_KEY=...`
   - PowerShell: `$env:ANTHROPIC_API_KEY="..."`
   - or a line `ANTHROPIC_API_KEY=...` in a `.env` file in the project folder
2. **The project relay** (no setup). A small hosted function holds the project's key on the server side and only forwards explanation requests.
3. **Offline engine.** If neither route answers (no network, a firewall, or an outage), you get a deterministic explanation built from the same payload. It is labeled as offline, and the app never breaks.

`GET /api/agent/status` shows which route is active. Corporate TLS proxies work, because the client uses the operating system's certificate store (`truststore`).

## All commands

```
python -m ledgerclf doctor [--require-model]     environment and artifact health check
python -m ledgerclf split [--force]              label cleanup, group key, holdout seed rule, holdout files
python -m ledgerclf eda                          EDA reports on the training portion
python -m ledgerclf benchmark [--quick]          leaderboard, ablations, validation schemes
python -m ledgerclf tune [--quick]               logged hyperparameter search and nested CV
python -m ledgerclf select                       champion rule, calibration, thresholds, flag policy
python -m ledgerclf train                        final fit on the training portion, artifacts, golden files
python -m ledgerclf evaluate-holdout --confirm   the single holdout evaluation (logged)
python -m ledgerclf predict FILE [--out CSV]
python -m ledgerclf explain FILE --row N [--out JSON]
python -m ledgerclf serve [--host H] [--port P] [--no-browser]
```

Exit codes:

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | input error (the message says how to fix it) |
| 2 | artifact or environment error |
| 3 | internal error (rerun with `--verbose`) |

## Rerun everything from the raw file

```
python -m ledgerclf split
python -m ledgerclf eda
python -m ledgerclf benchmark
python -m ledgerclf tune
python -m ledgerclf select
python -m ledgerclf train
python tools/compare_golden.py --tolerance same-machine
```

`split` reproduces the committed split byte for byte. If anything would change it, `split` refuses to run and explains why. `benchmark` and `tune` take the longest, and `--quick` cuts each to a few minutes. Every command is deterministic: seeds come from `ledgerclf/config.py`, and LightGBM runs single-threaded in deterministic mode.

## Quality checks

```
python -m pytest                     unit, edge-case, integration, adversarial, portability tests
python tools/gate.py --full          formatting, lint, types, tests + coverage, portability, doctor,
                                     edge-case catalog, reference ranges, manifest, notebooks,
                                     golden predictions, hygiene, holdout guard
python tools/run_notebooks.py        execute notebooks/01_eda and 02_preprocessing
```

## Project structure

```
ledgerclf/        the package: io, schema, values, labels, splitting, features, models, evaluation,
                  calibration, benchmark, tuning, selection, training, predict, explain, flags,
                  reference, overlap, service (facade), server (HTTP API), agent, agent_tools, cli
data/raw/         Challenge_Data.xls (never modified)
data/splits/      holdout row ids and split manifest
data/holdout/     holdout_features.xlsx / .csv (upload these in the dashboard)
artifacts/        model_champion.joblib, model_fallback.joblib, metadata.json, reference.json
reports/          every table and chart behind the dashboard and notebooks (reports/manifest.json)
notebooks/        01_eda, 02_preprocessing, 03_modeling (thin; they call the package)
tests/            pytest suite, including 79 catalogued edge cases
tools/            quality gate, golden comparison, wheel check, notebook runner
relay/            the hosted agent relay (Vercel function), for reference
```

## Troubleshooting

- **`doctor` says LightGBM cannot load (macOS).** Run `brew install libomp`. Until then the fallback model is used and predictions still work.
- **`No module named ledgerclf`.** Run `python -m pip install -e . --no-deps` inside the activated virtual environment.
- **Wrong Python version.** Python 3.11 to 3.14 have wheels for every pin. On Windows, use `py -3.12`.
- **The agent says "offline".** No key is set and the relay cannot be reached. Set `ANTHROPIC_API_KEY`, or check the network. The offline explanation uses the same exact numbers.
- **Port 8000 is in use.** Run `python -m ledgerclf serve --port 8010`.
- **A guaranteed environment.** Build and run the Docker image:
  ```
  docker build -t ledgerclf .
  docker run --rm -p 8000:8000 ledgerclf python -m ledgerclf serve --host 0.0.0.0 --no-browser
  ```
