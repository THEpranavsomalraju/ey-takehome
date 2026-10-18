# Ledger Classifier: EY AI Technical Assessment

A model that suggests the category of each accounts payable invoice line, with a dashboard to explore how it was built, classify a file, and have an AI agent explain any prediction.

| | |
|---|---|
| Holdout accuracy (590 lines set aside before any work, scored once) | **97.1%** |
| Expected accuracy on new data (grouped cross-validation) | **95.4%** |
| Always guessing the most common category | 88.4% |
| Lines sent to a person for review / accuracy of the rest (holdout) | 8.8% / 98.9% |

## Run it

Python 3.11 to 3.14. On macOS, run `brew install libomp` first so LightGBM can load (without it the app still runs, on a logistic-regression fallback).

```
python3 -m venv .venv                  # Windows: py -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m ledgerclf serve              # opens http://127.0.0.1:8000
```

Nothing needs training or building: the trained models, reports, and built dashboard are included. `python -m ledgerclf doctor` checks the setup if anything looks wrong.

## What to look at

1. **Data & Model tab.** From raw data to final model: exploration, cleaning and features, the split and testing, model choice, calibration, and the holdout result. Each chart has an **Explain** button that asks the AI about it.
2. **Classify & Explain tab.** Upload `data/holdout/holdout_features.xlsx` (or click *Use the holdout sample*). Every line gets a category, a confidence, and a review decision. Click **Explain** on any row for the AI agent's explanation, the inputs that drove the prediction, and supporting evidence from past invoices.
3. **Notebooks** (`notebooks/01_eda`, `02_preprocessing`, `03_modeling`), each readable on its own.
4. **`MODEL_CARD.md`**: results, design choices, and limitations on one page.

New to the terms? Open **Guide** in the dashboard header, or hover any dotted word.

## How it works

- **Validation.** Excel or CSV, with the original `Col1`..`Col7` headers, business names (Vendor, Amount, Posting date, GL account, ...), or no header at all. It also handles reordered columns, other encodings, currency strings, European numbers, and mixed date formats. Problems become warnings; every non-empty line gets a prediction.
- **Features.** Words per text column (TF-IDF), recurring vendors, accounts and vendor-account pairs, the amount (signed log plus credit and round flags), and the posting month. Col2, the invoice number, is an identifier and is not used.
- **Model.** A log-odds blend of logistic regression and LightGBM, calibrated so stated confidence matches observed accuracy. Each prediction splits exactly into per-input contributions.
- **Testing.** 10% held out before any analysis. All development used grouped cross-validation, because a quarter of the rows are exact copies of other rows and a random split would test the model on rows it had already seen.
- **Review.** Lines below 80% confidence, or with a critical warning such as a new vendor or GL account, go to a person. Warnings never block a prediction.

## The AI agent

The agent runs only when asked. It explains a row from the model's exact figures and the training history, and every number it quotes is checked against them. It uses Claude (`claude-haiku-5-5`) through the first route that works:

1. **Your own key** (optional): typed into the dashboard settings, or `ANTHROPIC_API_KEY` in the environment or a `.env` file.
2. **The project relay** (no setup): a small hosted function (`relay/`) that holds the project's key and forwards only this app's explanation requests. It exists so graders don't need their own key. It only accepts this app's two prompts and has a rate limit and a spend cap.
3. **Offline**: if Claude is unreachable (no internet, relay down, key revoked), the app writes a plain explanation from the same figures and labels it as offline. Nothing else in the app depends on the relay.

## Command line

```
python -m ledgerclf predict FILE [--out predictions.csv]
python -m ledgerclf explain FILE --row N
python -m ledgerclf serve [--port 8010]
```

Rebuild everything from the raw file (deterministic, same results):

```
python -m ledgerclf split
python -m ledgerclf eda
python -m ledgerclf benchmark          # add --quick for a short run
python -m ledgerclf tune
python -m ledgerclf select
python -m ledgerclf train
python tools/compare_golden.py --tolerance same-machine
```

Tests: `python -m pip install -r requirements-dev.txt`, then `python -m pytest`. Most of the suite covers messy input files: odd headers, encodings, broken cells, and files built to break the reader. Re-run the notebooks with `python tools/run_notebooks.py`.

## Project layout

```
ledgerclf/    Python package: reading and validation, features, models, evaluation,
              prediction service, HTTP API (server.py), AI agent (agent.py, chart_explainer.py)
frontend/     dashboard source (React, shadcn/ui); built files in frontend/dist
notebooks/    exploration, preprocessing, modeling
data/         raw file, holdout split, holdout files to upload
artifacts/    trained models and training statistics
reports/      every table behind the notebooks and dashboard
relay/        hosted relay for the AI agent
tests/        test suite
```

## Troubleshooting

- **LightGBM cannot load (macOS):** `brew install libomp`. Until then the fallback model (95.3% holdout accuracy) runs.
- **`No module named ledgerclf`:** run `python -m pip install -e . --no-deps` in the activated environment.
- **Agent says "offline":** no internet or a firewall; the offline explanation uses the same figures.
- **Port in use:** `python -m ledgerclf serve --port 8010`.
- **Docker:** `docker build -t ledgerclf .` then `docker run --rm -p 8000:8000 ledgerclf python -m ledgerclf serve --host 0.0.0.0 --no-browser`.
