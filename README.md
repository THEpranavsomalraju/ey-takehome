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

1. **Data & Model tab.** Goes from the raw data to the final model: exploration, cleaning and features, how the data was split and tested, model choice, calibration, and the holdout result. Each chart has an **Explain** button if you want the AI to walk through it.
2. **Classify & Explain tab.** Upload `data/holdout/holdout_features.xlsx`, or click *Use the holdout sample*. Each line gets a category, a confidence, and a review decision. **Explain** on a row shows what drove the prediction, similar past invoices, and a short write-up from the agent.
3. **Notebooks** in `notebooks/` (EDA, preprocessing, modeling). Each one reads on its own.
4. **`MODEL_CARD.md`** for results, design choices, and limitations on one page.

If a term is unfamiliar, hover the dotted word or open **Guide** in the header.

## How it works

Uploads can be Excel or CSV, with the original `Col1`..`Col7` headers, business names like Vendor, Amount, or GL account, or no header at all. The reader also copes with reordered columns, other encodings, currency strings, European number formats, and mixed date formats. Anything odd becomes a warning on the row instead of an error, so every non-empty line still gets a prediction.

Features are TF-IDF words for each text column, indicators for recurring vendors, accounts, and vendor-account pairs, the amount as a signed log with credit and round-number flags, and the posting month. Col2 is an invoice number, so it isn't used.

The model is a blend of logistic regression and LightGBM, averaged in log-odds and then temperature-calibrated. Both parts are additive, so each prediction breaks down into per-input contributions, which is what the explanations show.

I set aside 10% of the data before looking at anything and scored it once at the end. Everything else used grouped cross-validation: about a quarter of the training rows (25.7%) match another row on everything except the invoice number, and a random split would have tested the model on rows it had effectively already seen (it inflates accuracy by about 1.4 points).

Lines under 80% confidence, or with a critical warning such as a new vendor or GL account, go to review. They still get a prediction.

## The AI agent

The agent only runs when you click Explain. It works from the model's contribution figures and the training history, and the app rejects an answer that quotes a number not found in those figures. It calls Claude (`claude-haiku-5-5`) using the first option that works:

1. **Your own key**, if you have one: paste it into the dashboard settings, or set `ANTHROPIC_API_KEY` in the environment or a `.env` file.
2. **The project relay**, with no setup needed. This is a small Vercel function (`relay/`) that holds my key and only forwards this app's two prompts, with a rate limit and a spend cap. It's there so you don't need a key to try the agent.
3. **Offline.** If Claude can't be reached (no internet, relay down), the app builds the explanation itself from the same figures and labels it as offline. The rest of the app doesn't depend on the relay.

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
