# Ledger Classifier (EY AI Technical Assessment)

Supervised classifier for an anonymized accounts payable ledger (5,899 rows, inputs Col1 to Col7,
six classes), with a validated prediction pipeline, a calibrated log-odds blend of logistic
regression and LightGBM, exact per-row explanations, and an AI explanation agent.

> This README is completed in Phase 8. The install steps below are final.

## Requirements

- Python 3.11 to 3.14 (3.12 recommended). Windows 10/11, macOS 13+, or Linux x86_64.
- macOS only: `brew install libomp` lets LightGBM load. Without it the app still runs on the
  fallback model, and `doctor` says so.

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

The last line registers the package in the virtual environment so that `python -m ledgerclf`
works from any folder. Development tools: `python -m pip install -r requirements-dev.txt`.

## Check the environment

```
python -m ledgerclf doctor
```
