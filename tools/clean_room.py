"""Clean-room check: does the submission ZIP install and run on a machine like the grader's?

Usage:
    python tools/clean_room.py [--keep] [--skip-train]

Steps: build the ZIP, extract it into a folder whose name has spaces, a comma,
and accented letters, create a fresh virtual environment, install
requirements.txt, run doctor, predict, the golden comparison, and explain;
install requirements-dev.txt and run the fast tests; retrain from scratch and
compare with the golden file; uninstall LightGBM and check the fallback path.
Writes dist/clean_room.json. Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOLDER_NAME = "clean room, Unicode \u00fcml\u00e4ut test"
STEP_TIMEOUT_SECONDS = 1800


def _venv_python(venv: Path) -> Path:
    scripts = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" else ""
    return venv / scripts / f"python{suffix}"


def _run(
    name: str, command: list[str], cwd: Path, results: list[dict], env: dict | None = None
) -> bool:
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, **(env or {})},
        timeout=STEP_TIMEOUT_SECONDS,
        check=False,
    )
    ok = completed.returncode == 0
    tail = (completed.stdout + completed.stderr).strip().splitlines()[-6:]
    results.append(
        {"step": name, "ok": ok, "seconds": round(time.time() - started, 1), "output_tail": tail}
    )
    print(f"{'PASS' if ok else 'FAIL'}  {name} ({time.time() - started:.0f}s)", flush=True)
    if not ok:
        print("      " + "\n      ".join(tail), flush=True)
    return ok


def _extract(work: Path) -> Path:
    built = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "package_submission.py"),
            "--last",
            "Test",
            "--first",
            "Clean",
            "--out",
            str(work),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    print(built.stdout.strip(), flush=True)
    archive = next(work.glob("*.zip"))
    target = work / FOLDER_NAME
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(target)
    return next(p for p in target.iterdir() if p.is_dir())


def main() -> int:
    """Run every clean-room step and report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep", action="store_true", help="keep the temporary folder")
    parser.add_argument("--skip-train", action="store_true", help="skip the retraining step")
    args = parser.parse_args()
    work = Path(tempfile.mkdtemp(prefix="ledgerclf_"))
    results: list[dict] = []
    try:
        project = _extract(work)
        venv = work / "venv with space"
        _run("create venv", [sys.executable, "-m", "venv", str(venv)], work, results)
        py = str(_venv_python(venv))
        pip = [py, "-m", "pip", "install", "--disable-pip-version-check", "-q"]
        _run("install requirements.txt", [*pip, "-r", "requirements.txt"], project, results)
        _run("install package", [*pip, "-e", ".", "--no-deps"], project, results)
        outside = work  # run commands from outside the project folder
        holdout = str(project / "data" / "holdout" / "holdout_features.xlsx")
        _run("doctor", [py, "-m", "ledgerclf", "doctor", "--require-model"], outside, results)
        _run(
            "predict",
            [py, "-m", "ledgerclf", "predict", holdout, "--out", str(work / "predictions.csv")],
            outside,
            results,
        )
        _run("explain", [py, "-m", "ledgerclf", "explain", holdout, "--row", "1"], outside, results)
        golden = [py, "tools/compare_golden.py", "--tolerance", "cross-platform"]
        _run("golden comparison", golden, project, results)
        _run("install requirements-dev.txt", [*pip, "-r", "requirements-dev.txt"], project, results)
        _run(
            "fast tests",
            [py, "-m", "pytest", "-m", "not slow", "-q", "--no-cov", "-p", "no:cacheprovider"],
            project,
            results,
        )
        _run("notebooks", [py, "tools/run_notebooks.py"], project, results)
        if not args.skip_train:
            shutil.move(project / "artifacts", work / "artifacts_shipped")
            shutil.copytree(
                work / "artifacts_shipped",
                project / "artifacts",
                ignore=shutil.ignore_patterns("model_*.joblib"),
            )
            _run("retrain from scratch", [py, "-m", "ledgerclf", "train"], project, results)
            shipped_golden = work / "artifacts_shipped" / "golden_holdout_predictions.csv"
            shutil.copy2(shipped_golden, project / "artifacts" / shipped_golden.name)
            _run("retrained model matches shipped predictions", golden, project, results)
        _run(
            "uninstall lightgbm",
            [py, "-m", "pip", "uninstall", "-y", "-q", "lightgbm"],
            project,
            results,
        )
        _run("doctor without lightgbm", [py, "-m", "ledgerclf", "doctor"], outside, results)
        _run(
            "predict without lightgbm",
            [py, "-m", "ledgerclf", "predict", holdout],
            outside,
            results,
        )
    finally:
        report = ROOT / "dist" / "clean_room.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
    failed = [r["step"] for r in results if not r["ok"]]
    print(f"Clean room: {'PASS' if not failed else 'FAIL: ' + ', '.join(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
