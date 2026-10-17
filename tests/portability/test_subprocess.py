"""Subprocess portability checks: other working directories, LightGBM blocked, cp1252 console."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.portability

ROOT = Path(__file__).resolve().parents[2]
BLOCK_LIGHTGBM = (
    "import sys\n"
    "class _Blocker:\n"
    "    def find_spec(self, name, path=None, target=None):\n"
    "        if name == 'lightgbm' or name.startswith('lightgbm.'):\n"
    "            raise ImportError('lightgbm blocked for this test')\n"
    "        return None\n"
    "sys.meta_path.insert(0, _Blocker())\n"
)


def _run(
    args: list[str], cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    merged = {**os.environ, **(env or {})}
    return subprocess.run(args, cwd=cwd, env=merged, capture_output=True, timeout=300, check=False)


def test_doctor_runs_from_another_directory(tmp_path):
    completed = _run([sys.executable, "-m", "ledgerclf", "doctor"], cwd=tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.isascii()
    assert b"Summary:" in completed.stdout


def test_cli_output_survives_cp1252_console(tmp_path):
    completed = _run(
        [sys.executable, "-m", "ledgerclf", "doctor"],
        cwd=tmp_path,
        env={"PYTHONIOENCODING": "cp1252"},
    )
    assert completed.returncode == 0, completed.stderr


def test_every_module_imports_with_lightgbm_blocked():
    modules = sorted(p.stem for p in (ROOT / "ledgerclf").glob("*.py") if p.stem != "__main__")
    script = BLOCK_LIGHTGBM + "".join(f"import ledgerclf.{name}\n" for name in modules)
    script += "assert 'lightgbm' not in sys.modules\nprint('ok')\n"
    completed = _run([sys.executable, "-c", script], cwd=ROOT)
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert completed.stdout.strip() == b"ok"


def test_doctor_reports_fallback_when_lightgbm_blocked(tmp_path):
    script = BLOCK_LIGHTGBM + "from ledgerclf.cli import main\nraise SystemExit(main(['doctor']))\n"
    completed = _run([sys.executable, "-c", script], cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert b"fallback" in completed.stdout
