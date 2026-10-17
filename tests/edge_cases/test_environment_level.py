"""Edge cases E01 to E12 (environment level)."""

from __future__ import annotations

import os
import socket
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from ledgerclf.artifacts import CHAMPION_BUNDLE, FALLBACK_BUNDLE, load_best_available, load_bundle
from ledgerclf.doctor import Status, check_python
from ledgerclf.exceptions import ArtifactError
from ledgerclf.service import PredictionService
from tests.conftest import project_copy

pytestmark = pytest.mark.edge

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
    return subprocess.run(
        args,
        cwd=cwd,
        env={**os.environ, **(env or {})},
        capture_output=True,
        timeout=300,
        check=False,
    )


def test_e01_lightgbm_import_failure_activates_fallback(settings):
    holdout = settings.paths.holdout / "holdout_features.xlsx"
    script = BLOCK_LIGHTGBM + (
        "from ledgerclf.service import PredictionService\n"
        "s = PredictionService.load(auto_train=False)\n"
        f"r = s.predict_file(open(r'{holdout}', 'rb').read(), 'h.xlsx')\n"
        "print(r.model_used, len(r.predictions), 'lightgbm' in sys.modules)\n"
    )
    completed = _run([sys.executable, "-c", script], cwd=ROOT)
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert completed.stdout.split() == [b"fallback_logistic", b"590", b"False"]


def test_e01_fallback_bundle_unpickles_without_lightgbm(settings):
    script = BLOCK_LIGHTGBM + (
        "from ledgerclf.artifacts import load_bundle\n"
        "from ledgerclf.config import get_settings\n"
        f"b = load_bundle('{FALLBACK_BUNDLE}', get_settings().paths)\n"
        "print(b.name, 'lightgbm' in sys.modules)\n"
    )
    completed = _run([sys.executable, "-c", script], cwd=ROOT)
    assert completed.stdout.split() == [b"fallback_logistic", b"False"], completed.stderr


def test_e02_missing_artifacts_without_auto_train_names_the_fix(tmp_path, settings):
    copy = project_copy(settings, tmp_path / "project", with_artifacts=False)
    with pytest.raises(ArtifactError) as caught:
        PredictionService.load(auto_train=False, settings=copy)
    assert "python -m ledgerclf train" in caught.value.message


@pytest.mark.slow
def test_e02_missing_artifacts_are_retrained_automatically(tmp_path, settings):
    copy = project_copy(settings, tmp_path / "project", with_artifacts=False)
    (copy.paths.artifacts).mkdir(parents=True)
    (copy.paths.artifacts / "champion.json").write_bytes(
        (settings.paths.artifacts / "champion.json").read_bytes()
    )
    service = PredictionService.load(auto_train=True, settings=copy)
    assert (copy.paths.artifacts / FALLBACK_BUNDLE).exists()
    result = service.predict_file(
        (copy.paths.holdout / "holdout_features.xlsx").read_bytes(), "h.xlsx"
    )
    assert len(result.predictions) == 590


def test_e03_version_mismatch_warns_and_still_loads(tmp_path, settings, caplog):
    import joblib

    copy = project_copy(settings, tmp_path / "project")
    payload = joblib.load(copy.paths.artifacts / FALLBACK_BUNDLE)
    payload["sklearn_version"] = "0.1.0"
    joblib.dump(payload, copy.paths.artifacts / FALLBACK_BUNDLE)
    with caplog.at_level("WARNING", logger="ledgerclf"):
        logger = __import__("logging").getLogger("ledgerclf")
        logger.propagate = True
        bundle = load_bundle(FALLBACK_BUNDLE, copy.paths)
    assert bundle.name == "fallback_logistic"
    assert "scikit-learn 0.1.0" in caplog.text


def test_e04_commands_work_from_another_directory(tmp_path, settings):
    holdout = settings.paths.holdout / "holdout_features.csv"
    completed = _run([sys.executable, "-m", "ledgerclf", "predict", str(holdout)], cwd=tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert b"Rows predicted: 590" in completed.stdout


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="POSIX permissions only")
def test_e05_read_only_project_predicts_but_refuses_retraining(tmp_path, settings):
    from ledgerclf.training import run_training

    copy = project_copy(settings, tmp_path / "project")
    copy.paths.artifacts.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        service = PredictionService.load(auto_train=False, settings=copy)
        result = service.predict_file(
            (copy.paths.holdout / "holdout_features.csv").read_bytes(), "h.csv"
        )
        assert len(result.predictions) == 590
        with pytest.raises(ArtifactError) as caught:
            run_training(copy)
        assert caught.value.code == "ARTIFACTS_READ_ONLY"
    finally:
        copy.paths.artifacts.chmod(stat.S_IRWXU)


def test_e06_project_path_with_spaces_commas_and_unicode(tmp_path, settings):
    copy = project_copy(settings, tmp_path / "Smith, John_AI Assessment \u00fc\u00e9")
    service = PredictionService.load(auto_train=False, settings=copy)
    result = service.predict_file(
        (copy.paths.holdout / "holdout_features.xlsx").read_bytes(), "h.xlsx"
    )
    assert len(result.predictions) == 590


def test_e07_cli_prints_under_cp1252_console(tmp_path, settings):
    holdout = settings.paths.holdout / "holdout_features.csv"
    completed = _run(
        [sys.executable, "-m", "ledgerclf", "explain", str(holdout), "--row", "1"],
        cwd=tmp_path,
        env={"PYTHONIOENCODING": "cp1252"},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.decode("cp1252").startswith("**Summary:**")


def test_e08_prediction_and_explanation_never_use_the_network(service, holdout_xlsx, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    result = service.predict_file(holdout_xlsx, "h.xlsx")
    assert service.explain(result, 1)["fallback_text"]


def test_e09_unsupported_python_is_reported(monkeypatch):
    monkeypatch.setattr("ledgerclf.doctor.sys.version_info", (3, 10, 0))
    result = check_python()
    assert result.status is Status.FAIL
    assert "3.11" in result.fix


def test_e10_concurrent_predictions_match_sequential(service, holdout_xlsx):
    sequential = service.predict_file(holdout_xlsx, "h.xlsx").predictions
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(lambda _: service.predict_file(holdout_xlsx, "h.xlsx").predictions, range(4))
        )
    for result in results:
        np.testing.assert_array_equal(
            result["predicted_label"].to_numpy(), sequential["predicted_label"].to_numpy()
        )


def test_e11_corrupt_artifact_raises_clear_error_or_falls_back(tmp_path, settings):
    copy = project_copy(settings, tmp_path / "project")
    (copy.paths.artifacts / CHAMPION_BUNDLE).write_bytes(b"not a pickle at all")
    bundle, reason = load_best_available(copy.paths)
    assert bundle.is_fallback and reason
    (copy.paths.artifacts / FALLBACK_BUNDLE).write_bytes(b"\x80\x04garbage")
    with pytest.raises(ArtifactError) as caught:
        load_bundle(FALLBACK_BUNDLE, copy.paths)
    assert "python -m ledgerclf train" in caught.value.message


def test_e12_forced_fallback_environment_variable(monkeypatch, settings, holdout_xlsx):
    monkeypatch.setenv("LEDGERCLF_FORCE_FALLBACK", "1")
    service = PredictionService.load(auto_train=False, settings=settings)
    result = service.predict_file(holdout_xlsx, "h.xlsx")
    assert result.model_used == "fallback_logistic"
    assert "LEDGERCLF_FORCE_FALLBACK" in result.fallback_reason
