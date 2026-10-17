from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest

from ledgerclf import cli
from ledgerclf.config import Paths, Settings
from ledgerclf.doctor import (
    CheckResult,
    Status,
    check_holdout_files,
    check_lightgbm,
    check_packages,
    check_python,
    check_writable,
    format_report,
    lightgbm_status,
    required_pins,
    run_doctor,
)
from ledgerclf.exceptions import ArtifactError, LedgerClfError
from ledgerclf.logging_setup import AsciiFormatter, configure


def test_doctor_command_passes_on_dev_machine(capsys):
    assert cli.main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert "Summary:" in output
    assert output.isascii()


def test_split_command_prints_counts(capsys, monkeypatch, tmp_path):
    root = _project_copy(tmp_path)
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(paths=Paths.from_root(root)))
    assert cli.main(["split"]) == 0
    output = capsys.readouterr().out
    assert "Holdout seed: 0" in output
    assert (root / "data" / "splits" / "split_manifest.json").exists()


def _project_copy(tmp_path: Path) -> Path:
    root = tmp_path / "project copy, \u00fc"
    (root / "data" / "raw").mkdir(parents=True)
    source = Paths.from_root(Path(__file__).resolve().parents[2]).raw_data
    shutil.copy(source, root / "data" / "raw" / source.name)
    return root


def test_split_in_fresh_copy_is_byte_identical_to_repo(tmp_path, settings):
    from ledgerclf.splitting import make_holdout_split

    root = _project_copy(tmp_path)
    make_holdout_split(Settings(paths=Paths.from_root(root)))
    for relative in [
        "data/splits/holdout_row_ids.csv",
        "data/splits/split_manifest.json",
        "data/holdout/holdout_features.csv",
        "data/holdout/holdout_features.xlsx",
    ]:
        assert (root / relative).read_bytes() == (settings.paths.root / relative).read_bytes(), (
            relative
        )


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ArtifactError("missing"), cli.EXIT_ENVIRONMENT_ERROR),
        (LedgerClfError("bad input"), cli.EXIT_USER_ERROR),
        (RuntimeError("boom"), cli.EXIT_INTERNAL_ERROR),
        (KeyboardInterrupt(), cli.EXIT_USER_ERROR),
    ],
)
def test_cli_maps_errors_to_exit_codes(monkeypatch, capsys, error, expected):
    def failing(args, settings):
        raise error

    monkeypatch.setitem(cli.COMMANDS, "doctor", ("doctor", failing))
    assert cli.main(["doctor"]) == expected
    assert capsys.readouterr().out.isascii()


def test_echo_escapes_non_ascii(capsys):
    cli.echo("caf\u00e9")
    assert capsys.readouterr().out == "caf\\xe9\n"


def test_doctor_flags_unsupported_python(monkeypatch):
    monkeypatch.setattr("ledgerclf.doctor.sys.version_info", (3, 9, 0))
    assert check_python().status is Status.FAIL


def test_doctor_reports_forced_fallback(monkeypatch):
    monkeypatch.setenv("LEDGERCLF_FORCE_FALLBACK", "1")
    available, reason = lightgbm_status()
    assert not available
    assert "LEDGERCLF_FORCE_FALLBACK" in reason
    result = check_lightgbm()
    assert result.status is Status.WARN
    assert result.fix


def test_doctor_warns_on_version_mismatch(tmp_path, settings):
    (tmp_path / "requirements.txt").write_text("numpy==0.0.1\n", encoding="utf-8")
    custom = Settings(paths=Paths.from_root(tmp_path))
    numpy_check = next(c for c in check_packages(custom) if c.name == "package numpy")
    assert numpy_check.status is Status.WARN
    assert required_pins(Settings(paths=Paths.from_root(tmp_path / "missing"))) == {}


def test_doctor_reports_missing_package(monkeypatch, settings):
    monkeypatch.setattr(
        "ledgerclf.doctor.RUNTIME_PACKAGES", (("not-a-real-package", "not_a_real_module"),)
    )
    result = check_packages(settings)[0]
    assert result.status is Status.FAIL


def test_doctor_holdout_and_write_checks(tmp_path):
    custom = Settings(paths=Paths.from_root(tmp_path))
    assert check_holdout_files(custom).status is Status.WARN
    assert check_writable(custom).status is Status.PASS


def test_doctor_write_check_warns_when_read_only(tmp_path, monkeypatch):
    def deny(*args, **kwargs):
        raise OSError("read-only")

    monkeypatch.setattr("ledgerclf.doctor.tempfile.TemporaryFile", deny)
    assert check_writable(Settings(paths=Paths.from_root(tmp_path))).status is Status.WARN


def test_run_doctor_requires_model_when_asked(settings):
    def missing_model(_settings):
        return [CheckResult("artifacts", Status.WARN, "missing", "train")]

    assert run_doctor(settings, require_model=True, model_checks=missing_model).exit_code == 2
    report = run_doctor(settings, require_model=False, model_checks=missing_model)
    assert report.exit_code == 0
    assert any("Fix for artifacts" in line for line in format_report(report))
    assert report.to_dict()["checks"]


def test_ascii_formatter_escapes_non_ascii():
    record = logging.LogRecord("ledgerclf", logging.INFO, __file__, 1, "M\u00fcller", None, None)
    assert AsciiFormatter("%(message)s").format(record) == "M\\xfcller"


def test_configure_does_not_stack_handlers():
    configure(logging.INFO)
    configure(logging.DEBUG)
    handlers = [h for h in logging.getLogger("ledgerclf").handlers if h.get_name()]
    assert len(handlers) == 1
    configure(logging.WARNING)


def test_main_module_runs(monkeypatch):
    import runpy

    monkeypatch.setattr("sys.argv", ["ledgerclf", "--version"])
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("ledgerclf", run_name="__main__")
    assert caught.value.code == 0
