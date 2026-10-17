"""CLI messages and exit codes for user mistakes: plain English, never a traceback.

Exit codes: 0 success, 1 user input error, 2 artifact or environment error,
3 internal error.
"""

from __future__ import annotations

import pytest

from ledgerclf.cli import main
from ledgerclf.config import EXIT_USER_ERROR
from tests.adversarial._support import ROW_A, ROW_B, csv_text


def test_predict_on_missing_path_says_not_found_instead_of_empty(tmp_path, capsys):
    missing = tmp_path / "no_such_ledger.csv"
    code = main(["predict", str(missing)])
    output = capsys.readouterr().out
    assert code == EXIT_USER_ERROR
    assert "empty" not in output.lower()
    assert "not found" in output.lower()


def test_predict_out_into_missing_folder_is_a_user_error_with_a_fix(tmp_path, capsys):
    source = tmp_path / "ledger.csv"
    source.write_text(csv_text([ROW_A, ROW_B]), encoding="utf-8")
    target = tmp_path / "no_such_folder" / "predictions.csv"
    code = main(["predict", str(source), "--out", str(target)])
    output = capsys.readouterr().out
    assert "Internal error" not in output
    assert code == EXIT_USER_ERROR


def test_explain_with_non_integer_row_exits_with_user_error_code(tmp_path, capsys):
    source = tmp_path / "ledger.csv"
    source.write_text(csv_text([ROW_A, ROW_B]), encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        main(["explain", str(source), "--row", "1.5"])
    capsys.readouterr()
    assert caught.value.code == EXIT_USER_ERROR  # argparse uses 2, the environment-error code
