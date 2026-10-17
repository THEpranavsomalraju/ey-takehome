"""read_table with path sources that cannot be read must raise FileReadError, not OSError."""

from __future__ import annotations

import contextlib
import os
import sys

import pytest

from ledgerclf.config import get_settings
from ledgerclf.exceptions import FileReadError
from ledgerclf.io import read_table
from tests.adversarial._support import ROW_A, csv_text


def test_missing_path_raises_file_read_error(tmp_path):
    with pytest.raises(FileReadError):
        read_table(tmp_path / "does_not_exist.csv", "does_not_exist.csv", get_settings().limits)


def test_directory_path_raises_file_read_error(tmp_path):
    with pytest.raises(FileReadError):
        read_table(tmp_path, "folder.csv", get_settings().limits)


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="chmod 000 not enforced")
def test_unreadable_path_raises_file_read_error(tmp_path):
    path = tmp_path / "locked.csv"
    path.write_text(csv_text([ROW_A]), encoding="utf-8")
    path.chmod(0)
    try:
        with pytest.raises(FileReadError):
            read_table(path, path.name, get_settings().limits)
    finally:
        path.chmod(0o600)


def test_string_path_is_read_or_rejected_with_file_read_error(tmp_path):
    path = tmp_path / "ok.csv"
    path.write_text(csv_text([ROW_A]), encoding="utf-8")
    # A clear rejection is acceptable; a raw TypeError is not.
    with contextlib.suppress(FileReadError):
        read_table(str(path), path.name, get_settings().limits)  # type: ignore[arg-type]
