"""Shared fixtures: settings, the labeled raw file, training rows, and file factories."""

from __future__ import annotations

import csv
import io
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import pytest

from ledgerclf.config import Settings, get_settings
from ledgerclf.io import read_table
from ledgerclf.schema import INPUT_COLUMNS, LABEL_COLUMN, coerce_values, resolve_columns
from ledgerclf.splitting import LabeledFrame, TrainingData, load_labeled_raw, load_training_frame

HEADER = list(INPUT_COLUMNS)
SAMPLE_ROWS: list[list[object]] = [
    [
        "Word16 Word17 Word18 Word7",
        "4.80P+11",
        1200.0,
        "Word61",
        "2018-08-31",
        "Word563 Word564 Word9 Word26",
        "NoDoc",
    ],
    [
        "Word19 Word20 Word21 Word22 Word23 Word5",
        "10719",
        617.6,
        "Word562",
        "2018-07-31",
        "Word563 Word575",
        "NoDoc",
    ],
    [
        "Word8 Word9 Word10 Word11 Word5",
        "65975",
        75.0,
        None,
        "2018-10-31",
        "Word563 Word564 Word9 Word26",
        "Doc1",
    ],
]


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings()


@pytest.fixture(scope="session")
def labeled_raw(settings: Settings) -> LabeledFrame:
    return load_labeled_raw(settings.paths, settings.limits)


@pytest.fixture(scope="session")
def training(settings: Settings) -> TrainingData:
    return load_training_frame(settings)


def process(source: Path | bytes, file_name: str, settings: Settings | None = None):
    """Run the shared validation path: read, resolve, coerce. Returns (frame, issues, info)."""
    settings = settings or get_settings()
    raw, info, read_issues = read_table(source, file_name, settings.limits)
    resolved, _, resolve_issues = resolve_columns(raw, settings.limits)
    frame, coerce_issues = coerce_values(resolved, settings.limits)
    return frame, [*read_issues, *resolve_issues, *coerce_issues], info


def codes(issues) -> list[str]:
    return [i.code for i in issues]


def csv_bytes(
    rows: Sequence[Sequence[object]],
    header: Sequence[str] | None = HEADER,
    delimiter: str = ",",
    encoding: str = "utf-8",
    line_ending: str = "\n",
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter=delimiter, lineterminator=line_ending)
    if header is not None:
        writer.writerow(header)
    for row in rows:
        writer.writerow(["" if v is None else v for v in row])
    return buffer.getvalue().encode(encoding)


def xlsx_bytes(sheets: dict[str, pd.DataFrame], header: bool = True) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False, header=header)
    return buffer.getvalue()


def sample_frame(
    rows: Sequence[Sequence[object]] = SAMPLE_ROWS, with_label: bool = False
) -> pd.DataFrame:
    frame = pd.DataFrame([list(r) for r in rows], columns=HEADER)
    if with_label:
        frame[LABEL_COLUMN] = ["Category_1"] * len(frame)
    return frame


@pytest.fixture
def write_file(tmp_path: Path) -> Callable[[str, bytes], Path]:
    def _write(name: str, data: bytes) -> Path:
        path = tmp_path / name
        path.write_bytes(data)
        return path

    return _write
