"""Report files and the report manifest (docs/02 section 7).

Every table or chart the dashboard and notebooks show comes from a file
under ``reports/``, listed in ``reports/manifest.json`` with a title, the
producing command, a description, and its columns.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ledgerclf.config import Settings
from ledgerclf.exceptions import ConfigError

MANIFEST_NAME = "manifest.json"


def to_jsonable(value: Any) -> Any:
    """Convert numpy and pandas values to native, JSON-safe Python values.

    NaN and infinities become None so that ``json.dumps`` output stays strict.
    """
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.bool_ | bool):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating | float):
        number = float(value)
        return number if math.isfinite(number) else None
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_table(
    settings: Settings,
    file: str,
    frame: pd.DataFrame,
    title: str,
    producer: str,
    description: str,
) -> Path:
    """Write a CSV report and register it in the manifest.

    Args:
        settings: Project settings.
        file: Path relative to ``reports/`` (for example ``eda/class_distribution.csv``).
        frame: The table.
        title: Short title for the dashboard.
        producer: CLI command that produces the file.
        description: One or two sentences explaining the table.

    Returns:
        The written path.
    """
    path = settings.paths.reports / file
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    register(settings, file, title, producer, description, [str(c) for c in frame.columns])
    return path


def write_json(
    settings: Settings,
    file: str,
    payload: dict[str, Any],
    title: str,
    producer: str,
    description: str,
) -> Path:
    """Write a JSON report (sorted keys, strict JSON) and register it in the manifest."""
    path = settings.paths.reports / file
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    register(settings, file, title, producer, description, sorted(str(k) for k in payload))
    return path


def register(
    settings: Settings,
    file: str,
    title: str,
    producer: str,
    description: str,
    columns: list[str],
) -> None:
    """Add or replace one manifest entry, keeping entries sorted by file."""
    entries = {entry["file"]: entry for entry in read_manifest(settings)}
    entries[file] = {
        "file": file,
        "title": title,
        "producer": producer,
        "description": description,
        "columns": columns,
    }
    manifest = {"reports": [entries[name] for name in sorted(entries)]}
    path = settings.paths.reports / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_manifest(settings: Settings) -> list[dict[str, Any]]:
    """Return the manifest entries, or an empty list when no manifest exists."""
    path = settings.paths.reports / MANIFEST_NAME
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))["reports"]
    return entries


def load_report(settings: Settings, file: str) -> pd.DataFrame | dict[str, Any]:
    """Load one report by its manifest path.

    Args:
        settings: Project settings.
        file: Path relative to ``reports/``.

    Returns:
        A DataFrame for CSV reports, a dict for JSON reports.

    Raises:
        ConfigError: When the file is not listed or does not exist.
    """
    listed = {entry["file"] for entry in read_manifest(settings)}
    path = settings.paths.reports / file
    if file not in listed or not path.exists():
        raise ConfigError(
            f"Report '{file}' is not available. Run the command that produces it "
            "(see reports/manifest.json), for example: python -m ledgerclf eda",
            code="REPORT_MISSING",
        )
    if path.suffix == ".json":
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return loaded
    return pd.read_csv(path, encoding="utf-8", keep_default_na=False, na_values=[""])
