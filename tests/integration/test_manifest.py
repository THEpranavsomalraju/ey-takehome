"""Manifest integrity: every listed report exists and every report file is listed."""

from __future__ import annotations

import json

import pandas as pd


def _manifest(settings):
    return json.loads((settings.paths.reports / "manifest.json").read_text(encoding="utf-8"))


def test_every_listed_report_exists(settings):
    missing = [
        e["file"]
        for e in _manifest(settings)["reports"]
        if not (settings.paths.reports / e["file"]).exists()
    ]
    assert not missing, missing


def test_every_report_file_is_listed(settings):
    listed = {e["file"] for e in _manifest(settings)["reports"]}
    on_disk = {
        p.relative_to(settings.paths.reports).as_posix()
        for p in settings.paths.reports.rglob("*")
        if p.suffix in (".csv", ".json") and p.name != "manifest.json"
    }
    assert on_disk == listed, sorted(on_disk ^ listed)


def test_listed_columns_match_csv_headers(settings):
    for entry in _manifest(settings)["reports"]:
        if entry["file"].endswith(".csv"):
            header = pd.read_csv(settings.paths.reports / entry["file"], nrows=0).columns.tolist()
            assert header == entry["columns"], entry["file"]


def test_entries_have_titles_and_descriptions(settings):
    for entry in _manifest(settings)["reports"]:
        assert entry["title"] and entry["description"] and entry["producer"], entry["file"]
