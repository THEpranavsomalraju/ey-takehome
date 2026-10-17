from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pandas as pd
import pytest

from ledgerclf.config import Paths, Settings
from ledgerclf.eda import (
    EDA_CATALOG,
    class_distribution,
    col2_type_mix,
    cramers_v,
    run_eda,
    schema_summary,
    signed_log,
)
from ledgerclf.exceptions import ConfigError
from ledgerclf.reports import load_report, read_manifest, to_jsonable, write_json, write_table

DOCS_SHARES = {"Category_1": 0.8840, "Category_2": 0.1070, "Category_3": 0.0039}


@pytest.fixture(scope="module")
def eda_settings(tmp_path_factory, settings):
    reports = tmp_path_factory.mktemp("reports")
    custom = Settings(paths=dataclasses.replace(settings.paths, reports=reports))
    run_eda(custom)
    return custom


def test_run_eda_writes_every_catalog_report(eda_settings):
    listed = {entry["file"] for entry in read_manifest(eda_settings)}
    assert listed == set(EDA_CATALOG)
    for file in EDA_CATALOG:
        assert (eda_settings.paths.reports / file).exists()


def test_structural_facts_match_docs(eda_settings):
    summary = load_report(eda_settings, "eda/schema_summary.json")
    assert summary["rows"] == 5899
    assert summary["column_details"]["Col2"]["python_types"] == {
        "int": 3526,
        "str": 2370,
        "float": 3,
    }
    assert summary["column_details"]["Col4"]["missing"] == 153
    mix = load_report(eda_settings, "eda/col2_type_mix.csv")
    assert mix.set_index("category")["rows"].iloc[-1] == 326
    mapping = load_report(eda_settings, "eda/raw_label_mapping.csv")
    assert len(mapping) == 11 and mapping["rows"].sum() == 5899


def test_training_shares_close_to_docs(eda_settings):
    shares = load_report(eda_settings, "eda/class_distribution.csv").set_index("class")["share"]
    for cls, reference in DOCS_SHARES.items():
        assert abs(shares[cls] - reference) <= 0.01


def test_eda_uses_training_rows_only(eda_settings):
    distribution = load_report(eda_settings, "eda/class_distribution.csv")
    assert distribution["rows"].sum() == 5309
    duplicates = load_report(eda_settings, "eda/duplicates_summary.json")
    assert duplicates["rows"] == 5309


def test_cramers_v_bounds():
    a = pd.Series(list("aabbccaabbcc") * 10)
    assert cramers_v(a, a) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    noise = pd.Series(rng.integers(0, 3, 1000))
    other = pd.Series(rng.integers(0, 3, 1000))
    assert cramers_v(noise, other) < 0.1
    assert cramers_v(pd.Series(["x"] * 5), pd.Series(list("abcde"))) == 0.0


def test_signed_log_and_small_helpers():
    assert signed_log(np.array([-9.0, 9.0])).tolist() == pytest.approx(
        [-math.log(10), math.log(10)]
    )
    table = class_distribution(np.array(["Category_1", "Category_1", "Category_2"], dtype=object))
    assert table.loc[table["class"] == "Category_1", "rows"].item() == 2
    mix = col2_type_mix(pd.Series([1, "4.80P+11", None], dtype=object))
    assert mix["rows"].tolist() == [1, 1, 1]
    resolved = pd.DataFrame(
        {c: ["x"] for c in ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7"]}
    )
    assert schema_summary(resolved, pd.Series(["Category_1"]))["rows"] == 1


def test_to_jsonable_converts_numpy_and_nan():
    payload = {
        "a": np.int64(3),
        "b": np.float32(0.5),
        "c": float("nan"),
        "d": np.array([1, 2]),
        "e": np.bool_(True),
        "f": pd.Timestamp("2018-01-01"),
        "g": (1, None),
    }
    converted = to_jsonable(payload)
    json.dumps(converted, allow_nan=False)
    assert converted["c"] is None and converted["d"] == [1, 2] and converted["e"] is True


def test_report_writers_and_loader(tmp_path):
    custom = Settings(paths=Paths.from_root(tmp_path))
    write_table(custom, "x/table.csv", pd.DataFrame({"a": [1]}), "T", "test", "D")
    write_json(custom, "x/data.json", {"k": 1}, "J", "test", "D")
    assert load_report(custom, "x/table.csv")["a"].tolist() == [1]
    assert load_report(custom, "x/data.json") == {"k": 1}
    with pytest.raises(ConfigError):
        load_report(custom, "x/missing.csv")
    write_table(custom, "x/table.csv", pd.DataFrame({"b": [2]}), "T2", "test", "D")
    entries = read_manifest(custom)
    assert [e["file"] for e in entries] == ["x/data.json", "x/table.csv"]
    assert entries[1]["columns"] == ["b"]
