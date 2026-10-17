from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ledgerclf.config import HoldoutRule
from ledgerclf.exceptions import ConfigError, HoldoutAccessError
from ledgerclf.schema import INPUT_COLUMNS, LABEL_COLUMN, MODEL_INPUT_COLUMNS, Severity
from ledgerclf.splitting import (
    HOLDOUT_IDS_FILE,
    MANIFEST_FILE,
    ROW_ID,
    HoldoutAccessToken,
    build_group_key,
    issue_holdout_token,
    load_split,
    out_of_time_mask,
    read_holdout_labels,
    select_holdout_seed,
    vendor_groups,
)
from tests.conftest import process


def test_group_key_reproduces_duplicate_counts(labeled_raw):
    frame = labeled_raw.frame
    all_inputs = frame[list(INPUT_COLUMNS)].fillna("__missing__").astype(str)
    assert all_inputs.duplicated().sum() == 680
    groups = labeled_raw.groups
    assert len(groups) - len(np.unique(groups)) == 1519


def test_date_level_duplicates_match_docs(settings):
    raw = pd.read_excel(settings.paths.raw_data, engine="xlrd")
    assert raw[list(MODEL_INPUT_COLUMNS)].astype(str).duplicated().sum() == 1508


def test_one_conflicting_group_on_all_inputs(labeled_raw):
    frame = labeled_raw.frame[list(INPUT_COLUMNS)].fillna("__missing__").astype(str)
    key = frame.agg("|".join, axis=1)
    labels = pd.Series(labeled_raw.y)
    mixed = labels.groupby(key.to_numpy()).nunique()
    assert (mixed > 1).sum() == 1
    assert labels.groupby(key.to_numpy()).size()[mixed > 1].iloc[0] == 5


def test_group_key_ignores_invoice_number():
    frame = pd.DataFrame(
        {
            "Col1": ["a", "a"],
            "Col2": ["1", "2"],
            "Col3": ["x", "x"],
            "Col4": [None, None],
            "Col5": ["2018-08", "2018-08"],
            "Col6": ["b", "b"],
            "Col7": ["NoDoc", "NoDoc"],
            "amount": [5.0, 5.0],
            "period": [24224.0, 24224.0],
        }
    )
    assert build_group_key(frame).tolist() == [0, 0]


def test_persisted_split_satisfies_rule(settings, labeled_raw):
    split = load_split(settings.paths)
    rule = settings.holdout_rule
    held = labeled_raw.y[np.isin(labeled_raw.frame[ROW_ID], split.holdout_ids)]
    assert rule.min_rows <= len(held) <= rule.max_rows
    for cls in rule.required_classes:
        assert (held == cls).sum() >= 1
    assert (held == "Category_5").sum() == 0
    for cls in rule.share_classes:
        assert abs((held == cls).mean() - (labeled_raw.y == cls).mean()) <= rule.share_tolerance
    assert split.manifest["seed"] == 0


def test_no_group_on_both_sides(settings, labeled_raw):
    split = load_split(settings.paths)
    in_holdout = np.isin(labeled_raw.frame[ROW_ID], split.holdout_ids)
    assert not set(labeled_raw.groups[in_holdout]) & set(labeled_raw.groups[~in_holdout])


def test_seed_rule_is_deterministic(labeled_raw, settings):
    first = select_holdout_seed(labeled_raw.y, labeled_raw.groups, settings.holdout_rule)
    second = select_holdout_seed(labeled_raw.y, labeled_raw.groups, settings.holdout_rule)
    assert first.seed == second.seed
    np.testing.assert_array_equal(first.holdout_idx, second.holdout_idx)


def test_seed_rule_raises_when_unsatisfiable(labeled_raw):
    impossible = HoldoutRule(min_rows=10, max_rows=11, max_seed=3)
    with pytest.raises(ConfigError):
        select_holdout_seed(labeled_raw.y, labeled_raw.groups, impossible)


def test_split_files_match_manifest_hash(settings):
    from ledgerclf.io import sha256_file, sha256_text_file

    manifest = json.loads((settings.paths.splits / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["holdout_row_ids_sha256"] == sha256_text_file(
        settings.paths.splits / HOLDOUT_IDS_FILE
    )
    assert manifest["raw_data_sha256"] == sha256_file(settings.paths.raw_data)


def test_training_frame_excludes_holdout(training, settings):
    split = load_split(settings.paths)
    assert not set(training.frame[ROW_ID]) & set(split.holdout_ids)
    assert len(training.frame) == len(training.y) == len(training.groups) == 5309


@pytest.mark.parametrize("extension", ["xlsx", "csv"])
def test_holdout_files_read_without_errors(settings, extension):
    path = settings.paths.holdout / f"holdout_features.{extension}"
    frame, issues, _ = process(path, path.name)
    assert len(frame) == 590
    assert not [i for i in issues if i.severity is Severity.ERROR]
    assert LABEL_COLUMN not in frame.columns


def test_raw_and_holdout_files_coerce_identically(settings, labeled_raw):
    split = load_split(settings.paths)
    expected = labeled_raw.frame.set_index(ROW_ID).loc[split.holdout_ids]
    compare = [*INPUT_COLUMNS, "amount", "period"]
    for extension in ("xlsx", "csv"):
        path = settings.paths.holdout / f"holdout_features.{extension}"
        frame, _, _ = process(path, path.name)
        pd.testing.assert_frame_equal(
            frame[compare].reset_index(drop=True),
            expected[compare].reset_index(drop=True),
            check_dtype=False,
        )


def test_out_of_time_mask_uses_cutoff_month(training, settings):
    mask = out_of_time_mask(training.frame, settings.oot_cutoff)
    cutoff = 2018 * 12 + 9
    assert (training.frame.loc[mask, "period"] <= cutoff).all()
    assert (training.frame.loc[~mask, "period"] > cutoff).all()
    assert 0 < (~mask).sum() < len(mask)


def test_vendor_groups_are_case_insensitive():
    frame = pd.DataFrame({"Col1": ["Word1", "word1", "Word2", None]})
    groups = vendor_groups(frame)
    assert groups[0] == groups[1] != groups[2]


def test_holdout_guard_requires_confirmation():
    with pytest.raises(HoldoutAccessError):
        issue_holdout_token(confirmed=False, reason="test")


def test_holdout_guard_rejects_forged_tokens(settings):
    with pytest.raises(HoldoutAccessError):
        HoldoutAccessToken(object(), "forged")
    with pytest.raises(HoldoutAccessError):
        read_holdout_labels(settings.paths, token="not a token")


def test_holdout_guard_accepts_issued_token(tmp_path):
    from ledgerclf.config import Paths
    from ledgerclf.splitting import HOLDOUT_LABELS_FILE

    paths = Paths.from_root(tmp_path)
    paths.holdout.mkdir(parents=True)
    (paths.holdout / HOLDOUT_LABELS_FILE).write_text(
        "row_number,ClassificationLabel\n1,Category_1\n", encoding="utf-8"
    )
    token = issue_holdout_token(confirmed=True, reason="guard test")
    assert len(read_holdout_labels(paths, token)) == 1


def _copy_project(tmp_path, settings):
    import shutil

    root = tmp_path / "copy"
    for relative in ("data/raw", "data/splits"):
        shutil.copytree(settings.paths.root / relative, root / relative)
    from ledgerclf.config import Paths, Settings

    return Settings(paths=Paths.from_root(root))


def test_split_refuses_to_change_persisted_holdout(tmp_path, settings):
    from ledgerclf.splitting import make_holdout_split

    copy = _copy_project(tmp_path, settings)
    ids_path = copy.paths.splits / HOLDOUT_IDS_FILE
    ids = pd.read_csv(ids_path)
    ids.iloc[:-1].to_csv(ids_path, index=False)
    with pytest.raises(ConfigError) as caught:
        make_holdout_split(copy)
    assert caught.value.code == "SPLIT_WOULD_CHANGE"
    make_holdout_split(copy, force=True)
    assert len(pd.read_csv(ids_path)) == len(ids)


def test_load_split_detects_tampered_files(tmp_path, settings):
    copy = _copy_project(tmp_path, settings)
    ids_path = copy.paths.splits / HOLDOUT_IDS_FILE
    ids_path.write_text(ids_path.read_text(encoding="utf-8") + "5000\n", encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        load_split(copy.paths)
    assert caught.value.code == "SPLIT_HASH_MISMATCH"


def test_load_split_explains_missing_split(tmp_path):
    from ledgerclf.config import Paths

    with pytest.raises(ConfigError) as caught:
        load_split(Paths.from_root(tmp_path))
    assert "python -m ledgerclf split" in caught.value.message


def test_unmapped_training_label_stops_the_split(tmp_path, settings):
    from ledgerclf.config import Paths, Settings
    from ledgerclf.exceptions import SchemaError
    from ledgerclf.splitting import make_holdout_split
    from tests.conftest import sample_frame, xlsx_bytes

    paths = Paths.from_root(tmp_path)
    paths.raw_data.parent.mkdir(parents=True)
    bad = sample_frame(with_label=True)
    bad.loc[1, "ClassificationLabel"] = "Category_9"
    paths.raw_data.write_bytes(xlsx_bytes({"data": bad}))
    with pytest.raises(SchemaError) as caught:
        make_holdout_split(Settings(paths=paths))
    assert caught.value.code == "UNMAPPED_TRAINING_LABEL"
    assert "Category_9" in caught.value.message


def test_split_refuses_a_raw_file_changed_outside_the_group_key(tmp_path, settings):
    from ledgerclf.splitting import make_holdout_split

    copy = _copy_project(tmp_path, settings)
    raw = pd.read_excel(copy.paths.raw_data, engine="xlrd", dtype=object)
    raw.loc[0, "Col2"] = "EDITED-INVOICE"
    from tests.conftest import xlsx_bytes

    copy.paths.raw_data.write_bytes(xlsx_bytes({"data": raw}))
    with pytest.raises(ConfigError) as caught:
        make_holdout_split(copy)
    assert caught.value.code == "SPLIT_WOULD_CHANGE"
    assert "raw data file" in caught.value.message


def test_raw_cells_are_looked_up_by_spreadsheet_row(labeled_raw):
    ids = np.array([5, 0, 42])
    rows = labeled_raw.raw_rows(ids)
    assert rows["source_row"].tolist() == [7, 2, 44]
    by_frame = labeled_raw.frame.set_index(ROW_ID).loc[ids, "Col1"].tolist()
    assert rows["Col1"].tolist() == by_frame


def test_training_frame_carries_no_label_column(training):
    assert LABEL_COLUMN not in training.frame.columns
    assert LABEL_COLUMN not in training.raw.columns
    assert (training.raw["source_row"].to_numpy() == training.frame["source_row"].to_numpy()).all()


def test_split_hash_ignores_windows_line_endings(settings, tmp_path):
    from ledgerclf.io import sha256_text_file

    original = settings.paths.splits / HOLDOUT_IDS_FILE
    converted = tmp_path / HOLDOUT_IDS_FILE
    converted.write_bytes(original.read_bytes().replace(b"\n", b"\r\n"))
    assert sha256_text_file(converted) == sha256_text_file(original)
