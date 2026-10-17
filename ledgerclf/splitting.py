"""Group key, holdout seed rule, persisted holdout split, CV splitters, holdout guard.

The holdout split is computed once by the ``split`` command, written to
``data/splits``, and only ever re-read afterwards (docs/04 section 2).
Holdout labels open only through ``read_holdout_labels`` with a token that
the ``evaluate-holdout`` command alone creates (CLAUDE.md rule 1).
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from ledgerclf.config import HOLDOUT_FEATURES_STEM, HoldoutRule, Limits, Paths, Settings
from ledgerclf.exceptions import ConfigError, HoldoutAccessError, SchemaError
from ledgerclf.io import read_table, sha256_file, write_xlsx_deterministic
from ledgerclf.labels import CANONICAL_CLASSES, normalize_labels
from ledgerclf.schema import (
    AMOUNT,
    COL_ACCOUNT,
    COL_DESCRIPTION,
    COL_DOC_TYPE,
    COL_INVOICE,
    COL_VENDOR,
    INPUT_COLUMNS,
    LABEL_COLUMN,
    MISSING_TOKEN,
    PERIOD,
    ROW_NUMBER,
    SOURCE_ROW,
    coerce_values,
    resolve_columns,
)
from ledgerclf.values import is_missing, period_index, period_label

logger = logging.getLogger(__name__)

ROW_ID = "row_id"
HOLDOUT_IDS_FILE = "holdout_row_ids.csv"
MANIFEST_FILE = "split_manifest.json"
HOLDOUT_LABELS_FILE = "holdout_labels.csv"
HOLDOUT_WITH_LABELS_FILE = "holdout_with_labels.xlsx"
_GROUP_KEY_SEPARATOR = "\x1f"
_GUARD_MESSAGE = (
    "Holdout labels are only available to the evaluate-holdout command. Ask the project "
    "owner before running: python -m ledgerclf evaluate-holdout --confirm"
)


@contextmanager
def small_class_warning_suppressed() -> Iterator[None]:
    """Silence sklearn's warning about classes smaller than ``n_splits``.

    Category_5 has 2 rows, fewer than any fold count used here, so sklearn
    warns on every split. The warning carries no new information.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The least populated class", category=UserWarning)
        yield


def build_group_key(frame: pd.DataFrame) -> np.ndarray:
    """Return integer group ids from every model input except Col2.

    Rows identical on Col1, Col3, Col4, Col5 (monthly period), Col6, and Col7
    share a group, so copies never land on both sides of a split. Col2 stays
    out because many copies differ only by invoice number (DECISIONS.md D02, D22).

    Args:
        frame: Coerced frame from ``schema.coerce_values``.

    Returns:
        Integer group ids in order of first appearance.
    """
    parts = [
        frame[COL_VENDOR].fillna(MISSING_TOKEN).astype(str),
        frame[AMOUNT].map(lambda v: MISSING_TOKEN if pd.isna(v) else repr(float(v))),
        frame[COL_DESCRIPTION].fillna(MISSING_TOKEN).astype(str),
        frame[PERIOD].map(lambda v: MISSING_TOKEN if pd.isna(v) else str(int(v))),
        frame[COL_ACCOUNT].fillna(MISSING_TOKEN).astype(str),
        frame[COL_DOC_TYPE].fillna(MISSING_TOKEN).astype(str),
    ]
    keys = pd.concat(parts, axis=1).agg(_GROUP_KEY_SEPARATOR.join, axis=1)
    codes, _ = pd.factorize(keys, sort=False)
    return codes.astype(np.int64)


def vendor_groups(frame: pd.DataFrame) -> np.ndarray:
    """Return group ids from the lowercased vendor name, for unseen-vendor CV.

    Args:
        frame: Coerced frame.

    Returns:
        Integer group ids, one per distinct vendor.
    """
    vendors = frame[COL_VENDOR].fillna(MISSING_TOKEN).astype(str).str.lower()
    codes, _ = pd.factorize(vendors, sort=False)
    return codes.astype(np.int64)


def grouped_cv(seed: int, n_splits: int) -> StratifiedGroupKFold:
    """Return the primary grouped, stratified splitter.

    Args:
        seed: Shuffle seed.
        n_splits: Number of folds.

    Returns:
        A ``StratifiedGroupKFold`` with shuffling.
    """
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def random_cv(seed: int, n_splits: int) -> StratifiedKFold:
    """Return the ungrouped splitter, used only for the leakage demonstration.

    Args:
        seed: Shuffle seed.
        n_splits: Number of folds.

    Returns:
        A ``StratifiedKFold`` with shuffling.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def out_of_time_mask(frame: pd.DataFrame, cutoff: str) -> np.ndarray:
    """Return True for training rows at or before the cutoff month.

    Args:
        frame: Coerced frame with a ``period`` column.
        cutoff: ISO date; rows whose period is at or before its month train.

    Returns:
        Boolean mask; rows with missing periods are treated as training rows.
    """
    stamp = pd.Timestamp(cutoff)
    cutoff_period = period_index(stamp.year, stamp.month)
    period = frame[PERIOD].to_numpy(dtype=float)
    return np.asarray(np.isnan(period) | (period <= cutoff_period), dtype=bool)


@dataclass(frozen=True)
class HoldoutSelection:
    """Outcome of the holdout seed rule."""

    seed: int
    train_idx: np.ndarray
    holdout_idx: np.ndarray
    seeds_tried: int


def _rule_accepts(y: np.ndarray, holdout_idx: np.ndarray, rule: HoldoutRule) -> bool:
    if not rule.min_rows <= len(holdout_idx) <= rule.max_rows:
        return False
    held = y[holdout_idx]
    if any((held == c).sum() < 1 for c in rule.required_classes):
        return False
    if any((held == c).sum() > 0 for c in rule.excluded_classes):
        return False
    return all(
        abs((held == c).mean() - (y == c).mean()) <= rule.share_tolerance
        for c in rule.share_classes
    )


def select_holdout_seed(y: np.ndarray, groups: np.ndarray, rule: HoldoutRule) -> HoldoutSelection:
    """Apply the written holdout seed rule (DECISIONS.md D03).

    The rule depends only on label counts, never on model results.

    Args:
        y: Canonical labels for every raw row.
        groups: Group ids from ``build_group_key``.
        rule: Thresholds of the rule.

    Returns:
        The first accepted seed with its train and holdout indices.

    Raises:
        ConfigError: When no seed in range satisfies the rule.
    """
    placeholder = np.zeros(len(y))
    for seed in range(rule.max_seed):
        splitter = StratifiedGroupKFold(n_splits=rule.n_splits, shuffle=True, random_state=seed)
        with small_class_warning_suppressed():
            train_idx, holdout_idx = next(iter(splitter.split(placeholder, y, groups)))
        if _rule_accepts(y, holdout_idx, rule):
            return HoldoutSelection(seed, np.sort(train_idx), np.sort(holdout_idx), seed + 1)
    raise ConfigError(
        f"No seed below {rule.max_seed} satisfies the holdout rule. "
        "Review HoldoutRule in config.py.",
        code="HOLDOUT_RULE_UNSATISFIED",
    )


@dataclass
class SplitResult:
    """Persisted split: training and holdout row ids plus the manifest."""

    train_ids: np.ndarray
    holdout_ids: np.ndarray
    manifest: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LabeledFrame:
    """The full raw file: raw cells, coerced frame, canonical labels, and group ids."""

    raw: pd.DataFrame
    frame: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray


def load_labeled_raw(paths: Paths, limits: Limits) -> LabeledFrame:
    """Read the raw training file through the shared validation path.

    ``row_id`` is the 0-based position of the row below the header in the raw
    sheet, derived from the spreadsheet row number, so it stays correct even
    if blank rows are dropped during coercion.

    Args:
        paths: Project paths.
        limits: Input limits.

    Returns:
        The raw cells (resolved columns, one row per sheet row), the coerced
        frame with ``row_id``, labels, and group ids.

    Raises:
        FileReadError: When the raw file is missing or unreadable.
        SchemaError: When any raw label fails to map to a canonical class.
    """
    raw, _, _ = read_table(paths.raw_data, paths.raw_data.name, limits)
    resolved, _, _ = resolve_columns(raw, limits)
    frame, _ = coerce_values(resolved, limits)
    labels, _ = normalize_labels(frame[LABEL_COLUMN])
    if labels.isna().any():
        bad = sorted(set(frame.loc[labels.isna(), LABEL_COLUMN].astype(str)))
        raise SchemaError(
            f"Training labels failed to map to a canonical class: {', '.join(bad)}. "
            "Fix the labels in the raw file or extend labels.normalize_label.",
            code="UNMAPPED_TRAINING_LABEL",
        )
    first_data_row = int(resolved[SOURCE_ROW].iloc[0])
    frame = frame.copy()
    frame.insert(0, ROW_ID, frame[SOURCE_ROW].to_numpy() - first_data_row)
    return LabeledFrame(resolved, frame, labels.to_numpy(dtype=object), build_group_key(frame))


def make_holdout_split(settings: Settings, force: bool = False) -> SplitResult:
    """Run the seed rule on the raw file and persist the split and holdout files.

    Rerunning is safe: when a split already exists, the recomputed holdout
    must equal the persisted one (same files, same bytes). A different result
    (new sklearn version, edited rule, changed raw file) raises instead of
    silently moving rows between training and holdout.

    Args:
        settings: Project settings.
        force: Replace an existing split even when the holdout would change.

    Returns:
        The persisted split.

    Raises:
        ConfigError: When an existing split would change and ``force`` is False,
            or when no seed satisfies the rule.
        SchemaError: When a raw label does not map to a canonical class.
    """
    data = load_labeled_raw(settings.paths, settings.limits)
    selection = select_holdout_seed(data.y, data.groups, settings.holdout_rule)
    train_ids = data.frame[ROW_ID].to_numpy()[selection.train_idx]
    holdout_ids = data.frame[ROW_ID].to_numpy()[selection.holdout_idx]
    if not force:
        _refuse_changed_split(settings.paths, holdout_ids)
    splits_dir, holdout_dir = settings.paths.splits, settings.paths.holdout
    splits_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir.mkdir(parents=True, exist_ok=True)
    ids_path = splits_dir / HOLDOUT_IDS_FILE
    _write_csv(pd.DataFrame({ROW_ID: holdout_ids}), ids_path)
    _write_holdout_files(settings, data.raw, holdout_ids)
    manifest = _build_manifest(settings, data, selection, ids_path)
    _write_json(splits_dir / MANIFEST_FILE, manifest)
    return SplitResult(np.sort(train_ids), np.sort(holdout_ids), manifest)


def _refuse_changed_split(paths: Paths, holdout_ids: np.ndarray) -> None:
    ids_path = paths.splits / HOLDOUT_IDS_FILE
    if not ids_path.exists():
        return
    persisted = pd.read_csv(ids_path, encoding="utf-8")[ROW_ID].to_numpy(dtype=np.int64)
    if not np.array_equal(np.sort(persisted), np.sort(holdout_ids)):
        raise ConfigError(
            "Recomputing the split gives a different holdout than the persisted one in "
            "data/splits. Models trained on the old split would then see holdout rows. Keep "
            "the persisted split, or rerun with --force and retrain every model.",
            code="SPLIT_WOULD_CHANGE",
        )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _write_holdout_files(settings: Settings, raw: pd.DataFrame, holdout_ids: np.ndarray) -> None:
    """Write holdout features with original raw values, so uploads exercise full validation."""
    held = raw.iloc[np.sort(holdout_ids)].reset_index(drop=True)
    features = held[list(INPUT_COLUMNS)].copy()
    features[COL_INVOICE] = features[COL_INVOICE].map(_invoice_text)
    holdout_dir = settings.paths.holdout
    write_xlsx_deterministic(features, holdout_dir / f"{HOLDOUT_FEATURES_STEM}.xlsx")
    _write_csv(features, holdout_dir / f"{HOLDOUT_FEATURES_STEM}.csv")
    labels = pd.DataFrame(
        {ROW_NUMBER: np.arange(1, len(held) + 1), LABEL_COLUMN: held[LABEL_COLUMN]}
    )
    _write_csv(labels, holdout_dir / HOLDOUT_LABELS_FILE)
    write_xlsx_deterministic(
        features.assign(**{LABEL_COLUMN: held[LABEL_COLUMN]}),
        holdout_dir / HOLDOUT_WITH_LABELS_FILE,
    )


def _invoice_text(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return "" if is_missing(value) else str(value)


def _class_counts(y: np.ndarray) -> dict[str, int]:
    return {c: int((y == c).sum()) for c in CANONICAL_CLASSES}


def _build_manifest(
    settings: Settings, data: LabeledFrame, selection: HoldoutSelection, ids_path: Path
) -> dict[str, Any]:
    rule = settings.holdout_rule
    periods = data.frame[PERIOD].dropna().astype(int)
    return {
        "seed": selection.seed,
        "seeds_tried": selection.seeds_tried,
        "rule": {
            "splitter": f"StratifiedGroupKFold(n_splits={rule.n_splits}, shuffle=True)",
            "fold_used": "first",
            "rows": [rule.min_rows, rule.max_rows],
            "required_classes": list(rule.required_classes),
            "excluded_classes": list(rule.excluded_classes),
            "share_classes": list(rule.share_classes),
            "share_tolerance": rule.share_tolerance,
        },
        "rows": {
            "total": len(data.y),
            "train": len(selection.train_idx),
            "holdout": len(selection.holdout_idx),
        },
        "class_counts": {
            "total": _class_counts(data.y),
            "train": _class_counts(data.y[selection.train_idx]),
            "holdout": _class_counts(data.y[selection.holdout_idx]),
        },
        "groups": {
            "total": len(np.unique(data.groups)),
            "train": len(np.unique(data.groups[selection.train_idx])),
            "holdout": len(np.unique(data.groups[selection.holdout_idx])),
        },
        "period_range": [period_label(int(periods.min())), period_label(int(periods.max()))],
        "raw_data_sha256": sha256_file(settings.paths.raw_data),
        "holdout_row_ids_sha256": sha256_file(ids_path),
        "sklearn_version": sklearn.__version__,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_split(paths: Paths) -> SplitResult:
    """Read the persisted split, verifying its hashes. Never recomputes it.

    Args:
        paths: Project paths.

    Returns:
        Training and holdout row ids plus the manifest.

    Raises:
        ConfigError: When the split files do not exist, or when the raw file
            or the id file no longer match the hashes in the manifest.
    """
    ids_path = paths.splits / HOLDOUT_IDS_FILE
    manifest_path = paths.splits / MANIFEST_FILE
    if not ids_path.exists() or not manifest_path.exists():
        raise ConfigError(
            "The holdout split does not exist yet. Run: python -m ledgerclf split",
            code="SPLIT_MISSING",
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _verify_hash(ids_path, manifest["holdout_row_ids_sha256"])
    _verify_hash(paths.raw_data, manifest["raw_data_sha256"])
    holdout_ids = pd.read_csv(ids_path, encoding="utf-8")[ROW_ID].to_numpy(dtype=np.int64)
    total = int(manifest["rows"]["total"])
    train_ids = np.setdiff1d(np.arange(total), holdout_ids)
    return SplitResult(train_ids, np.sort(holdout_ids), manifest)


def _verify_hash(path: Path, expected: str) -> None:
    if sha256_file(path) != expected:
        raise ConfigError(
            f"{path.name} does not match the hash recorded in split_manifest.json. The raw data "
            "or the split changed after the split was made. Restore the original files from "
            "the submission, or rerun: python -m ledgerclf split --force (then retrain).",
            code="SPLIT_HASH_MISMATCH",
        )


@dataclass(frozen=True)
class TrainingData:
    """Training rows only: coerced frame, canonical labels, group ids, raw cells.

    ``raw`` holds the original cells of the same rows (resolved columns, before
    coercion), aligned with ``frame``, for before-and-after views.
    """

    frame: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    raw: pd.DataFrame


def load_training_frame(settings: Settings) -> TrainingData:
    """Return the training portion. The single entry point for modeling code.

    Holdout rows (and their labels) are dropped immediately after reading.

    Args:
        settings: Project settings.

    Returns:
        Training frame (index reset), labels, and group ids.

    Raises:
        ConfigError: When the split is missing or does not match the raw file.
    """
    split = load_split(settings.paths)
    data = load_labeled_raw(settings.paths, settings.limits)
    if len(data.y) != int(split.manifest["rows"]["total"]):
        raise ConfigError(
            "The raw file row count differs from the split manifest. Restore the original "
            "raw file, or rerun: python -m ledgerclf split --force (then retrain).",
            code="SPLIT_ROWS_MISMATCH",
        )
    keep = ~np.isin(data.frame[ROW_ID].to_numpy(), split.holdout_ids)
    frame = data.frame.loc[keep].reset_index(drop=True)
    raw = data.raw.iloc[frame[ROW_ID].to_numpy()].drop(columns=[LABEL_COLUMN])
    return TrainingData(frame, data.y[keep], data.groups[keep], raw.reset_index(drop=True))


class HoldoutAccessToken:
    """Proof that the caller is the evaluate-holdout command.

    Only ``issue_holdout_token`` creates instances; it requires the explicit
    confirmation flag that the CLI collects from the user.
    """

    __slots__ = ("reason",)
    _issuer_key = object()

    def __init__(self, issuer_key: object, reason: str) -> None:
        if issuer_key is not HoldoutAccessToken._issuer_key:
            raise HoldoutAccessError(_GUARD_MESSAGE)
        self.reason = reason


def issue_holdout_token(confirmed: bool, reason: str) -> HoldoutAccessToken:
    """Create a holdout access token. Call only from the evaluate-holdout command.

    Args:
        confirmed: The user's explicit ``--confirm`` flag.
        reason: Why the holdout is being read (written to the access log).

    Returns:
        The token.

    Raises:
        HoldoutAccessError: Without explicit confirmation.
    """
    if not confirmed:
        raise HoldoutAccessError(
            "Holdout evaluation needs explicit confirmation. Run: "
            "python -m ledgerclf evaluate-holdout --confirm",
            code="HOLDOUT_NOT_CONFIRMED",
        )
    return HoldoutAccessToken(HoldoutAccessToken._issuer_key, reason)


def read_holdout_labels(paths: Paths, token: object) -> pd.DataFrame:
    """Read the holdout labels. Requires a token from ``issue_holdout_token``.

    Args:
        paths: Project paths.
        token: The access token.

    Returns:
        Frame with ``row_number`` and the raw ``ClassificationLabel``.

    Raises:
        HoldoutAccessError: When the token is missing or forged.
    """
    if not isinstance(token, HoldoutAccessToken):
        raise HoldoutAccessError(_GUARD_MESSAGE)
    return pd.read_csv(
        paths.holdout / HOLDOUT_LABELS_FILE, dtype=object, keep_default_na=False, encoding="utf-8"
    )
