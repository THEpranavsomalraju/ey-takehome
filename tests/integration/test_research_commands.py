"""Tuning, selection, training, golden comparison, holdout evaluation, and CLI commands.

Heavy commands run on a subsample with small budgets in a temporary project
copy. The holdout evaluation test replaces the holdout labels with synthetic
ones first, so the real holdout labels are never read (only evaluate-holdout may).
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from ledgerclf import cli, selection, tuning
from ledgerclf.config import Settings
from ledgerclf.golden import compare_with_golden
from ledgerclf.splitting import HOLDOUT_LABELS_FILE, TrainingData, issue_holdout_token
from tests.conftest import project_copy

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def small_training(training):
    rare = np.flatnonzero(training.y != "Category_1")
    rows = np.unique(np.concatenate([rare, np.arange(0, len(training.y), 6)]))
    return TrainingData(
        training.frame.iloc[rows].reset_index(drop=True),
        training.y[rows],
        training.groups[rows],
        training.raw.iloc[rows].reset_index(drop=True),
    )


@pytest.fixture(scope="module")
def small_project(settings, tmp_path_factory):
    copy = project_copy(settings, tmp_path_factory.mktemp("project"), with_artifacts=False)
    tune = dataclasses.replace(
        settings.tuning,
        quick_linear_C=(3.0, 10.0),
        quick_tree_trials=2,
        quick_nested_tree_trials=1,
        tree_space={**settings.tuning.tree_space, "n_estimators": (30,), "num_leaves": (7,)},
    )
    bench = dataclasses.replace(settings.benchmark, cv_jobs=1)
    tree = dataclasses.replace(settings.tree, n_estimators=30)
    return Settings(paths=copy.paths, tuning=tune, benchmark=bench, tree=tree, seeds=(0,))


def test_tune_select_train_chain_in_a_project_copy(small_project, small_training, monkeypatch):
    for module in (tuning, selection):
        monkeypatch.setattr(module, "load_training_frame", lambda settings: small_training)
    written = tuning.run_tuning(small_project, quick=True)
    assert written == list(tuning.TUNING_REPORTS)
    best = json.loads(
        (small_project.paths.reports / "tuning/best_params.json").read_text(encoding="utf-8")
    )
    assert best["quick_mode"] is True
    nested = json.loads(
        (small_project.paths.reports / "tuning/nested_cv.json").read_text(encoding="utf-8")
    )
    assert len(nested["outer_folds"]) == small_project.cv_folds

    written = selection.run_selection(small_project)
    assert written == list(selection.SELECTION_REPORTS)
    champion = json.loads(
        (small_project.paths.artifacts / "champion.json").read_text(encoding="utf-8")
    )
    assert champion["champion"]["name"] in {
        selection.candidate_name(f, p) for f, p in selection.CANDIDATES
    }
    assert champion["fallback"]["family"] == "logistic"
    assert champion["champion"]["temperature"] > 0
    decision = json.loads(
        (small_project.paths.reports / "selection/champion_decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert decision["trail"] and decision["champion"] == champion["champion"]["name"]


def test_load_tuned_explains_missing_tuning(tmp_path):
    from ledgerclf.config import Paths
    from ledgerclf.exceptions import ConfigError

    with pytest.raises(ConfigError) as caught:
        selection.load_tuned(Settings(paths=Paths.from_root(tmp_path)))
    assert "python -m ledgerclf tune" in caught.value.message


def test_golden_comparison_passes_and_detects_drift(settings, tmp_path):
    report = compare_with_golden(
        settings.paths.artifacts / "golden_holdout_predictions.csv", 1.0, 1e-6
    )
    assert report.passed, report.lines()
    golden = pd.read_csv(settings.paths.artifacts / "golden_holdout_predictions.csv")
    golden.loc[0, "p_Category_1"] += 0.01
    drifted = tmp_path / "golden.csv"
    golden.to_csv(drifted, index=False)
    assert not compare_with_golden(drifted, 1.0, 1e-6).passed


def test_holdout_evaluation_on_synthetic_labels(settings, tmp_path):
    from ledgerclf.exceptions import HoldoutAccessError
    from ledgerclf.holdout_eval import run_holdout_evaluation

    copy = project_copy(settings, tmp_path / "project")
    synthetic = pd.DataFrame(
        {"row_number": np.arange(1, 591), "ClassificationLabel": ["Category_1"] * 590}
    )
    synthetic.to_csv(copy.paths.holdout / HOLDOUT_LABELS_FILE, index=False)
    token = issue_holdout_token(confirmed=True, reason="synthetic labels test")
    summary = run_holdout_evaluation(copy, token, rerun=False)
    assert summary["rows"] == 590 and "accuracy" in summary["champion"]
    assert (copy.paths.reports / "holdout/access_log.csv").exists()
    with pytest.raises(HoldoutAccessError):
        run_holdout_evaluation(copy, token, rerun=False)
    run_holdout_evaluation(copy, token, rerun=True)
    log = pd.read_csv(copy.paths.reports / "holdout/access_log.csv")
    assert len(log) == 2


def test_cli_commands_end_to_end(settings, capsys, monkeypatch, tmp_path):
    holdout = str(settings.paths.holdout / "holdout_features.csv")
    out_csv, out_json = tmp_path / "pred.csv", tmp_path / "explain.json"
    assert cli.main(["predict", holdout, "--out", str(out_csv)]) == 0
    assert "Rows predicted: 590" in capsys.readouterr().out
    assert len(pd.read_csv(out_csv, encoding="utf-8-sig")) == 590
    assert cli.main(["explain", holdout, "--row", "3", "--out", str(out_json)]) == 0
    assert json.loads(out_json.read_text(encoding="utf-8"))["row_number"] == 3
    assert cli.main(["explain", holdout, "--row", "999"]) == cli.EXIT_USER_ERROR
    assert cli.main(["predict", str(tmp_path / "missing.csv")]) == cli.EXIT_USER_ERROR
    assert cli.main(["evaluate-holdout"]) == cli.EXIT_USER_ERROR
    assert "--confirm" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("command", "target"),
    [
        (["eda"], ("ledgerclf.eda", "run_eda")),
        (["benchmark", "--quick"], ("ledgerclf.benchmark", "run_benchmark")),
        (["tune", "--quick"], ("ledgerclf.tuning", "run_tuning")),
        (["select"], ("ledgerclf.selection", "run_selection")),
    ],
)
def test_cli_research_commands_dispatch(command, target, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        f"{target[0]}.{target[1]}", lambda settings, **kwargs: calls.append(kwargs) or ["x"]
    )
    monkeypatch.setattr(
        "ledgerclf.preprocessing_reports.run_preprocessing_reports", lambda settings: ["y"]
    )
    assert cli.main(command) == 0
    assert calls and capsys.readouterr().out.strip()


def test_cli_train_dispatch(monkeypatch, capsys):
    summary = {"champion": "c", "fallback": "f", "training_rows": 1}
    monkeypatch.setattr("ledgerclf.training.run_training", lambda settings: summary)
    assert cli.main(["train"]) == 0
    assert "Trained champion c" in capsys.readouterr().out


def test_preprocessing_reports_write_every_file(settings, tmp_path):
    from ledgerclf.preprocessing_reports import PREPROCESSING_CATALOG, run_preprocessing_reports

    custom = Settings(paths=dataclasses.replace(settings.paths, reports=tmp_path))
    assert run_preprocessing_reports(custom) == list(PREPROCESSING_CATALOG)
    space = pd.read_csv(tmp_path / "preprocessing/feature_space_summary.csv")
    assert space["features"].sum() > 1000
    samples = pd.read_csv(tmp_path / "preprocessing/before_after_samples.csv")
    assert len(samples) == 5 * 7


def test_service_reports_and_health(service):
    info = service.model_info()
    assert info["model_used"] == "champion" and info["training_rows"] == 5309
    assert any(entry["file"] == "eda/class_distribution.csv" for entry in service.list_reports())
    assert isinstance(service.load_report("eda/class_distribution.csv"), pd.DataFrame)
    assert service.health()["exit_code"] == 0
