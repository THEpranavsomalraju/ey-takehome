"""Benchmark stages on a subsample with small trees: every stage runs and writes its report."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from ledgerclf import benchmark
from ledgerclf.config import Settings
from ledgerclf.splitting import TrainingData

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def small_settings(settings, tmp_path_factory):
    tree = dataclasses.replace(settings.tree, n_estimators=30, learning_rate=0.1)
    bench = dataclasses.replace(
        settings.benchmark,
        leaf20_tree=dataclasses.replace(settings.benchmark.leaf20_tree, n_estimators=30),
        quick_dial_powers=(0.0, 1.0),
        quick_leaf_sizes=(2, 20),
        cv_jobs=1,
    )
    paths = dataclasses.replace(settings.paths, reports=tmp_path_factory.mktemp("reports"))
    return Settings(paths=paths, tree=tree, benchmark=bench)


@pytest.fixture(scope="module")
def small_training(training):
    rows = np.sort(
        np.concatenate(
            [np.flatnonzero(training.y != "Category_1"), np.arange(0, len(training.y), 5)]
        )
    )
    rows = np.unique(rows)
    return TrainingData(
        training.frame.iloc[rows].reset_index(drop=True),
        training.y[rows],
        training.groups[rows],
        training.raw.iloc[rows].reset_index(drop=True),
    )


def test_run_benchmark_writes_every_report(small_settings, small_training, monkeypatch):
    monkeypatch.setattr(benchmark, "load_training_frame", lambda settings: small_training)
    written = benchmark.run_benchmark(small_settings, quick=True)
    assert written == list(benchmark.BENCHMARK_REPORTS)
    for file in written:
        assert (small_settings.paths.reports / file).exists()
    board = pd.read_csv(small_settings.paths.reports / "benchmark/leaderboard.csv")
    assert set(board["model"]) == set(benchmark.model_zoo(small_settings))
    assert board["accuracy_mean"].is_monotonic_decreasing
    schemes = pd.read_csv(small_settings.paths.reports / "benchmark/validation_schemes.csv")
    assert {"grouped CV (primary)", "random CV (leakage demo)", "unseen-vendor CV"} <= set(
        schemes["scheme"]
    )
    ablation = pd.read_csv(small_settings.paths.reports / "benchmark/ablation.csv")
    assert ablation.loc[ablation["variant"] == "full feature set", "delta_accuracy"].item() == 0.0


def test_full_plan_uses_every_seed(settings):
    full = benchmark.plan_for(settings, quick=False)
    quick = benchmark.plan_for(settings, quick=True)
    assert full.seeds == settings.seeds and len(quick.seeds) == 1
    assert full.leaf_sizes == settings.benchmark.leaf_sizes


def test_catboost_research_row_reads_the_research_file(tmp_path):
    research = tmp_path / "catboost.json"
    research.write_text(
        '{"accuracy": 0.92, "macro_f1_5": 0.63, "fit_seconds": 193.0}', encoding="utf-8"
    )
    row = benchmark._catboost_row(research)
    assert row["model"].tolist() == ["catboost"]
    assert "research only" in row["note"].item()
