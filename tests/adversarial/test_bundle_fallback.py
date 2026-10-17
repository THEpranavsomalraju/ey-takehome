"""A missing champion file falls back to the logistic bundle (docs/02 sections 4 and 6).

``load_best_available``: "blend, else fallback, logs reason". Today a missing
champion raises ArtifactError even though model_fallback.joblib loads fine, and
``PredictionService.load(auto_train=True)`` neither falls back nor retrains.
"""

from __future__ import annotations

from ledgerclf.artifacts import CHAMPION_BUNDLE, load_best_available
from tests.conftest import project_copy


def test_missing_champion_file_loads_fallback_with_reason(tmp_path, settings):
    copy = project_copy(settings, tmp_path / "project")
    (copy.paths.artifacts / CHAMPION_BUNDLE).unlink()
    bundle, reason = load_best_available(copy.paths)
    assert bundle.is_fallback
    assert reason
