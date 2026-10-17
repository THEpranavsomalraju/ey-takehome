"""A missing champion file falls back to the logistic bundle.

``load_best_available``: "blend, else fallback, logs reason". A missing champion
must not raise ArtifactError while model_fallback.joblib loads fine, and
``PredictionService.load(auto_train=True)`` must fall back or retrain.
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
