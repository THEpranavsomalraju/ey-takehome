"""AmountFlags follows docs/04 section 4 (``is_round = x % 10 == 0``) at extreme magnitudes.

The cents cast to int64 overflows above about 9.2e16, which emits a numpy
RuntimeWarning to the user's console and yields a platform-dependent flag.
Amounts below half a cent count as round because they round to 0 cents.
"""

from __future__ import annotations

import warnings

import numpy as np

from ledgerclf.features import AmountFlags
from tests.adversarial._support import ROW_A, csv_text, with_value


def test_huge_amount_predicts_without_numpy_cast_warning(service):
    data = csv_text([with_value(ROW_A, 2, "1e17")]).encode("utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = service.predict_file(data, "huge_amount.csv")
    assert len(result.predictions) == 1


def test_round_flag_matches_spec_for_huge_and_sub_cent_amounts():
    amounts = np.array([[1e17], [-1e18], [0.001], [0.004]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        is_round = AmountFlags().fit(amounts).transform(amounts)[:, 1]
    np.testing.assert_array_equal(is_round, [1.0, 1.0, 0.0, 0.0])
