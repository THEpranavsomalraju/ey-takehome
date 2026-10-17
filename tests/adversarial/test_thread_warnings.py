"""Concurrent predictions must not leak suppressed library warnings (docs/05 E10, section 5.6).

``warnings.catch_warnings`` in ledgerclf/models.py swaps the process-wide filter
list, so it is not thread-safe: under concurrent calls the scikit-learn
"X does not have valid feature names" warning escapes to the console.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor

from tests.adversarial._support import ROW_A, ROW_B, csv_text

TRIALS = 10
CALLS = 32
THREADS = 8


def test_concurrent_predictions_leak_no_feature_name_warnings(service):
    data = csv_text([ROW_A, ROW_B] * 50).encode("utf-8")
    leaked = 0
    for _ in range(TRIALS):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with ThreadPoolExecutor(THREADS) as pool:
                list(pool.map(lambda _: service.predict_file(data, "a.csv"), range(CALLS)))
        leaked += sum("valid feature names" in str(w.message) for w in caught)
    assert leaked == 0
