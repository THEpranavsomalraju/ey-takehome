"""The what_if tool must score a changed cell exactly as an upload with that cell would.

docs/05 section 1 principle 5: every source passes through ``schema.coerce_values``;
V04 (extra spaces give the same prediction), V15 (``doc1`` canonicalizes),
section 2.5 (``N/A`` is a missing marker). ``agent_tools.what_if`` writes the
raw text straight into the coerced frame, so the agent quotes probabilities
that differ from what re-uploading the edited row returns.
"""

from __future__ import annotations

import pytest

from ledgerclf.agent_tools import ToolContext, run_tool
from ledgerclf.labels import CANONICAL_CLASSES
from tests.adversarial._support import HEADER, ROW_A, csv_text, with_value


def _upload_probabilities(service, row: list[str]) -> dict[str, float]:
    result = service.predict_file(csv_text([row]).encode("utf-8"), "edited.csv")
    first = result.predictions.iloc[0]
    return {c: round(float(first[f"p_{c}"]), 4) for c in CANONICAL_CLASSES}


@pytest.mark.parametrize(
    ("column", "value"),
    [
        pytest.param("Col7", "doc 1", id="doc_type_with_space"),
        pytest.param("Col6", " Word563   Word575 ", id="account_with_extra_spaces"),
        pytest.param("Col4", "N/A", id="description_missing_marker"),
    ],
)
def test_what_if_matches_upload_of_the_edited_row(service, column, value):
    result = service.predict_file(csv_text([ROW_A]).encode("utf-8"), "one_row.csv")
    context = ToolContext(service, result, 1)
    changed = run_tool(context, "what_if", {"column": column, "value": value})
    edited = with_value(ROW_A, HEADER.index(column), value)
    assert changed["new_probabilities"] == _upload_probabilities(service, edited)
