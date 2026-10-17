"""The offline explanation must not name a negligible input as the main reason.

For an extreme amount (1e50) with every text value unseen, the amount group
carries the margin (about +5.3 log-odds against a total of +2.7, with the base
rate pulling -4.9 and the next supporting group under +1.2). Yet the offline
Summary of POST /api/explain (use_llm false) says the prediction holds "mainly
because 517 of 4166 past invoices with the same document flag (Col7) were
Category_2", a group that contributes about +0.02. ``_main_reason`` skips every
supporting group without training evidence (amount and period never have any)
and cites the first one that has some, however small. DECISIONS.md
(explanations) requires the Summary to state the model facts; a "main reason"
that the payload contradicts is not one.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from ledgerclf.explain import GROUP_LABELS
from ledgerclf.server import create_app
from tests.adversarial._support import csv_text

EXTREME_AMOUNT_ROW = ["Zq1 Zq2", "", "1e50", "Zq3", "2018-08-31", "Zq4", "NoDoc"]


def test_offline_summary_cites_the_strongest_supporting_input(service):
    api = TestClient(create_app(service))
    body = csv_text([EXTREME_AMOUNT_ROW]).encode("utf-8")
    upload = api.post("/api/predict", files={"file": ("amount.csv", body, "text/csv")}).json()

    response = api.post(
        "/api/explain",
        json={"upload_id": upload["upload_id"], "row_number": 1, "use_llm": False},
    )
    assert response.status_code == 200, response.text
    contributions = response.json()["payload"]["contributions"]
    supporting = [g for g in contributions["by_group"] if g["contribution"] > 0]
    strongest = max(supporting, key=lambda g: g["contribution"])
    summary = response.json()["answer"]["text"].split("\n")[0]
    cited = [group for group, label in GROUP_LABELS.items() if label in summary]

    assert strongest["group"] == "Col3"
    assert cited in ([], ["Col3"]), f"Summary cites {cited}, strongest is Col3: {summary}"
