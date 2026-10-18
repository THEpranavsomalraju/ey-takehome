"""The offline explanation names the input that actually drove the prediction.

With an extreme amount (1e50) and unseen text, the amount carries most of the
margin. The summary's "main reason" must name the amount, not a weak input that
happens to have training history (such as the document flag).
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
