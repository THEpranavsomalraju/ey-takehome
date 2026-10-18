"""POST /api/whatif reads an edited date in the upload's own day and month order.

The upload decides the date order for the whole Col5 column (one value such as
``31/08/2018`` makes it day-first). An edited date such as ``04/02/2018`` must
then mean 4 February, exactly as if the edited file were uploaded again.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from ledgerclf.labels import CANONICAL_CLASSES
from ledgerclf.server import create_app
from tests.adversarial._support import csv_text

DAY_FIRST_ROW = [
    "Word16 Word17",
    "10719",
    "1200",
    "Word61",
    "31/08/2018",
    "Word563 Word575",
    "NoDoc",
]
AMBIGUOUS_ROW = ["Word19 Word20", "65975", "75", "Word562", "05/03/2018", "Word563", "Doc1"]
EDITED_DATE = "04/02/2018"  # 4 February 2018 in a day-first column


def _upload(api: TestClient, rows: list[list[str]]) -> dict:
    body = csv_text(rows).encode("utf-8")
    response = api.post("/api/predict", files={"file": ("dates.csv", body, "text/csv")})
    assert response.status_code == 200, response.text
    return response.json()


def test_what_if_date_edit_follows_the_upload_day_first_order(service):
    api = TestClient(create_app(service))
    upload = _upload(api, [DAY_FIRST_ROW, AMBIGUOUS_ROW])
    assert upload["page"]["rows"][1]["Col5"] == "2018-03"  # the upload reads day-first

    changed = api.post(
        "/api/whatif",
        json={
            "upload_id": upload["upload_id"],
            "row_number": 2,
            "column": "Col5",
            "value": EDITED_DATE,
        },
    )
    assert changed.status_code == 200, changed.text

    edited = _upload(api, [DAY_FIRST_ROW, [*AMBIGUOUS_ROW[:4], EDITED_DATE, *AMBIGUOUS_ROW[5:]]])
    reuploaded = edited["page"]["rows"][1]
    assert reuploaded["Col5"] == "2018-02"
    expected = {c: round(float(reuploaded[f"p_{c}"]), 4) for c in CANONICAL_CLASSES}
    assert changed.json()["new_probabilities"] == expected
