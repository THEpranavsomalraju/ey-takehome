"""POST /api/predict answers in plain English when the file part has no file name.

A client that sends file content with an empty file name (for example
``curl -F "file=@ledger.csv;filename="``) must get a message saying what to
upload, not a Python validation error.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from ledgerclf.server import create_app
from tests.adversarial._support import ROW_A, csv_text


def test_unnamed_file_part_gets_a_plain_message(service):
    api = TestClient(create_app(service))
    content = csv_text([ROW_A]).encode("utf-8")

    response = api.post("/api/predict", files={"file": ("", content, "text/csv")})

    body = response.json()
    assert set(body) >= {"code", "message"}
    assert "<class" not in body["message"] and "UploadFile" not in body["message"], body
