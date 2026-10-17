"""POST /api/predict with a file part that has no file name must answer in plain English.

A script (``curl -F "file=@ledger.csv;filename="``) or any client that sends
the ``file`` part with content but an empty file name gets ``body.file: Value
error, Expected UploadFile, received: <class 'str'>``: a Python type name
instead of the plain-English message with the fix that the server docstring
("Every error reaches the browser as JSON with a plain-English message") and
CLAUDE.md section 9 require. (A browser form with no file chosen sends empty
content and already gets the clear FILE_EMPTY message.)
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
