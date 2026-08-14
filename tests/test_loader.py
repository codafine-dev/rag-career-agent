from __future__ import annotations

import json
from pathlib import Path

from app.ingestion.loader import load_tickets_from_dir

SAMPLE_TICKETS_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_tickets"


def test_load_tickets_from_sample_dir() -> None:
    tickets = load_tickets_from_dir(SAMPLE_TICKETS_DIR)

    assert len(tickets) == 5
    assert {t.id for t in tickets} == {
        "TICKET-001",
        "TICKET-002",
        "TICKET-003",
        "TICKET-004",
        "TICKET-005",
    }


def test_load_tickets_skips_invalid_records(tmp_path: Path, caplog) -> None:
    valid = {
        "id": "TICKET-OK",
        "title": "Valid",
        "description": "Valid ticket.",
        "project": "proj",
        "created_at": "2025-01-01",
    }
    invalid_missing_field = {"id": "TICKET-BAD", "description": "Missing title."}

    (tmp_path / "ok.json").write_text(json.dumps(valid))
    (tmp_path / "bad.json").write_text(json.dumps(invalid_missing_field))
    (tmp_path / "not_json.json").write_text("{not valid json")

    tickets = load_tickets_from_dir(tmp_path)

    assert [t.id for t in tickets] == ["TICKET-OK"]


def test_load_tickets_never_logs_ticket_body(tmp_path: Path, caplog) -> None:
    invalid_missing_field = {"id": "TICKET-SECRET", "description": "SENSITIVE_PAYLOAD_TEXT"}
    (tmp_path / "bad.json").write_text(json.dumps(invalid_missing_field))

    with caplog.at_level("WARNING"):
        load_tickets_from_dir(tmp_path)

    assert "SENSITIVE_PAYLOAD_TEXT" not in caplog.text
