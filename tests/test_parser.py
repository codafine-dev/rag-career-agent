from __future__ import annotations

import pytest

from app.domain.exceptions import TicketParseError
from app.ingestion.parser import parse_ticket


def _valid_raw_ticket() -> dict:
    return {
        "id": "TICKET-100",
        "title": "Example ticket",
        "description": "Example description.",
        "project": "example-project",
        "technologies": ["Django", " Django ", "PostgreSQL", ""],
        "created_at": "2025-01-01",
    }


def test_parse_ticket_valid_minimal_record() -> None:
    ticket = parse_ticket(_valid_raw_ticket(), source="test")

    assert ticket.id == "TICKET-100"
    assert ticket.created_at.isoformat() == "2025-01-01"
    assert ticket.comments == []


def test_parse_ticket_normalizes_technology_list() -> None:
    ticket = parse_ticket(_valid_raw_ticket(), source="test")

    # Whitespace-only duplicate and empty string are removed, order preserved.
    assert ticket.technologies == ["Django", "PostgreSQL"]


def test_parse_ticket_missing_required_field_raises() -> None:
    raw = _valid_raw_ticket()
    del raw["title"]

    with pytest.raises(TicketParseError) as exc_info:
        parse_ticket(raw, source="TICKET-100.json")

    assert "TICKET-100.json" in str(exc_info.value)
    assert "title" in exc_info.value.reason


def test_parse_ticket_invalid_date_raises() -> None:
    raw = _valid_raw_ticket()
    raw["created_at"] = "not-a-date"

    with pytest.raises(TicketParseError):
        parse_ticket(raw, source="test")


def test_parse_ticket_parses_comments_and_optional_fields() -> None:
    raw = _valid_raw_ticket()
    raw["comments"] = [{"author": "lin", "text": "A comment.", "date": "2025-01-02"}]
    raw["resolution"] = "Fixed."
    raw["resolved_at"] = "2025-01-03"

    ticket = parse_ticket(raw, source="test")

    assert len(ticket.comments) == 1
    assert ticket.comments[0].author == "lin"
    assert ticket.resolution == "Fixed."
