from __future__ import annotations

import datetime as dt

from pymongo.database import Database

from app.db.ticket_repository import get_ticket, list_tickets, upsert_ticket
from app.domain.models import Ticket


def _make_ticket(**overrides: object) -> Ticket:
    defaults: dict = {
        "id": "TICKET-1",
        "title": "Title",
        "description": "Description",
        "project": "core-api",
        "technologies": ["Django", "PostgreSQL"],
        "created_at": dt.date(2025, 1, 1),
    }
    defaults.update(overrides)
    return Ticket.model_validate(defaults)


def test_upsert_and_get_ticket_roundtrip(db: Database) -> None:
    ticket = _make_ticket()

    upsert_ticket(db, ticket)
    fetched = get_ticket(db, "TICKET-1")

    assert fetched == ticket


def test_upsert_ticket_is_idempotent(db: Database) -> None:
    upsert_ticket(db, _make_ticket(title="First version"))
    upsert_ticket(db, _make_ticket(title="Second version"))

    assert db.tickets.count_documents({}) == 1
    assert get_ticket(db, "TICKET-1").title == "Second version"


def test_get_ticket_missing_returns_none(db: Database) -> None:
    assert get_ticket(db, "does-not-exist") is None


def test_list_tickets_filters_by_technology_and_project(db: Database) -> None:
    upsert_ticket(db, _make_ticket(id="T-1", project="core-api", technologies=["AWS"]))
    upsert_ticket(db, _make_ticket(id="T-2", project="billing-service", technologies=["AWS"]))
    upsert_ticket(db, _make_ticket(id="T-3", project="core-api", technologies=["Django"]))

    assert {t.id for t in list_tickets(db, technology="AWS")} == {"T-1", "T-2"}
    assert {t.id for t in list_tickets(db, project="core-api")} == {"T-1", "T-3"}
    assert {t.id for t in list_tickets(db, technology="AWS", project="core-api")} == {"T-1"}
