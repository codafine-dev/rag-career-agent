from __future__ import annotations

from typing import Any

from pymongo.database import Database

from app.db.schemas import document_to_ticket, ticket_to_document
from app.domain.models import Ticket


def upsert_ticket(db: Database, ticket: Ticket) -> None:
    db.tickets.replace_one({"_id": ticket.id}, ticket_to_document(ticket), upsert=True)


def get_ticket(db: Database, ticket_id: str) -> Ticket | None:
    doc = db.tickets.find_one({"_id": ticket_id})
    return document_to_ticket(doc) if doc else None


def list_tickets(
    db: Database,
    *,
    technology: str | None = None,
    project: str | None = None,
) -> list[Ticket]:
    """List tickets, optionally filtered by exact technology or project match."""
    query: dict[str, Any] = {}
    if technology is not None:
        query["technologies"] = technology
    if project is not None:
        query["project"] = project
    return [document_to_ticket(doc) for doc in db.tickets.find(query)]
