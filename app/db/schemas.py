from __future__ import annotations

import datetime as dt
from typing import Any

from app.domain.models import Ticket

_DATE_FIELDS = ("created_at", "updated_at", "resolved_at")


def ticket_to_document(ticket: Ticket) -> dict[str, Any]:
    """Convert a domain :class:`Ticket` into a MongoDB document.

    BSON has no bare "date" type, so dates are stored as UTC midnight
    datetimes and converted back on read.
    """
    doc = ticket.model_dump()
    doc["_id"] = doc.pop("id")
    for field in _DATE_FIELDS:
        if doc.get(field) is not None:
            doc[field] = dt.datetime.combine(doc[field], dt.time.min)
    for comment in doc.get("comments", []):
        if comment.get("date") is not None:
            comment["date"] = dt.datetime.combine(comment["date"], dt.time.min)
    return doc


def document_to_ticket(doc: dict[str, Any]) -> Ticket:
    data = dict(doc)
    data["id"] = data.pop("_id")
    for field in _DATE_FIELDS:
        value = data.get(field)
        if isinstance(value, dt.datetime):
            data[field] = value.date()
    for comment in data.get("comments", []):
        value = comment.get("date")
        if isinstance(value, dt.datetime):
            comment["date"] = value.date()
    return Ticket.model_validate(data)
