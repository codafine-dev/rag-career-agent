from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.domain.exceptions import TicketParseError
from app.domain.models import Ticket


def _normalize_string_list(values: Any) -> list[str]:
    """Strip whitespace, drop empties, and dedupe while preserving order."""
    if not values:
        return []
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            normalized.append(cleaned)
    return normalized


def parse_ticket(raw: dict[str, Any], source: str) -> Ticket:
    """Normalize a raw ticket record into a validated :class:`Ticket`.

    :param raw: Raw ticket data as loaded from a source file.
    :param source: Identifier of the origin (e.g. file path), used in error messages.
    :raise TicketParseError: If the record is missing required fields or fails validation.
    """
    normalized = dict(raw)
    for field in ("technologies", "labels", "links", "related_tickets"):
        if field in normalized:
            normalized[field] = _normalize_string_list(normalized[field])

    try:
        return Ticket.model_validate(normalized)
    except ValidationError as exc:
        reasons = "; ".join(
            f"{'.'.join(str(loc) for loc in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise TicketParseError(source, reasons) from exc
