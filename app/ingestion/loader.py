from __future__ import annotations

import json
import logging
from pathlib import Path

from app.domain.exceptions import TicketParseError
from app.domain.models import Ticket
from app.ingestion.parser import parse_ticket

logger = logging.getLogger(__name__)


def load_tickets_from_dir(directory: Path) -> list[Ticket]:
    """Load and parse every ``*.json`` ticket file in ``directory``.

    Files that fail to parse are logged (by filename only, never ticket
    content) and skipped rather than aborting the whole batch.

    :param directory: Directory containing one JSON ticket record per file.
    :return: Successfully parsed tickets.
    """
    tickets: list[Ticket] = []
    for path in sorted(directory.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Skipping %s: invalid JSON (%s)", path.name, exc.msg)
            continue

        try:
            tickets.append(parse_ticket(raw, source=path.name))
        except TicketParseError as exc:
            logger.warning("Skipping %s: %s", path.name, exc.reason)
            continue

    logger.info("Loaded %d ticket(s) from %s", len(tickets), directory)
    return tickets
