from __future__ import annotations

from typing import Any

from pymongo.database import Database

from app.db.schemas import chunk_to_document, document_to_chunk
from app.retrieval.models import TicketChunk


def upsert_chunks(db: Database, chunks: list[TicketChunk]) -> None:
    for chunk in chunks:
        db.ticket_chunks.replace_one({"_id": chunk.id}, chunk_to_document(chunk), upsert=True)


def find_chunks(
    db: Database,
    *,
    technology: str | None = None,
    project: str | None = None,
) -> list[TicketChunk]:
    query: dict[str, Any] = {}
    if technology is not None:
        query["technologies"] = technology
    if project is not None:
        query["project"] = project
    return [document_to_chunk(doc) for doc in db.ticket_chunks.find(query)]
