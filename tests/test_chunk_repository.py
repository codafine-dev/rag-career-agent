from __future__ import annotations

import datetime as dt

from pymongo.database import Database

from app.db.chunk_repository import find_chunks, upsert_chunks
from app.retrieval.models import TicketChunk


def _make_chunk(**overrides: object) -> TicketChunk:
    defaults: dict = {
        "id": "TICKET-1::summary::0",
        "ticket_id": "TICKET-1",
        "source_field": "summary",
        "chunk_index": 0,
        "text": "Some text.",
        "project": "core-api",
        "technologies": ["AWS"],
        "created_at": dt.date(2025, 1, 1),
    }
    defaults.update(overrides)
    return TicketChunk.model_validate(defaults)


def test_upsert_and_find_chunks_roundtrip(db: Database) -> None:
    chunk = _make_chunk(embedding=[0.1, 0.2, 0.3])

    upsert_chunks(db, [chunk])
    found = find_chunks(db)

    assert found == [chunk]


def test_upsert_chunks_is_idempotent(db: Database) -> None:
    upsert_chunks(db, [_make_chunk(text="v1")])
    upsert_chunks(db, [_make_chunk(text="v2")])

    assert db.ticket_chunks.count_documents({}) == 1
    assert find_chunks(db)[0].text == "v2"


def test_find_chunks_filters_by_technology_and_project(db: Database) -> None:
    upsert_chunks(
        db,
        [
            _make_chunk(id="c1", project="core-api", technologies=["AWS"]),
            _make_chunk(id="c2", project="billing-service", technologies=["AWS"]),
            _make_chunk(id="c3", project="core-api", technologies=["Django"]),
        ],
    )

    assert {c.id for c in find_chunks(db, technology="AWS")} == {"c1", "c2"}
    assert {c.id for c in find_chunks(db, project="core-api")} == {"c1", "c3"}
