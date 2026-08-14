from __future__ import annotations

import datetime as dt

from pymongo.database import Database

from app.db.chunk_repository import upsert_chunks
from app.retrieval.models import TicketChunk
from app.retrieval.retriever import search_chunks


def _make_chunk(**overrides: object) -> TicketChunk:
    defaults: dict = {
        "id": "chunk",
        "ticket_id": "ticket",
        "source_field": "summary",
        "chunk_index": 0,
        "text": "text",
        "project": "core-api",
        "technologies": ["AWS"],
        "created_at": dt.date(2025, 1, 1),
        "embedding": [1.0, 0.0],
    }
    defaults.update(overrides)
    return TicketChunk.model_validate(defaults)


def test_search_chunks_orders_by_similarity_descending(db: Database) -> None:
    upsert_chunks(
        db,
        [
            _make_chunk(id="c-low", ticket_id="t1", embedding=[0.5, 0.5]),
            _make_chunk(id="c-high", ticket_id="t2", embedding=[1.0, 0.0]),
            _make_chunk(id="c-mid", ticket_id="t3", embedding=[0.9, 0.1]),
        ],
    )

    results = search_chunks(db, query_vector=[1.0, 0.0], min_score=0.0)

    assert [r.chunk.id for r in results] == ["c-high", "c-mid", "c-low"]
    assert results[0].score > results[1].score > results[2].score


def test_search_chunks_drops_results_below_min_score(db: Database) -> None:
    upsert_chunks(
        db,
        [
            _make_chunk(id="c-match", ticket_id="t1", embedding=[1.0, 0.0]),
            _make_chunk(id="c-orthogonal", ticket_id="t2", embedding=[0.0, 1.0]),
        ],
    )

    results = search_chunks(db, query_vector=[1.0, 0.0], min_score=0.15)

    assert [r.chunk.id for r in results] == ["c-match"]


def test_search_chunks_respects_top_k(db: Database) -> None:
    upsert_chunks(
        db,
        [_make_chunk(id=f"c{i}", ticket_id=f"t{i}", embedding=[1.0, 0.0]) for i in range(5)],
    )

    results = search_chunks(db, query_vector=[1.0, 0.0], min_score=0.0, top_k=2)

    assert len(results) == 2


def test_search_chunks_caps_results_per_ticket(db: Database) -> None:
    upsert_chunks(
        db,
        [
            _make_chunk(id="t1-a", ticket_id="t1", embedding=[1.0, 0.0]),
            _make_chunk(id="t1-b", ticket_id="t1", embedding=[1.0, 0.0]),
            _make_chunk(id="t1-c", ticket_id="t1", embedding=[1.0, 0.0]),
            _make_chunk(id="t2-a", ticket_id="t2", embedding=[0.99, 0.01]),
        ],
    )

    results = search_chunks(db, query_vector=[1.0, 0.0], min_score=0.0, max_per_ticket=2, top_k=10)

    assert sum(1 for r in results if r.chunk.ticket_id == "t1") == 2
    assert any(r.chunk.ticket_id == "t2" for r in results)


def test_search_chunks_applies_metadata_filters(db: Database) -> None:
    upsert_chunks(
        db,
        [
            _make_chunk(id="c-aws", ticket_id="t1", technologies=["AWS"], embedding=[1.0, 0.0]),
            _make_chunk(id="c-django", ticket_id="t2", technologies=["Django"], embedding=[1.0, 0.0]),
        ],
    )

    results = search_chunks(db, query_vector=[1.0, 0.0], min_score=0.0, technology="AWS")

    assert [r.chunk.id for r in results] == ["c-aws"]


def test_search_chunks_ignores_chunks_without_embedding(db: Database) -> None:
    upsert_chunks(db, [_make_chunk(id="c-none", ticket_id="t1", embedding=None)])

    results = search_chunks(db, query_vector=[1.0, 0.0], min_score=0.0)

    assert results == []
