from __future__ import annotations

import datetime as dt

from pymongo.database import Database

from app.db.chunk_repository import find_chunks
from app.db.usage_repository import get_today_spend_usd
from app.retrieval.indexer import embed_and_store_chunks
from app.retrieval.models import TicketChunk
from tests.fakes import FakeEmbeddingsProvider


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


def test_embed_and_store_chunks_persists_vectors(db: Database) -> None:
    chunks = [_make_chunk()]
    provider = FakeEmbeddingsProvider(vectors=[[0.1, 0.2, 0.3]])

    embed_and_store_chunks(db, chunks, provider)

    stored = find_chunks(db)
    assert len(stored) == 1
    assert stored[0].embedding == [0.1, 0.2, 0.3]


def test_embed_and_store_chunks_records_usage(db: Database) -> None:
    chunks = [_make_chunk()]
    provider = FakeEmbeddingsProvider(vectors=[[0.1, 0.2]], tokens_used=50, cost_usd=0.00001)

    embed_and_store_chunks(db, chunks, provider)

    assert get_today_spend_usd(db) == 0.00001


def test_embed_and_store_chunks_handles_empty_list(db: Database) -> None:
    embed_and_store_chunks(db, [], FakeEmbeddingsProvider(vectors=[]))

    assert find_chunks(db) == []
    assert get_today_spend_usd(db) == 0.0
