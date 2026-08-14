from __future__ import annotations

from pymongo.database import Database

from app.db.chunk_repository import upsert_chunks
from app.db.usage_repository import record_usage
from app.llm.embeddings import EmbeddingsProvider
from app.retrieval.models import TicketChunk


def embed_and_store_chunks(
    db: Database,
    chunks: list[TicketChunk],
    provider: EmbeddingsProvider,
) -> None:
    """Embed chunks, persist them, and record the API cost incurred."""
    if not chunks:
        return

    result = provider.embed([chunk.text for chunk in chunks])
    embedded = [
        chunk.model_copy(update={"embedding": vector})
        for chunk, vector in zip(chunks, result.vectors, strict=True)
    ]
    upsert_chunks(db, embedded)
    record_usage(db, tokens=result.tokens_used, cost_usd=result.cost_usd)
