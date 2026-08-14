from __future__ import annotations

from pymongo.database import Database

from app.db.chunk_repository import find_chunks
from app.retrieval.models import ScoredChunk
from app.retrieval.similarity import cosine_similarity

_DEFAULT_TOP_K = 5
_DEFAULT_MIN_SCORE = 0.15
_DEFAULT_MAX_PER_TICKET = 2


def search_chunks(
    db: Database,
    *,
    query_vector: list[float],
    top_k: int = _DEFAULT_TOP_K,
    min_score: float = _DEFAULT_MIN_SCORE,
    technology: str | None = None,
    project: str | None = None,
    max_per_ticket: int = _DEFAULT_MAX_PER_TICKET,
) -> list[ScoredChunk]:
    """Rank chunks by cosine similarity to ``query_vector``.

    Metadata filters narrow the candidate set before ranking. ``min_score``
    drops weak matches instead of always returning ``top_k`` regardless of
    relevance. ``max_per_ticket`` prevents one verbose ticket's chunks from
    crowding out evidence from other tickets.
    """
    candidates = find_chunks(db, technology=technology, project=project)
    scored = sorted(
        (
            ScoredChunk(chunk=chunk, score=cosine_similarity(query_vector, chunk.embedding))
            for chunk in candidates
            if chunk.embedding is not None
        ),
        key=lambda scored_chunk: scored_chunk.score,
        reverse=True,
    )

    results: list[ScoredChunk] = []
    chunks_per_ticket: dict[str, int] = {}
    for scored_chunk in scored:
        if scored_chunk.score < min_score:
            break
        used = chunks_per_ticket.get(scored_chunk.chunk.ticket_id, 0)
        if used >= max_per_ticket:
            continue
        results.append(scored_chunk)
        chunks_per_ticket[scored_chunk.chunk.ticket_id] = used + 1
        if len(results) >= top_k:
            break

    return results
