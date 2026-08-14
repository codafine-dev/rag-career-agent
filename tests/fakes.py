from __future__ import annotations

from app.llm.embeddings import EmbeddingBatchResult


class FakeEmbeddingsProvider:
    """Deterministic stand-in for a real embeddings API in tests."""

    def __init__(self, vectors: list[list[float]], *, tokens_used: int = 10, cost_usd: float = 0.0001) -> None:
        self._vectors = vectors
        self._tokens_used = tokens_used
        self._cost_usd = cost_usd

    def embed(self, texts: list[str]) -> EmbeddingBatchResult:
        assert len(texts) == len(self._vectors), "fake provider must be seeded with one vector per text"
        return EmbeddingBatchResult(
            vectors=self._vectors,
            tokens_used=self._tokens_used,
            cost_usd=self._cost_usd,
        )
