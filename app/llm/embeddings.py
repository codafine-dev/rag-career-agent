from __future__ import annotations

from typing import Protocol

from openai import OpenAI
from pydantic import BaseModel

# USD per 1M input tokens. Update if OpenAI changes pricing.
_PRICE_PER_MILLION_TOKENS = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}


class EmbeddingBatchResult(BaseModel):
    vectors: list[list[float]]
    tokens_used: int
    cost_usd: float


class EmbeddingsProvider(Protocol):
    def embed(self, texts: list[str]) -> EmbeddingBatchResult: ...


class OpenAIEmbeddingsProvider:
    """Embeddings provider backed by the OpenAI embeddings API.

    Used only for query-time and ingest-time embeddings, never for text
    generation, so cost per call stays tiny and bounded by the daily
    budget cap enforced in :mod:`app.db.usage_repository`.
    """

    def __init__(self, *, api_key: str, model: str = "text-embedding-3-small") -> None:
        if model not in _PRICE_PER_MILLION_TOKENS:
            raise ValueError(f"Unknown pricing for embeddings model: {model}")
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> EmbeddingBatchResult:
        response = self._client.embeddings.create(model=self._model, input=texts)
        vectors = [item.embedding for item in response.data]
        tokens_used = response.usage.total_tokens
        cost_usd = tokens_used / 1_000_000 * _PRICE_PER_MILLION_TOKENS[self._model]
        return EmbeddingBatchResult(vectors=vectors, tokens_used=tokens_used, cost_usd=cost_usd)
