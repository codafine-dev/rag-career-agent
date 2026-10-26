from __future__ import annotations

from typing import Protocol

from google import genai
from google.genai import types as genai_types
from openai import OpenAI
from pydantic import BaseModel

# USD per 1M input tokens. Update if OpenAI changes pricing.
_PRICE_PER_MILLION_TOKENS = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}

# USD per 1M input tokens. Update if Google changes pricing.
_GEMINI_PRICE_PER_MILLION_TOKENS = {
    "gemini-embedding-001": 0.15,
}

# Gemini's embed_content response reports billable *characters*, not tokens
# (unlike OpenAI's usage.total_tokens). ~4 chars/token is the same rough
# English-text approximation OpenAI's own docs use; tokens_used below is
# therefore an estimate, not a value read directly off the API response.
_CHARS_PER_TOKEN_ESTIMATE = 4


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


class GeminiEmbeddingsProvider:
    """Embeddings provider backed by the Gemini API (``google-genai`` SDK).

    See :class:`OpenAIEmbeddingsProvider` for the role this plays in the
    daily budget cap; ``tokens_used`` here is an estimate (see
    ``_CHARS_PER_TOKEN_ESTIMATE``) since Gemini reports billable characters,
    not tokens.
    """

    def __init__(self, *, api_key: str, model: str = "gemini-embedding-001") -> None:
        if model not in _GEMINI_PRICE_PER_MILLION_TOKENS:
            raise ValueError(f"Unknown pricing for Gemini embeddings model: {model}")
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> EmbeddingBatchResult:
        response = self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        vectors = [embedding.values for embedding in response.embeddings]
        billable_characters = response.metadata.billable_character_count if response.metadata else 0
        tokens_used = (billable_characters or 0) // _CHARS_PER_TOKEN_ESTIMATE
        cost_usd = tokens_used / 1_000_000 * _GEMINI_PRICE_PER_MILLION_TOKENS[self._model]
        return EmbeddingBatchResult(vectors=vectors, tokens_used=tokens_used, cost_usd=cost_usd)
