from __future__ import annotations

from app.llm.chat import ChatMessage, ChatResult
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


class FakeChatProvider:
    """Scripted stand-in for a real chat/tool-calling API in tests.

    Seed it with a list of ``ChatResult``s (returned in order, one per call)
    and/or ``Exception``s (raised in order) to script a multi-step agent run
    without a real LLM. Records every ``messages`` list it was called with,
    so tests can assert on what the orchestrator actually sent back.
    """

    def __init__(self, results: list[ChatResult | Exception]) -> None:
        self._results = list(results)
        self.calls: list[list[ChatMessage]] = []

    def complete(self, *, messages: list[ChatMessage], tools: list[dict] | None = None) -> ChatResult:
        self.calls.append(list(messages))
        if not self._results:
            raise AssertionError("FakeChatProvider ran out of scripted results.")
        item = self._results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
