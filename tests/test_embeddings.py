from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.llm.embeddings import OpenAIEmbeddingsProvider


def test_openai_embeddings_provider_returns_vectors_and_cost(monkeypatch) -> None:
    fake_response = SimpleNamespace(
        data=[
            SimpleNamespace(embedding=[0.1, 0.2]),
            SimpleNamespace(embedding=[0.3, 0.4]),
        ],
        usage=SimpleNamespace(total_tokens=100),
    )
    fake_client = MagicMock()
    fake_client.embeddings.create.return_value = fake_response
    monkeypatch.setattr("app.llm.embeddings.OpenAI", lambda api_key: fake_client)

    provider = OpenAIEmbeddingsProvider(api_key="sk-test", model="text-embedding-3-small")
    result = provider.embed(["a", "b"])

    assert result.vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert result.tokens_used == 100
    assert result.cost_usd == pytest.approx(100 / 1_000_000 * 0.02)
    fake_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small", input=["a", "b"]
    )


def test_openai_embeddings_provider_rejects_unknown_model(monkeypatch) -> None:
    monkeypatch.setattr("app.llm.embeddings.OpenAI", lambda api_key: MagicMock())

    with pytest.raises(ValueError):
        OpenAIEmbeddingsProvider(api_key="sk-test", model="not-a-real-model")
