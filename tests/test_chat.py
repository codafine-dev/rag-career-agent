from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.llm.chat import (
    ChatMessage,
    ChatUsage,
    GeminiChatProvider,
    OpenAIChatProvider,
    ProposedToolCall,
    _to_gemini_contents,
    _to_gemini_tool,
    _to_openai_message,
)

_SEARCH_TOOL_SCHEMA = {
    "name": "search_tickets",
    "description": "Search the ticket corpus.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}


def test_openai_chat_provider_returns_tool_call_and_cost(monkeypatch) -> None:
    fake_message = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="search_tickets", arguments='{"query": "pool exhaustion"}'),
            )
        ],
    )
    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=fake_message)],
        usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10),
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response
    monkeypatch.setattr("app.llm.chat.OpenAI", lambda api_key: fake_client)

    provider = OpenAIChatProvider(api_key="sk-test", model="gpt-4o-mini")
    result = provider.complete(
        messages=[ChatMessage(role="user", content="What fixed the pool exhaustion ticket?")],
        tools=[_SEARCH_TOOL_SCHEMA],
    )

    assert result.turn.wants_tool_calls
    assert result.turn.tool_calls[0].name == "search_tickets"
    assert result.turn.tool_calls[0].arguments == {"query": "pool exhaustion"}
    assert result.cost_usd == pytest.approx(50 / 1_000_000 * 0.15 + 10 / 1_000_000 * 0.60)


def test_openai_chat_provider_malformed_tool_arguments_degrade_to_empty_dict(monkeypatch) -> None:
    fake_message = SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(id="call_1", function=SimpleNamespace(name="search_tickets", arguments="not json"))],
    )
    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=fake_message)],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response
    monkeypatch.setattr("app.llm.chat.OpenAI", lambda api_key: fake_client)

    provider = OpenAIChatProvider(api_key="sk-test", model="gpt-4o-mini")
    result = provider.complete(messages=[ChatMessage(role="user", content="hi")])

    assert result.turn.tool_calls[0].arguments == {}


def test_openai_chat_provider_rejects_unknown_model(monkeypatch) -> None:
    monkeypatch.setattr("app.llm.chat.OpenAI", lambda api_key: MagicMock())

    with pytest.raises(ValueError):
        OpenAIChatProvider(api_key="sk-test", model="not-a-real-model")


def test_openai_chat_provider_sends_full_multi_turn_history(monkeypatch) -> None:
    """Exercises the tool-role and assistant-with-tool-calls branches of _to_openai_message."""
    fake_message = SimpleNamespace(content="Fixed via TICKET-002.", tool_calls=None)
    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=fake_message)],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5),
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response
    monkeypatch.setattr("app.llm.chat.OpenAI", lambda api_key: fake_client)

    provider = OpenAIChatProvider(api_key="sk-test", model="gpt-4o-mini")
    provider.complete(
        messages=[
            ChatMessage(role="user", content="What fixed it?"),
            ChatMessage(
                role="assistant",
                tool_calls=[ProposedToolCall(id="c1", name="search_tickets", arguments={"query": "x"})],
            ),
            ChatMessage(role="tool", tool_call_id="c1", name="search_tickets", content="TICKET-002"),
        ]
    )

    sent_messages = fake_client.chat.completions.create.call_args.kwargs["messages"]
    assert sent_messages[1] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "search_tickets", "arguments": '{"query": "x"}'}}],
    }
    assert sent_messages[2] == {"role": "tool", "content": "TICKET-002", "tool_call_id": "c1"}


def test_to_openai_message_plain_roles_pass_through_content() -> None:
    assert _to_openai_message(ChatMessage(role="user", content="hi")) == {"role": "user", "content": "hi"}
    assert _to_openai_message(ChatMessage(role="system", content="be helpful")) == {
        "role": "system",
        "content": "be helpful",
    }


def test_chat_usage_total_tokens_sums_input_and_output() -> None:
    assert ChatUsage(input_tokens=10, output_tokens=5).total_tokens == 15


def test_gemini_chat_provider_returns_tool_call_and_cost(monkeypatch) -> None:
    fake_function_call = SimpleNamespace(id="fc_1", name="search_tickets", args={"query": "pool exhaustion"})
    fake_response = SimpleNamespace(
        function_calls=[fake_function_call],
        text=None,
        usage_metadata=SimpleNamespace(prompt_token_count=50, candidates_token_count=10),
    )
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response
    monkeypatch.setattr("app.llm.chat.genai.Client", lambda api_key: fake_client)

    provider = GeminiChatProvider(api_key="test-key", model="gemini-2.5-flash")
    result = provider.complete(
        messages=[ChatMessage(role="user", content="What fixed the pool exhaustion ticket?")],
        tools=[_SEARCH_TOOL_SCHEMA],
    )

    assert result.turn.wants_tool_calls
    assert result.turn.tool_calls[0].name == "search_tickets"
    assert result.turn.tool_calls[0].arguments == {"query": "pool exhaustion"}
    assert result.cost_usd == pytest.approx(50 / 1_000_000 * 0.30 + 10 / 1_000_000 * 2.50)


def test_gemini_chat_provider_plain_text_answer_has_no_tool_calls(monkeypatch) -> None:
    fake_response = SimpleNamespace(
        function_calls=None,
        text="It was fixed by increasing pool size [TICKET-002].",
        usage_metadata=SimpleNamespace(prompt_token_count=5, candidates_token_count=5),
    )
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response
    monkeypatch.setattr("app.llm.chat.genai.Client", lambda api_key: fake_client)

    provider = GeminiChatProvider(api_key="test-key")
    result = provider.complete(messages=[ChatMessage(role="user", content="Recap the fix.")])

    assert not result.turn.wants_tool_calls
    assert result.turn.content == "It was fixed by increasing pool size [TICKET-002]."


def test_gemini_chat_provider_rejects_unknown_model(monkeypatch) -> None:
    monkeypatch.setattr("app.llm.chat.genai.Client", lambda api_key: MagicMock())

    with pytest.raises(ValueError):
        GeminiChatProvider(api_key="test-key", model="not-a-real-model")


def test_to_gemini_contents_splits_system_and_maps_roles() -> None:
    messages = [
        ChatMessage(role="system", content="You are helpful."),
        ChatMessage(role="user", content="What fixed it?"),
        ChatMessage(
            role="assistant",
            tool_calls=[ProposedToolCall(id="c1", name="search_tickets", arguments={"query": "x"})],
        ),
        ChatMessage(role="tool", tool_call_id="c1", name="search_tickets", content="TICKET-002"),
        ChatMessage(role="assistant", content="Fixed via TICKET-002."),
    ]

    system, contents = _to_gemini_contents(messages)

    assert system == "You are helpful."
    assert [c.role for c in contents] == ["user", "model", "user", "model"]
    assert contents[1].parts[0].function_call.name == "search_tickets"
    assert contents[2].parts[0].function_response.name == "search_tickets"
    assert contents[2].parts[0].function_response.response == {"result": "TICKET-002"}


def test_to_gemini_contents_requires_name_on_tool_messages() -> None:
    messages = [ChatMessage(role="tool", tool_call_id="c1", content="result")]

    with pytest.raises(ValueError, match="ChatMessage.name"):
        _to_gemini_contents(messages)


def test_to_gemini_tool_converts_schema_list() -> None:
    tool = _to_gemini_tool([_SEARCH_TOOL_SCHEMA])

    assert len(tool.function_declarations) == 1
    assert tool.function_declarations[0].name == "search_tickets"
    assert tool.function_declarations[0].parameters_json_schema["required"] == ["query"]
