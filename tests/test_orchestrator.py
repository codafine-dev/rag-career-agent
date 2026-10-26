from __future__ import annotations

from pymongo.database import Database

from app.agent.models import AgentRunStatus
from app.agent.orchestrator import AgentRunner
from app.agent.tools import Tool
from app.db.agent_run_repository import load_run_state
from app.db.memory_repository import remember_fact
from app.llm.chat import AssistantTurn, ChatResult, ChatUsage, ProposedToolCall
from tests.fakes import FakeChatProvider


def _result(*, tool_calls: list[ProposedToolCall] | None = None, content: str | None = None) -> ChatResult:
    return ChatResult(
        turn=AssistantTurn(content=content, tool_calls=tool_calls or []),
        usage=ChatUsage(input_tokens=10, output_tokens=5),
        cost_usd=0.001,
    )


def _search_tool(handler) -> dict[str, Tool]:
    schema = {
        "name": "search_tickets",
        "description": "search",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }
    return {"search_tickets": Tool(schema=schema, handler=handler)}


def test_agent_stops_when_model_answers_without_tool_calls(db: Database) -> None:
    chat = FakeChatProvider([_result(content="No tickets needed to answer this.")])
    runner = AgentRunner(db=db, chat_provider=chat, tools={})

    state = runner.run("What's 2+2?")

    assert state.status == AgentRunStatus.COMPLETED
    assert state.answer == "No tickets needed to answer this."
    assert len(chat.calls) == 1


def test_agent_recalls_long_term_facts_into_first_system_prompt(db: Database) -> None:
    remember_fact(db, key="default", fact="Owner prefers Django over Flask.")
    chat = FakeChatProvider([_result(content="Sure.")])
    runner = AgentRunner(db=db, chat_provider=chat, tools={})

    runner.run("What framework does the owner use?")

    first_call_messages = chat.calls[0]
    system_message = next(m for m in first_call_messages if m.role == "system")
    assert "Owner prefers Django over Flask." in (system_message.content or "")


def test_agent_uses_default_memory_key_when_not_specified(db: Database) -> None:
    remember_fact(db, key="default", fact="Fact under the default key.")
    chat = FakeChatProvider([_result(content="ok")])
    runner = AgentRunner(db=db, chat_provider=chat, tools={})  # memory_key not passed

    runner.run("question")

    system_message = next(m for m in chat.calls[0] if m.role == "system")
    assert "Fact under the default key." in (system_message.content or "")


def test_agent_recalls_facts_scoped_to_its_own_memory_key(db: Database) -> None:
    remember_fact(db, key="other-agent", fact="Should not leak across agents.")
    chat = FakeChatProvider([_result(content="ok")])
    runner = AgentRunner(db=db, chat_provider=chat, tools={}, memory_key="this-agent")

    runner.run("question")

    system_message = next(m for m in chat.calls[0] if m.role == "system")
    assert "Should not leak across agents." not in (system_message.content or "")


def test_agent_calls_tool_then_answers_using_observation(db: Database) -> None:
    tool_call = ProposedToolCall(id="c1", name="search_tickets", arguments={"query": "pool exhaustion"})
    chat = FakeChatProvider(
        [
            _result(tool_calls=[tool_call]),
            _result(content="Fixed by increasing pool size [TICKET-002]."),
        ]
    )
    tools = _search_tool(lambda args: "TICKET-002: increased pool size")
    runner = AgentRunner(db=db, chat_provider=chat, tools=tools)

    state = runner.run("What fixed the pool exhaustion ticket?")

    assert state.status == AgentRunStatus.COMPLETED
    assert state.answer == "Fixed by increasing pool size [TICKET-002]."
    assert len(state.steps) == 2
    assert state.steps[0].observations[0].ok is True
    assert state.steps[0].observations[0].output == "TICKET-002: increased pool size"
    # the second planner call must have seen the tool's observation
    second_call_messages = chat.calls[1]
    assert any(m.role == "tool" and "TICKET-002" in (m.content or "") for m in second_call_messages)


def test_agent_accumulates_cost_and_tokens_across_steps(db: Database) -> None:
    tool_call = ProposedToolCall(id="c1", name="search_tickets", arguments={"query": "x"})
    chat = FakeChatProvider([_result(tool_calls=[tool_call]), _result(content="done")])
    tools = _search_tool(lambda args: "some result")
    runner = AgentRunner(db=db, chat_provider=chat, tools=tools)

    state = runner.run("question")

    assert state.total_input_tokens == 20
    assert state.total_output_tokens == 10
    assert state.total_cost_usd > 0


def test_agent_stops_at_max_steps_with_honest_message(db: Database) -> None:
    tool_call = ProposedToolCall(id="c1", name="search_tickets", arguments={"query": "x"})
    chat = FakeChatProvider([_result(tool_calls=[tool_call]) for _ in range(3)])
    tools = _search_tool(lambda args: "result")
    runner = AgentRunner(db=db, chat_provider=chat, tools=tools, max_steps=3)

    state = runner.run("question")

    assert state.status == AgentRunStatus.MAX_STEPS_EXCEEDED
    assert "couldn't reach" in (state.answer or "")
    assert len(state.steps) == 3


def test_unknown_tool_call_is_reported_as_error_not_a_crash(db: Database) -> None:
    tool_call = ProposedToolCall(id="c1", name="does_not_exist", arguments={})
    chat = FakeChatProvider([_result(tool_calls=[tool_call]), _result(content="fallback answer")])
    runner = AgentRunner(db=db, chat_provider=chat, tools={})

    state = runner.run("question")

    assert state.status == AgentRunStatus.COMPLETED
    assert state.steps[0].observations[0].ok is False
    assert "Unknown tool" in (state.steps[0].observations[0].error or "")


def test_tool_handler_exception_is_caught_and_reported(db: Database) -> None:
    def _boom(args: dict) -> str:
        raise RuntimeError("upstream API is down")

    tool_call = ProposedToolCall(id="c1", name="search_tickets", arguments={"query": "x"})
    chat = FakeChatProvider([_result(tool_calls=[tool_call]), _result(content="handled the error")])
    tools = _search_tool(_boom)
    runner = AgentRunner(db=db, chat_provider=chat, tools=tools)

    state = runner.run("question")

    assert state.status == AgentRunStatus.COMPLETED
    assert state.steps[0].observations[0].ok is False
    assert "upstream API is down" in (state.steps[0].observations[0].error or "")


def test_chat_provider_failure_after_retries_marks_run_failed(db: Database) -> None:
    chat = FakeChatProvider([RuntimeError("connection reset"), RuntimeError("connection reset")])
    runner = AgentRunner(db=db, chat_provider=chat, tools={}, chat_retries=1)

    state = runner.run("question")

    assert state.status == AgentRunStatus.FAILED
    assert "connection reset" in (state.error or "")
    assert len(chat.calls) == 2  # one initial attempt + one retry


def test_run_state_is_checkpointed_after_each_step(db: Database) -> None:
    tool_call = ProposedToolCall(id="c1", name="search_tickets", arguments={"query": "x"})
    chat = FakeChatProvider([_result(tool_calls=[tool_call]), _result(content="done")])
    tools = _search_tool(lambda args: "result")
    runner = AgentRunner(db=db, chat_provider=chat, tools=tools)

    state = runner.run("question")

    checkpointed = load_run_state(db, state.run_id)
    assert checkpointed is not None
    assert checkpointed.status == AgentRunStatus.COMPLETED
    assert checkpointed.answer == state.answer
