from __future__ import annotations

import time
import uuid
from collections.abc import Mapping

from pymongo.database import Database

from app.agent.memory import ShortTermMemory, build_system_prompt
from app.agent.models import AgentRunState, AgentRunStatus, AgentStep, ToolObservation
from app.agent.tools import Tool
from app.db.agent_run_repository import save_run_state
from app.db.memory_repository import recall_facts
from app.llm.chat import ChatMessage, ChatProvider, ProposedToolCall

_DEFAULT_MAX_STEPS = 6
_DEFAULT_CHAT_RETRIES = 2


class AgentRunner:
    """Plan -> call tool -> observe -> iterate loop over the ticket knowledge base.

    Checkpoints :class:`AgentRunState` to Mongo after every step, so a run
    interrupted mid-loop (process killed, crash) can be resumed by reloading
    the last checkpoint with :func:`app.db.agent_run_repository.load_run_state`
    instead of starting over.
    """

    def __init__(
        self,
        *,
        db: Database,
        chat_provider: ChatProvider,
        tools: Mapping[str, Tool],
        max_steps: int = _DEFAULT_MAX_STEPS,
        memory_key: str = "default",
        chat_retries: int = _DEFAULT_CHAT_RETRIES,
    ) -> None:
        self._db = db
        self._chat = chat_provider
        self._tools = tools
        self._max_steps = max_steps
        self._memory_key = memory_key
        self._chat_retries = chat_retries

    def run(self, question: str) -> AgentRunState:
        recalled = [fact.fact for fact in recall_facts(self._db, key=self._memory_key)]
        system_prompt = build_system_prompt(recalled_facts=recalled)

        short_term = ShortTermMemory()
        short_term.append(ChatMessage(role="system", content=system_prompt))
        short_term.append(ChatMessage(role="user", content=question))

        state = AgentRunState(run_id=str(uuid.uuid4()), question=question)
        tool_schemas = [tool.schema for tool in self._tools.values()]

        for step_index in range(self._max_steps):
            try:
                result = self._complete_with_retry(short_term.all(), tool_schemas)
            except Exception as exc:  # noqa: BLE001 - top-level run boundary: any planner failure ends the run cleanly
                state.status = AgentRunStatus.FAILED
                state.error = f"Chat provider call failed after retries: {exc}"
                save_run_state(self._db, state)
                return state

            state.total_input_tokens += result.usage.input_tokens
            state.total_output_tokens += result.usage.output_tokens
            state.total_cost_usd += result.cost_usd

            if not result.turn.wants_tool_calls:
                state.status = AgentRunStatus.COMPLETED
                state.answer = result.turn.content
                state.steps.append(AgentStep(index=step_index, thought=result.turn.content))
                save_run_state(self._db, state)
                return state

            short_term.append(ChatMessage(role="assistant", tool_calls=result.turn.tool_calls))
            observations = [self._execute_tool(call, short_term) for call in result.turn.tool_calls]

            state.steps.append(AgentStep(index=step_index, tool_calls=result.turn.tool_calls, observations=observations))
            save_run_state(self._db, state)

        state.status = AgentRunStatus.MAX_STEPS_EXCEEDED
        state.answer = (
            "I couldn't reach a confident, evidence-backed answer within the allotted steps. "
            "Try narrowing the question or asking about a specific ticket id."
        )
        save_run_state(self._db, state)
        return state

    def _complete_with_retry(self, messages: list[ChatMessage], tool_schemas: list[dict]):
        """Retry the planner call on transient failures; let the last error propagate."""
        last_error: Exception | None = None
        for attempt in range(self._chat_retries + 1):
            try:
                return self._chat.complete(messages=messages, tools=tool_schemas)
            except Exception as exc:  # noqa: BLE001 - provider errors aren't a fixed typed set across OpenAI/Gemini
                last_error = exc
                if attempt < self._chat_retries:
                    time.sleep(min(2**attempt, 5))
        assert last_error is not None
        raise last_error

    def _execute_tool(self, call: ProposedToolCall, short_term: ShortTermMemory) -> ToolObservation:
        start = time.monotonic()
        tool = self._tools.get(call.name)

        if tool is None:
            observation = ToolObservation(
                tool_call_id=call.id,
                tool_name=call.name,
                ok=False,
                error=f"Unknown tool: {call.name!r}. Available tools: {sorted(self._tools)}.",
            )
        else:
            try:
                output = tool.handler(call.arguments)
                observation = ToolObservation(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    ok=True,
                    output=output,
                    duration_ms=int((time.monotonic() - start) * 1000),
                )
            except Exception as exc:  # noqa: BLE001 - a tool handler failing must not crash the run
                observation = ToolObservation(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    ok=False,
                    error=str(exc),
                    duration_ms=int((time.monotonic() - start) * 1000),
                )

        # Feed the outcome (success or error) back to the model so it can
        # adjust its next call rather than repeat the same mistake blind.
        short_term.append(
            ChatMessage(
                role="tool",
                tool_call_id=call.id,
                name=call.name,
                content=observation.output if observation.ok else f"Error: {observation.error}",
            )
        )
        return observation
