from __future__ import annotations

from app.agent.models import AgentRunState, AgentRunStatus, AgentStep, ToolObservation
from app.llm.chat import ProposedToolCall


def test_agent_run_state_defaults() -> None:
    state = AgentRunState(run_id="run-1", question="What fixed it?")

    assert state.status == AgentRunStatus.RUNNING
    assert state.steps == []
    assert state.answer is None
    assert state.total_cost_usd == 0.0


def test_agent_step_carries_tool_calls_and_observations() -> None:
    step = AgentStep(
        index=0,
        thought="Need to search tickets.",
        tool_calls=[ProposedToolCall(id="c1", name="search_tickets", arguments={"query": "pool"})],
        observations=[
            ToolObservation(tool_call_id="c1", tool_name="search_tickets", ok=True, output="TICKET-002")
        ],
    )

    assert step.tool_calls[0].name == "search_tickets"
    assert step.observations[0].ok is True
    assert step.observations[0].output == "TICKET-002"


def test_tool_observation_defaults_to_no_error() -> None:
    observation = ToolObservation(tool_call_id="c1", tool_name="search_tickets", ok=False, error="timeout")

    assert observation.error == "timeout"
    assert observation.output == ""
