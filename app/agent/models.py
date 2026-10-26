from __future__ import annotations

import datetime as dt
import enum

from pydantic import BaseModel, Field

from app.llm.chat import ProposedToolCall


class AgentRunStatus(str, enum.Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    FAILED = "failed"


class ToolObservation(BaseModel):
    """Outcome of executing one requested tool call."""

    tool_call_id: str
    tool_name: str
    ok: bool
    output: str = ""
    error: str | None = None
    duration_ms: int = 0


class AgentStep(BaseModel):
    """One planner decision plus everything that happened because of it."""

    index: int
    thought: str | None = None
    tool_calls: list[ProposedToolCall] = Field(default_factory=list)
    observations: list[ToolObservation] = Field(default_factory=list)
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))


class AgentRunState(BaseModel):
    """Checkpointable state of a single agent run.

    Persisted after every planning step so an interrupted run can be resumed
    from its last checkpoint instead of starting over.
    """

    run_id: str
    question: str
    status: AgentRunStatus = AgentRunStatus.RUNNING
    steps: list[AgentStep] = Field(default_factory=list)
    answer: str | None = None
    error: str | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    updated_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
