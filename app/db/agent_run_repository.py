from __future__ import annotations

from pymongo.database import Database

from app.agent.models import AgentRunState

_COLLECTION = "agent_runs"


def save_run_state(db: Database, state: AgentRunState) -> None:
    """Upsert a checkpoint of the agent run keyed by its run id."""
    doc = state.model_dump(mode="json")
    doc["_id"] = doc.pop("run_id")
    db[_COLLECTION].replace_one({"_id": doc["_id"]}, doc, upsert=True)


def load_run_state(db: Database, run_id: str) -> AgentRunState | None:
    """Return the checkpointed run with ``run_id``, or ``None`` if unknown."""
    doc = db[_COLLECTION].find_one({"_id": run_id})
    if doc is None:
        return None
    data = dict(doc)
    data["run_id"] = data.pop("_id")
    return AgentRunState.model_validate(data)


def list_recent_runs(db: Database, *, limit: int = 20) -> list[AgentRunState]:
    """Return the most recently started runs, newest first."""
    cursor = db[_COLLECTION].find().sort("created_at", -1).limit(limit)
    results: list[AgentRunState] = []
    for doc in cursor:
        data = dict(doc)
        data["run_id"] = data.pop("_id")
        results.append(AgentRunState.model_validate(data))
    return results
