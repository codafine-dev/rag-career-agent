from __future__ import annotations

from pymongo.database import Database

from app.agent.models import AgentRunState, AgentRunStatus, AgentStep
from app.db.agent_run_repository import list_recent_runs, load_run_state, save_run_state


def _make_run(**overrides: object) -> AgentRunState:
    defaults: dict = {"run_id": "run-1", "question": "What fixed the pool exhaustion ticket?"}
    defaults.update(overrides)
    return AgentRunState.model_validate(defaults)


def test_save_and_load_run_state_roundtrip(db: Database) -> None:
    state = _make_run(steps=[AgentStep(index=0, thought="Searching tickets.")])

    save_run_state(db, state)
    loaded = load_run_state(db, "run-1")

    assert loaded == state


def test_save_run_state_checkpoints_over_previous_version(db: Database) -> None:
    save_run_state(db, _make_run(status=AgentRunStatus.RUNNING))
    save_run_state(db, _make_run(status=AgentRunStatus.COMPLETED, answer="Done."))

    assert db.agent_runs.count_documents({}) == 1
    loaded = load_run_state(db, "run-1")
    assert loaded is not None
    assert loaded.status == AgentRunStatus.COMPLETED
    assert loaded.answer == "Done."


def test_load_run_state_missing_returns_none(db: Database) -> None:
    assert load_run_state(db, "does-not-exist") is None


def test_list_recent_runs_newest_first(db: Database) -> None:
    import datetime as dt

    save_run_state(db, _make_run(run_id="run-1", created_at=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)))
    save_run_state(db, _make_run(run_id="run-2", created_at=dt.datetime(2026, 1, 2, tzinfo=dt.timezone.utc)))

    runs = list_recent_runs(db)

    assert [r.run_id for r in runs] == ["run-2", "run-1"]


def test_list_recent_runs_respects_limit(db: Database) -> None:
    import datetime as dt

    for i in range(5):
        save_run_state(
            db,
            _make_run(run_id=f"run-{i}", created_at=dt.datetime(2026, 1, 1 + i, tzinfo=dt.timezone.utc)),
        )

    assert len(list_recent_runs(db, limit=2)) == 2
