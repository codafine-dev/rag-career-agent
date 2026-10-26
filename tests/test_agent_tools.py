from __future__ import annotations

import datetime as dt

from pymongo.database import Database

from app.agent.tools import build_tools
from app.db.chunk_repository import upsert_chunks
from app.db.memory_repository import recall_facts
from app.db.ticket_repository import upsert_ticket
from app.domain.models import Ticket
from app.retrieval.models import TicketChunk
from tests.fakes import FakeEmbeddingsProvider


def _make_ticket(**overrides: object) -> Ticket:
    defaults: dict = {
        "id": "TICKET-002",
        "title": "Fix connection pool exhaustion",
        "description": "The service ran out of DB connections under load.",
        "project": "billing-service",
        "technologies": ["Django", "PostgreSQL"],
        "implementation_notes": "Raised the pool size and added a queue timeout.",
        "resolution": "Increased pool size from 5 to 20; monitored under load.",
        "created_at": dt.date(2026, 1, 1),
    }
    defaults.update(overrides)
    return Ticket.model_validate(defaults)


def _make_chunk(**overrides: object) -> TicketChunk:
    defaults: dict = {
        "id": "TICKET-002-summary-0",
        "ticket_id": "TICKET-002",
        "source_field": "summary",
        "chunk_index": 0,
        "text": "Fix connection pool exhaustion: increased pool size from 5 to 20.",
        "project": "billing-service",
        "technologies": ["Django", "PostgreSQL"],
        "created_at": dt.date(2026, 1, 1),
        "embedding": [1.0, 0.0],
    }
    defaults.update(overrides)
    return TicketChunk.model_validate(defaults)


def test_search_tickets_returns_scored_snippets(db: Database) -> None:
    upsert_chunks(db, [_make_chunk()])
    tools = build_tools(db, FakeEmbeddingsProvider(vectors=[[1.0, 0.0]]))

    output = tools["search_tickets"].handler({"query": "pool exhaustion"})

    assert "[TICKET-002]" in output
    assert "pool size" in output


def test_search_tickets_filters_by_technology(db: Database) -> None:
    upsert_chunks(
        db,
        [
            _make_chunk(id="c-django", technologies=["Django"], text="Django pool fix.", embedding=[1.0, 0.0]),
            _make_chunk(id="c-node", technologies=["Node"], text="Node pool fix.", embedding=[1.0, 0.0]),
        ],
    )
    tools = build_tools(db, FakeEmbeddingsProvider(vectors=[[1.0, 0.0]]))

    output = tools["search_tickets"].handler({"query": "pool", "technology": "Django"})

    assert "Django pool fix." in output
    assert "Node pool fix." not in output


def test_search_tickets_filters_by_project(db: Database) -> None:
    upsert_chunks(
        db,
        [
            _make_chunk(id="c-a", project="billing-service", text="Billing fix.", embedding=[1.0, 0.0]),
            _make_chunk(id="c-b", project="core-api", text="Core fix.", embedding=[1.0, 0.0]),
        ],
    )
    tools = build_tools(db, FakeEmbeddingsProvider(vectors=[[1.0, 0.0]]))

    output = tools["search_tickets"].handler({"query": "fix", "project": "billing-service"})

    assert "Billing fix." in output
    assert "Core fix." not in output


def test_search_tickets_requires_query(db: Database) -> None:
    tools = build_tools(db, FakeEmbeddingsProvider(vectors=[]))

    output = tools["search_tickets"].handler({})

    assert output.startswith("Error:")


def test_search_tickets_reports_no_matches(db: Database) -> None:
    tools = build_tools(db, FakeEmbeddingsProvider(vectors=[[1.0, 0.0]]))

    output = tools["search_tickets"].handler({"query": "nothing indexed yet"})

    assert "No matching tickets" in output


def test_get_ticket_returns_full_detail(db: Database) -> None:
    upsert_ticket(db, _make_ticket())
    tools = build_tools(db, FakeEmbeddingsProvider(vectors=[]))

    output = tools["get_ticket"].handler({"ticket_id": "TICKET-002"})

    assert "Fix connection pool exhaustion" in output
    assert "Increased pool size from 5 to 20" in output


def test_get_ticket_requires_ticket_id(db: Database) -> None:
    tools = build_tools(db, FakeEmbeddingsProvider(vectors=[]))

    output = tools["get_ticket"].handler({})

    assert output.startswith("Error:")


def test_get_ticket_missing_id_reports_not_found(db: Database) -> None:
    tools = build_tools(db, FakeEmbeddingsProvider(vectors=[]))

    output = tools["get_ticket"].handler({"ticket_id": "TICKET-999"})

    assert "No ticket found" in output


def test_list_tickets_filters_by_technology(db: Database) -> None:
    upsert_ticket(db, _make_ticket(id="T-1", technologies=["AWS"]))
    upsert_ticket(db, _make_ticket(id="T-2", technologies=["Django"]))
    tools = build_tools(db, FakeEmbeddingsProvider(vectors=[]))

    output = tools["list_tickets"].handler({"technology": "AWS"})

    assert "[T-1]" in output
    assert "[T-2]" not in output


def test_list_tickets_reports_no_matches(db: Database) -> None:
    tools = build_tools(db, FakeEmbeddingsProvider(vectors=[]))

    output = tools["list_tickets"].handler({})

    assert output == "No tickets matched those filters."


def test_remember_fact_persists_and_is_idempotent(db: Database) -> None:
    tools = build_tools(db, FakeEmbeddingsProvider(vectors=[]), memory_key="owner")

    tools["remember_fact"].handler({"fact": "Prefers concise answers."})
    tools["remember_fact"].handler({"fact": "Prefers concise answers."})

    facts = recall_facts(db, key="owner")
    assert [f.fact for f in facts] == ["Prefers concise answers."]


def test_remember_fact_requires_nonempty_fact(db: Database) -> None:
    tools = build_tools(db, FakeEmbeddingsProvider(vectors=[]))

    output = tools["remember_fact"].handler({"fact": "   "})

    assert output.startswith("Error:")
