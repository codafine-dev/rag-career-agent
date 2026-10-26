from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pymongo.database import Database

from app.db.memory_repository import remember_fact
from app.db.ticket_repository import get_ticket, list_tickets
from app.llm.embeddings import EmbeddingsProvider
from app.retrieval.retriever import search_chunks

ToolHandler = Callable[[dict[str, Any]], str]

_TICKET_SNIPPET_CHARS = 300


@dataclass
class Tool:
    """A callable the agent can invoke, paired with the schema it's advertised under."""

    schema: dict[str, Any]
    handler: ToolHandler


def build_tools(
    db: Database,
    embeddings: EmbeddingsProvider,
    *,
    memory_key: str = "default",
) -> dict[str, Tool]:
    """Build the agent's tool registry.

    Each handler takes the raw arguments dict the planner model produced and
    returns a plain-text observation - never raises for expected "no results"
    cases, only for genuinely unexpected failures the orchestrator's error
    recovery should see.
    """

    def _search_tickets(args: dict[str, Any]) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return "Error: 'query' argument is required and cannot be empty."

        embedded = embeddings.embed([query])
        chunks = search_chunks(
            db,
            query_vector=embedded.vectors[0],
            technology=args.get("technology"),
            project=args.get("project"),
        )
        if not chunks:
            return "No matching tickets found for that query. Try a different phrasing or broader terms."

        lines = [
            f"[{scored.chunk.ticket_id}] ({scored.chunk.source_field}, score={scored.score:.2f}): "
            f"{scored.chunk.text[:_TICKET_SNIPPET_CHARS]}"
            for scored in chunks
        ]
        return "\n".join(lines)

    def _get_ticket(args: dict[str, Any]) -> str:
        ticket_id = (args.get("ticket_id") or "").strip()
        if not ticket_id:
            return "Error: 'ticket_id' argument is required."

        ticket = get_ticket(db, ticket_id)
        if ticket is None:
            return f"No ticket found with id {ticket_id!r}."

        parts = [f"Title: {ticket.title}", f"Description: {ticket.description}"]
        if ticket.implementation_notes:
            parts.append(f"Implementation notes: {ticket.implementation_notes}")
        if ticket.resolution:
            parts.append(f"Resolution: {ticket.resolution}")
        return "\n".join(parts)

    def _list_tickets(args: dict[str, Any]) -> str:
        tickets = list_tickets(db, technology=args.get("technology"), project=args.get("project"))
        if not tickets:
            return "No tickets matched those filters."
        return "\n".join(f"[{ticket.id}] {ticket.title} (project={ticket.project})" for ticket in tickets)

    def _remember_fact(args: dict[str, Any]) -> str:
        fact = (args.get("fact") or "").strip()
        if not fact:
            return "Error: 'fact' argument is required and cannot be empty."
        remember_fact(db, key=memory_key, fact=fact)
        return "Saved."

    return {
        "search_tickets": Tool(
            schema={
                "name": "search_tickets",
                "description": (
                    "Semantic search over ticket content (titles, descriptions, implementation "
                    "notes, comments). Use this to find tickets relevant to a question."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural-language search query."},
                        "technology": {"type": "string", "description": "Optional exact technology filter."},
                        "project": {"type": "string", "description": "Optional exact project filter."},
                    },
                    "required": ["query"],
                },
            },
            handler=_search_tickets,
        ),
        "get_ticket": Tool(
            schema={
                "name": "get_ticket",
                "description": "Fetch the full detail of one ticket by its exact id, once you know it.",
                "parameters": {
                    "type": "object",
                    "properties": {"ticket_id": {"type": "string", "description": "e.g. 'TICKET-002'."}},
                    "required": ["ticket_id"],
                },
            },
            handler=_get_ticket,
        ),
        "list_tickets": Tool(
            schema={
                "name": "list_tickets",
                "description": "List tickets (titles only), optionally filtered by exact technology or project.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "technology": {"type": "string", "description": "Optional exact technology filter."},
                        "project": {"type": "string", "description": "Optional exact project filter."},
                    },
                    "required": [],
                },
            },
            handler=_list_tickets,
        ),
        "remember_fact": Tool(
            schema={
                "name": "remember_fact",
                "description": "Save a durable fact about the owner or their projects for future conversations.",
                "parameters": {
                    "type": "object",
                    "properties": {"fact": {"type": "string", "description": "The fact to remember, in one sentence."}},
                    "required": ["fact"],
                },
            },
            handler=_remember_fact,
        ),
    }
