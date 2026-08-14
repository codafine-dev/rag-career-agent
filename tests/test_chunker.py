from __future__ import annotations

import datetime as dt

from app.domain.models import Comment, Ticket
from app.retrieval.chunker import chunk_ticket


def _make_ticket(**overrides: object) -> Ticket:
    defaults: dict = {
        "id": "TICKET-1",
        "title": "Title",
        "description": "Description",
        "project": "core-api",
        "technologies": ["Django", "PostgreSQL"],
        "created_at": dt.date(2025, 1, 1),
    }
    defaults.update(overrides)
    return Ticket.model_validate(defaults)


def test_chunk_ticket_produces_summary_and_implementation_chunks() -> None:
    ticket = _make_ticket(implementation_notes="Did the thing.", resolution="Shipped.")

    chunks = chunk_ticket(ticket)

    assert [c.source_field for c in chunks] == ["summary", "implementation"]
    assert chunks[0].text == "Title\n\nDescription"
    assert "Did the thing." in chunks[1].text
    assert "Shipped." in chunks[1].text


def test_chunk_ticket_carries_metadata_onto_every_chunk() -> None:
    ticket = _make_ticket(resolution="Done.")

    chunks = chunk_ticket(ticket)

    for chunk in chunks:
        assert chunk.ticket_id == "TICKET-1"
        assert chunk.project == "core-api"
        assert chunk.technologies == ["Django", "PostgreSQL"]
        assert chunk.created_at == dt.date(2025, 1, 1)


def test_chunk_ticket_skips_empty_optional_fields() -> None:
    ticket = _make_ticket()

    chunks = chunk_ticket(ticket)

    assert [c.source_field for c in chunks] == ["summary"]


def test_chunk_ticket_short_comments_become_one_chunk() -> None:
    ticket = _make_ticket(comments=[Comment(author="lin", text="Short comment.")])

    chunks = chunk_ticket(ticket)

    comment_chunks = [c for c in chunks if c.source_field == "comments"]
    assert len(comment_chunks) == 1
    assert "Short comment." in comment_chunks[0].text


def test_chunk_ticket_splits_long_comments_with_overlap() -> None:
    long_text = "x" * 2000
    ticket = _make_ticket(comments=[Comment(author="lin", text=long_text)])

    chunks = chunk_ticket(ticket, max_chars=800, overlap_chars=150)

    comment_chunks = [c for c in chunks if c.source_field == "comments"]
    assert len(comment_chunks) > 1
    # chunk_index increments continuously across all chunks for the ticket.
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # Consecutive pieces overlap: the tail of one appears at the head of the next.
    first_tail = comment_chunks[0].text[-100:]
    assert first_tail in comment_chunks[1].text


def test_chunk_ids_are_unique_and_ticket_scoped() -> None:
    ticket = _make_ticket(
        resolution="Done.",
        comments=[Comment(author="lin", text="Short comment.")],
    )

    chunks = chunk_ticket(ticket)

    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(cid.startswith("TICKET-1::") for cid in ids)
