from __future__ import annotations

from app.domain.models import Ticket
from app.retrieval.models import SourceField, TicketChunk

_DEFAULT_MAX_CHARS = 800
_DEFAULT_OVERLAP_CHARS = 150


def chunk_ticket(
    ticket: Ticket,
    *,
    max_chars: int = _DEFAULT_MAX_CHARS,
    overlap_chars: int = _DEFAULT_OVERLAP_CHARS,
) -> list[TicketChunk]:
    """Split a ticket into retrievable chunks, field-aware.

    Title+description and implementation-notes+resolution each become one
    chunk (they're short and belong together). Comment threads are
    concatenated and split with overlap only if they exceed ``max_chars``.
    """
    chunks: list[TicketChunk] = []
    index = 0

    summary_text = f"{ticket.title}\n\n{ticket.description}".strip()
    if summary_text:
        chunks.append(_make_chunk(ticket, "summary", index, summary_text))
        index += 1

    implementation_parts = [p for p in (ticket.implementation_notes, ticket.resolution) if p]
    if implementation_parts:
        implementation_text = "\n\n".join(implementation_parts)
        chunks.append(_make_chunk(ticket, "implementation", index, implementation_text))
        index += 1

    if ticket.comments:
        comments_text = "\n\n".join(
            f"{comment.author or 'unknown'}: {comment.text}" for comment in ticket.comments
        )
        for piece in _split_with_overlap(comments_text, max_chars, overlap_chars):
            chunks.append(_make_chunk(ticket, "comments", index, piece))
            index += 1

    return chunks


def _make_chunk(ticket: Ticket, source_field: SourceField, index: int, text: str) -> TicketChunk:
    return TicketChunk(
        id=f"{ticket.id}::{source_field}::{index}",
        ticket_id=ticket.id,
        source_field=source_field,
        chunk_index=index,
        text=text,
        project=ticket.project,
        technologies=list(ticket.technologies),
        created_at=ticket.created_at,
    )


def _split_with_overlap(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        pieces.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap_chars
    return pieces
