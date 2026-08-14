from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field

SourceField = Literal["summary", "implementation", "comments"]


class TicketChunk(BaseModel):
    id: str
    ticket_id: str
    source_field: SourceField
    chunk_index: int
    text: str
    project: str
    technologies: list[str] = Field(default_factory=list)
    created_at: dt.date
    embedding: list[float] | None = None


class ScoredChunk(BaseModel):
    chunk: TicketChunk
    score: float
