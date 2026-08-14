from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field


class Comment(BaseModel):
    author: str | None = None
    text: str
    date: dt.date | None = None


class Ticket(BaseModel):
    id: str
    title: str
    description: str
    project: str
    technologies: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)
    related_tickets: list[str] = Field(default_factory=list)
    comments: list[Comment] = Field(default_factory=list)
    implementation_notes: str | None = None
    resolution: str | None = None
    created_at: dt.date
    updated_at: dt.date | None = None
    resolved_at: dt.date | None = None
