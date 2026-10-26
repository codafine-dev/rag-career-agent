from __future__ import annotations

import datetime as dt
import hashlib

from pydantic import BaseModel
from pymongo.database import Database


class LongTermFact(BaseModel):
    """A durable fact remembered across conversations."""

    id: str
    key: str
    fact: str
    created_at: dt.datetime


def _fact_id(key: str, fact: str) -> str:
    """Deterministic id so re-storing the same fact is idempotent."""
    return hashlib.sha256(f"{key}::{fact}".encode()).hexdigest()


def remember_fact(
    db: Database,
    *,
    key: str,
    fact: str,
    now: dt.datetime | None = None,
) -> LongTermFact:
    """Store a durable fact owned by ``key``; storing duplicates changes nothing."""
    cleaned = fact.strip()
    record = LongTermFact(
        id=_fact_id(key, cleaned),
        key=key,
        fact=cleaned,
        created_at=now or dt.datetime.now(dt.timezone.utc),
    )
    doc = record.model_dump(mode="json")
    doc["_id"] = doc.pop("id")
    db.long_term_facts.replace_one({"_id": doc["_id"]}, doc, upsert=True)
    return record


def recall_facts(db: Database, *, key: str = "default", limit: int = 20) -> list[LongTermFact]:
    """Return the most recent facts stored for ``key``, newest first."""
    cursor = db.long_term_facts.find({"key": key}).sort("created_at", -1).limit(limit)
    results: list[LongTermFact] = []
    for doc in cursor:
        data = dict(doc)
        data["id"] = data.pop("_id")
        results.append(LongTermFact.model_validate(data))
    return results
