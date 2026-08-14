from __future__ import annotations

import datetime as dt

from pymongo.database import Database

# Embedding calls cost fractions of a micro-dollar each (e.g. ~$0.0000004
# for a short query), so micro-dollar integers would round every call down
# to zero. Nano-dollars keep three more orders of magnitude of precision.
_NANOS_PER_USD = 1_000_000_000


def _today_key() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def record_usage(db: Database, *, tokens: int, cost_usd: float) -> None:
    """Atomically add today's token/cost usage.

    Cost is stored as integer nano-dollars to avoid float accumulation
    drift across many small increments.
    """
    cost_nanos = round(cost_usd * _NANOS_PER_USD)
    db.daily_api_usage.update_one(
        {"_id": _today_key()},
        {"$inc": {"tokens_used": tokens, "cost_nanos": cost_nanos}},
        upsert=True,
    )


def get_today_spend_usd(db: Database) -> float:
    doc = db.daily_api_usage.find_one({"_id": _today_key()})
    if doc is None:
        return 0.0
    return doc.get("cost_nanos", 0) / _NANOS_PER_USD


def is_budget_exceeded(db: Database, *, daily_budget_usd: float) -> bool:
    return get_today_spend_usd(db) >= daily_budget_usd
