from __future__ import annotations

import pytest
from pymongo.database import Database

from app.db.usage_repository import get_today_spend_usd, is_budget_exceeded, record_usage


def test_get_today_spend_starts_at_zero(db: Database) -> None:
    assert get_today_spend_usd(db) == 0.0


def test_record_usage_accumulates_across_calls(db: Database) -> None:
    record_usage(db, tokens=20, cost_usd=0.0000004)
    record_usage(db, tokens=15, cost_usd=0.0000003)

    assert get_today_spend_usd(db) == pytest.approx(0.0000007)


def test_is_budget_exceeded_false_under_limit(db: Database) -> None:
    record_usage(db, tokens=20, cost_usd=0.10)

    assert is_budget_exceeded(db, daily_budget_usd=0.50) is False


def test_is_budget_exceeded_true_at_or_over_limit(db: Database) -> None:
    record_usage(db, tokens=20, cost_usd=0.50)

    assert is_budget_exceeded(db, daily_budget_usd=0.50) is True
