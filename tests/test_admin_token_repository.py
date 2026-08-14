from __future__ import annotations

import datetime as dt

from pymongo.database import Database

from app.db.admin_token_repository import consume_token, create_token


def test_valid_token_can_be_consumed_once(db: Database) -> None:
    token = create_token(db, ttl_minutes=10)

    assert consume_token(db, token) is True
    assert consume_token(db, token) is False


def test_unknown_token_is_rejected(db: Database) -> None:
    assert consume_token(db, "not-a-real-token") is False


def test_expired_token_is_rejected(db: Database) -> None:
    token = create_token(db, ttl_minutes=10)
    db.admin_tokens.update_one(
        {},
        {"$set": {"expires_at": dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=1)}},
    )

    assert consume_token(db, token) is False
