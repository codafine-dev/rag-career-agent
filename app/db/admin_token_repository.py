from __future__ import annotations

import datetime as dt
import hashlib
import secrets

from pymongo.database import Database


def _hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def create_token(db: Database, *, ttl_minutes: int) -> str:
    """Mint a one-time admin access token and store only its hash.

    :return: The raw token. This is the only time it is available in plaintext.
    """
    raw_token = secrets.token_urlsafe(32)
    now = dt.datetime.now(dt.timezone.utc)
    db.admin_tokens.insert_one(
        {
            "token_hash": _hash_token(raw_token),
            "created_at": now,
            "expires_at": now + dt.timedelta(minutes=ttl_minutes),
            "used_at": None,
        }
    )
    return raw_token


def consume_token(db: Database, raw_token: str) -> bool:
    """Atomically consume a one-time token if it is valid and unused.

    :return: True if the token was valid and has just been consumed,
        False if it was invalid, expired, or already used.
    """
    now = dt.datetime.now(dt.timezone.utc)
    result = db.admin_tokens.find_one_and_update(
        {
            "token_hash": _hash_token(raw_token),
            "used_at": None,
            "expires_at": {"$gt": now},
        },
        {"$set": {"used_at": now}},
    )
    return result is not None
