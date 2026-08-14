from __future__ import annotations

from functools import lru_cache

from pymongo import MongoClient
from pymongo.database import Database

from app.config import Settings


@lru_cache
def get_database(settings: Settings | None = None) -> Database:
    """Return a cached MongoDB database handle.

    Cached at module scope so serverless invocations reuse the connection
    across warm starts instead of reconnecting on every request.
    """
    settings = settings or Settings()
    client: MongoClient = MongoClient(settings.mongodb_uri)
    return client[settings.mongodb_db_name]
