from __future__ import annotations

import mongomock
import pytest
from pymongo.database import Database


@pytest.fixture
def db() -> Database:
    return mongomock.MongoClient().rag_career_agent_test
