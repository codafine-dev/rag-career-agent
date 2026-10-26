from __future__ import annotations

import datetime as dt

from pymongo.database import Database

from app.db.memory_repository import recall_facts, remember_fact


def test_remember_and_recall_fact_roundtrip(db: Database) -> None:
    remember_fact(db, key="owner", fact="Prefers concise answers.")

    facts = recall_facts(db, key="owner")

    assert [f.fact for f in facts] == ["Prefers concise answers."]


def test_remember_fact_is_idempotent_for_identical_fact(db: Database) -> None:
    remember_fact(db, key="owner", fact="Uses PostgreSQL for most services.")
    remember_fact(db, key="owner", fact="Uses PostgreSQL for most services.")

    assert db.long_term_facts.count_documents({}) == 1


def test_remember_fact_strips_whitespace_before_dedup(db: Database) -> None:
    remember_fact(db, key="owner", fact="Same fact.")
    remember_fact(db, key="owner", fact="  Same fact.  ")

    assert db.long_term_facts.count_documents({}) == 1


def test_recall_facts_scoped_by_key(db: Database) -> None:
    remember_fact(db, key="owner", fact="Fact about owner.")
    remember_fact(db, key="project-x", fact="Fact about project-x.")

    assert [f.fact for f in recall_facts(db, key="owner")] == ["Fact about owner."]
    assert [f.fact for f in recall_facts(db, key="project-x")] == ["Fact about project-x."]


def test_recall_facts_newest_first_and_respects_limit(db: Database) -> None:
    now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    remember_fact(db, key="owner", fact="oldest", now=now)
    remember_fact(db, key="owner", fact="middle", now=now + dt.timedelta(days=1))
    remember_fact(db, key="owner", fact="newest", now=now + dt.timedelta(days=2))

    facts = recall_facts(db, key="owner", limit=2)

    assert [f.fact for f in facts] == ["newest", "middle"]
