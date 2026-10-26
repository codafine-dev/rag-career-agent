from __future__ import annotations

import pytest

from app.agent.memory import ShortTermMemory, build_system_prompt
from app.llm.chat import ChatMessage


def test_short_term_memory_appends_in_order() -> None:
    memory = ShortTermMemory()
    memory.append(ChatMessage(role="system", content="You are helpful."))
    memory.append(ChatMessage(role="user", content="Hi"))

    assert [m.role for m in memory.all()] == ["system", "user"]


def test_short_term_memory_rejects_too_small_window() -> None:
    with pytest.raises(ValueError):
        ShortTermMemory(max_messages=1)


def test_short_term_memory_extend_appends_each_message_in_order() -> None:
    memory = ShortTermMemory()
    memory.extend(
        [
            ChatMessage(role="user", content="first"),
            ChatMessage(role="assistant", content="second"),
        ]
    )

    assert [m.content for m in memory.all()] == ["first", "second"]


def test_short_term_memory_trim_noop_when_only_system_messages_present() -> None:
    memory = ShortTermMemory(max_messages=2)
    # No non-system message exists to drop, so the window is allowed to
    # exceed max_messages rather than delete a system message.
    memory.append(ChatMessage(role="system", content="a"))
    memory.append(ChatMessage(role="system", content="b"))
    memory.append(ChatMessage(role="system", content="c"))

    assert [m.role for m in memory.all()] == ["system", "system", "system"]


def test_short_term_memory_trims_oldest_non_system_message_first() -> None:
    memory = ShortTermMemory(max_messages=3)
    memory.append(ChatMessage(role="system", content="System prompt."))
    memory.append(ChatMessage(role="user", content="first"))
    memory.append(ChatMessage(role="user", content="second"))
    memory.append(ChatMessage(role="user", content="third"))

    contents = [m.content for m in memory.all()]

    assert contents == ["System prompt.", "second", "third"]


def test_short_term_memory_clear_empties_history() -> None:
    memory = ShortTermMemory()
    memory.append(ChatMessage(role="user", content="hi"))
    memory.clear()

    assert memory.all() == []


def test_build_system_prompt_lists_recalled_facts() -> None:
    prompt = build_system_prompt(recalled_facts=["Uses Django.", "Prefers Postgres."])

    assert "- Uses Django." in prompt
    assert "- Prefers Postgres." in prompt


def test_build_system_prompt_handles_no_facts() -> None:
    prompt = build_system_prompt(recalled_facts=[])

    assert "(none yet)" in prompt
