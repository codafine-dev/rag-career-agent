from __future__ import annotations

from collections.abc import Sequence

from app.llm.chat import ChatMessage

DEFAULT_SYSTEM_PROMPT = """You are a career knowledge agent answering questions about the owner's engineering ticket history.

Rules:
- Use the provided tools to gather evidence before answering; never invent ticket ids, projects, or outcomes.
- Cite supporting tickets inline with their ids, e.g. [TICKET-002].
- If the evidence is insufficient, say so plainly instead of guessing.
- Save durable facts about the owner or their projects with the remember_fact tool.
- You have a bounded number of steps: prefer one good search over many vague ones.

Facts recalled from long-term memory:
{facts}
"""


class ShortTermMemory:
    """Rolling window of the current conversation, kept in process memory.

    The system message is pinned: trimming always drops the oldest non-system
    messages first so instructions never fall out of context mid-run.
    """

    def __init__(self, *, max_messages: int = 40) -> None:
        if max_messages < 2:
            raise ValueError("max_messages must be at least 2.")
        self._max_messages = max_messages
        self._messages: list[ChatMessage] = []

    def append(self, message: ChatMessage) -> None:
        self._messages.append(message)
        self._trim()

    def extend(self, messages: Sequence[ChatMessage]) -> None:
        for message in messages:
            self.append(message)

    def all(self) -> list[ChatMessage]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def _trim(self) -> None:
        while len(self._messages) > self._max_messages:
            for index in range(1, len(self._messages)):
                if self._messages[index].role != "system":
                    del self._messages[index]
                    break
            else:
                return


def build_system_prompt(
    *,
    recalled_facts: Sequence[str],
    template: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Render the system prompt, injecting recalled long-term facts."""
    if recalled_facts:
        facts_block = "\n".join(f"- {fact}" for fact in recalled_facts)
    else:
        facts_block = "(none yet)"
    return template.format(facts=facts_block)
