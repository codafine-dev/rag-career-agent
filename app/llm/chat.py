from __future__ import annotations

import json
from typing import Any, Literal, Protocol

from google import genai
from google.genai import types as genai_types
from openai import OpenAI
from pydantic import BaseModel, Field

# USD per 1M tokens as (input, output). Update if OpenAI changes pricing.
_PRICE_PER_MILLION_TOKENS = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}

# USD per 1M tokens as (input, output). Update if Google changes pricing.
# gemini-2.5-flash is also reachable free of charge via an AI Studio API key
# (rate-limited); the cost tracked here reflects paid-tier pricing so the
# daily budget cap still means something if that key is ever swapped for one
# without free-tier limits.
_GEMINI_PRICE_PER_MILLION_TOKENS = {
    "gemini-2.5-flash": (0.30, 2.50),
}


class ProposedToolCall(BaseModel):
    """A tool invocation requested by the planner model."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """One message in the planner conversation.

    ``tool_calls`` is only populated on assistant messages that request tool
    executions; ``tool_call_id``/``name`` are only set on role="tool" result
    messages. OpenAI matches tool results by id alone; Gemini's function
    response requires the tool name too, so it's carried here rather than
    looked up by providers that don't need it.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ProposedToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


class AssistantTurn(BaseModel):
    """The planner's raw output for one step of the loop."""

    content: str | None = None
    tool_calls: list[ProposedToolCall] = Field(default_factory=list)

    @property
    def wants_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class ChatUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ChatResult(BaseModel):
    turn: AssistantTurn
    usage: ChatUsage
    cost_usd: float


class ChatProvider(Protocol):
    def complete(
        self,
        *,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult: ...


class OpenAIChatProvider:
    """Chat provider backed by the OpenAI chat completions API.

    Used as the agent planner. Cost per call is computed from the pricing
    table above and recorded by the caller against the daily budget.
    """

    def __init__(self, *, api_key: str, model: str = "gpt-4o-mini") -> None:
        if model not in _PRICE_PER_MILLION_TOKENS:
            raise ValueError(f"Unknown pricing for chat model: {model}")
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete(
        self,
        *,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [_to_openai_message(message) for message in messages],
        }
        if tools:
            kwargs["tools"] = [{"type": "function", "function": schema} for schema in tools]

        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        tool_calls = [
            ProposedToolCall(
                id=call.id,
                name=call.function.name,
                arguments=_parse_arguments(call.function.arguments),
            )
            for call in (message.tool_calls or [])
        ]

        price_in, price_out = _PRICE_PER_MILLION_TOKENS[self._model]
        usage = ChatUsage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
        cost_usd = (
            usage.input_tokens / 1_000_000 * price_in
            + usage.output_tokens / 1_000_000 * price_out
        )
        return ChatResult(
            turn=AssistantTurn(content=message.content, tool_calls=tool_calls),
            usage=usage,
            cost_usd=cost_usd,
        )


class GeminiChatProvider:
    """Chat provider backed by the Gemini API (``google-genai`` SDK).

    Reachable on Google AI Studio's free tier for this project's low request
    volume, behind the same :class:`ChatProvider` protocol as the OpenAI
    provider - either can be swapped in without touching the agent loop.
    """

    def __init__(self, *, api_key: str, model: str = "gemini-2.5-flash") -> None:
        if model not in _GEMINI_PRICE_PER_MILLION_TOKENS:
            raise ValueError(f"Unknown pricing for Gemini chat model: {model}")
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def complete(
        self,
        *,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        system_instruction, contents = _to_gemini_contents(messages)
        config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction or None,
            tools=[_to_gemini_tool(tools)] if tools else None,
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        tool_calls = [
            ProposedToolCall(id=call.id or f"{call.name}-{index}", name=call.name, arguments=call.args or {})
            for index, call in enumerate(response.function_calls or [])
        ]

        usage = response.usage_metadata
        input_tokens = (usage.prompt_token_count if usage else None) or 0
        output_tokens = (usage.candidates_token_count if usage else None) or 0
        price_in, price_out = _GEMINI_PRICE_PER_MILLION_TOKENS[self._model]
        cost_usd = input_tokens / 1_000_000 * price_in + output_tokens / 1_000_000 * price_out

        return ChatResult(
            turn=AssistantTurn(content=None if tool_calls else response.text, tool_calls=tool_calls),
            usage=ChatUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            cost_usd=cost_usd,
        )


def _to_gemini_tool(tools: list[dict[str, Any]]) -> genai_types.Tool:
    """Convert this project's provider-agnostic tool schemas to a Gemini ``Tool``."""
    declarations = [
        genai_types.FunctionDeclaration(
            name=schema["name"],
            description=schema.get("description", ""),
            parameters_json_schema=schema.get("parameters", {"type": "object", "properties": {}}),
        )
        for schema in tools
    ]
    return genai_types.Tool(function_declarations=declarations)


def _to_gemini_contents(messages: list[ChatMessage]) -> tuple[str, list[genai_types.Content]]:
    """Split this project's role-tagged messages into a Gemini system instruction + turns.

    Gemini has no "system" or "tool" role: system content is a separate
    config field, and tool results are sent back as ``function_response``
    parts on a "user" turn (Gemini only accepts "user"/"model" as roles).
    """
    system_parts: list[str] = []
    contents: list[genai_types.Content] = []

    for message in messages:
        if message.role == "system":
            if message.content:
                system_parts.append(message.content)
        elif message.role == "user":
            contents.append(genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=message.content or "")]))
        elif message.role == "assistant":
            if message.tool_calls:
                parts = [
                    genai_types.Part.from_function_call(name=call.name, args=call.arguments)
                    for call in message.tool_calls
                ]
            else:
                parts = [genai_types.Part.from_text(text=message.content or "")]
            contents.append(genai_types.Content(role="model", parts=parts))
        elif message.role == "tool":
            if not message.name:
                raise ValueError("Gemini tool-result messages require ChatMessage.name to be set.")
            part = genai_types.Part.from_function_response(
                name=message.name,
                response={"result": message.content or ""},
            )
            contents.append(genai_types.Content(role="user", parts=[part]))

    return "\n\n".join(system_parts), contents


def _parse_arguments(raw: str) -> dict[str, Any]:
    """Parse a tool-call argument blob; malformed JSON degrades to no arguments."""
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _to_openai_message(message: ChatMessage) -> dict[str, Any]:
    if message.role == "tool":
        return {
            "role": "tool",
            "content": message.content or "",
            "tool_call_id": message.tool_call_id or "",
        }
    if message.role == "assistant" and message.tool_calls:
        return {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in message.tool_calls
            ],
        }
    return {"role": message.role, "content": message.content or ""}
