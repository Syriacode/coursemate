import copy
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NotRequired, Protocol, TypedDict

import httpx

DEFAULT_REPLY = '{"found": true, "answer": "Fake provider: no model was called.", "citations": [1]}'


class Message(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    # Only on an assistant message that asked for tools (Ollama's format).
    tool_calls: NotRequired[list[dict[str, Any]]]
    # Only on a tool message: which tool produced this result.
    tool_name: NotRequired[str]


@dataclass(frozen=True)
class ToolCall:
    """The model asked to run the tool `name` with these keyword arguments."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMError(Exception):
    """The model could not be reached or returned an error."""


def assistant_message(response: LLMResponse) -> Message:
    """The reply as a history message. Tool calls stay in it, in Ollama's format.

    The model must see its own call before the tool's result, so an agent adds this
    message to the history before it adds the result.
    """
    message: Message = {"role": "assistant", "content": response.text}
    if response.tool_calls:
        message["tool_calls"] = [
            {"function": {"name": call.name, "arguments": dict(call.arguments)}}
            for call in response.tool_calls
        ]
    return message


class LLMClient(Protocol):
    model: str

    def chat(
        self,
        messages: Sequence[Message],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse: ...


class OllamaClient:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        num_ctx: int = 8192,
        http: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.num_ctx = num_ctx
        self._http = http or httpx.Client(base_url=base_url, timeout=timeout)

    def chat(
        self,
        messages: Sequence[Message],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": self.num_ctx},
        }
        if schema is not None:
            body["format"] = schema
        if tools is not None:
            body["tools"] = list(tools)

        start = time.perf_counter()
        try:
            response = self._http.post("/api/chat", json=body)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000

        data = response.json()
        message = data["message"]
        return LLMResponse(
            text=message["content"],
            model=data.get("model", self.model),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_ms=latency_ms,
            tool_calls=_parse_tool_calls(message),
        )


def _parse_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    """Read message["tool_calls"]. Arguments are a JSON object, or a JSON string (OpenAI)."""
    calls = []
    for entry in message.get("tool_calls") or []:
        function = entry["function"]
        name = function["name"]
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise LLMError(f"tool call {name!r} has arguments that are not JSON") from exc
        if not isinstance(arguments, dict):
            raise LLMError(f"tool call {name!r} has arguments that are not a JSON object")
        calls.append(ToolCall(name, arguments))
    return calls


class FakeClient:
    def __init__(self, replies: Sequence[str | list[ToolCall] | Exception] | None = None) -> None:
        self.model = "fake"
        self.calls: list[list[Message]] = []
        self.tools_seen: list[list[dict[str, Any]] | None] = []
        self._replies = None if replies is None else list(replies)

    def chat(
        self,
        messages: Sequence[Message],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        # A deep copy: the caller keeps changing its list, and we want what the model saw.
        self.calls.append(copy.deepcopy(list(messages)))
        self.tools_seen.append(None if tools is None else copy.deepcopy(list(tools)))
        if self._replies is None:
            reply: str | list[ToolCall] | Exception = DEFAULT_REPLY
        elif not self._replies:
            raise AssertionError("FakeClient ran out of replies")
        else:
            reply = self._replies.pop(0)

        if isinstance(reply, Exception):
            raise reply

        # A list reply is "the model asked for these tools": no text, only tool calls.
        text, tool_calls = ("", list(reply)) if isinstance(reply, list) else (reply, [])
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=100,
            output_tokens=20,
            latency_ms=1.0,
            tool_calls=tool_calls,
        )
