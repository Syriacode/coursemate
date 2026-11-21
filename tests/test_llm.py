import json

import httpx
import pytest

from coursemate.llm import (
    FakeClient,
    LLMError,
    LLMResponse,
    OllamaClient,
    ToolCall,
    assistant_message,
)

MESSAGES = [{"role": "user", "content": "What is the late penalty?"}]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "Search the course notes.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]


def ollama_reply(message):
    return httpx.Response(200, json={"model": "qwen2.5:3b", "message": message, "done": True})


def tool_call_message(arguments):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "search_notes", "arguments": arguments}}],
    }


def client_with(handler):
    http = httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    return OllamaClient(model="qwen2.5:3b", http=http)


def test_sends_schema_temperature_and_context_size():
    requests = []
    content = '{"found": true}'

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5:3b",
                "message": {"role": "assistant", "content": content},
                "done": True,
                "prompt_eval_count": 900,
                "eval_count": 60,
            },
        )

    schema = {"type": "object", "properties": {"found": {"type": "boolean"}}}

    response = client_with(handler).chat(MESSAGES, schema=schema)

    assert requests[0].url.path == "/api/chat"
    body = json.loads(requests[0].content)
    assert body["stream"] is False
    assert body["format"] == schema
    assert body["options"] == {"temperature": 0.0, "num_ctx": 8192}
    assert response.text == content
    assert (response.input_tokens, response.output_tokens) == (900, 60)


def test_server_error_becomes_llm_error():
    def handler(request):
        return httpx.Response(500)

    with pytest.raises(LLMError):
        client_with(handler).chat(MESSAGES)


def test_sends_tools():
    requests = []

    def handler(request):
        requests.append(request)
        return ollama_reply({"role": "assistant", "content": "ok"})

    client_with(handler).chat(MESSAGES, tools=TOOLS)

    assert json.loads(requests[0].content)["tools"] == TOOLS


def test_no_tools_key_without_tools():
    requests = []

    def handler(request):
        requests.append(request)
        return ollama_reply({"role": "assistant", "content": "ok"})

    response = client_with(handler).chat(MESSAGES)

    assert "tools" not in json.loads(requests[0].content)
    assert response.tool_calls == []


def test_reads_tool_calls():
    def handler(request):
        return ollama_reply(tool_call_message({"query": "late penalty"}))

    response = client_with(handler).chat(MESSAGES, tools=TOOLS)

    assert response.text == ""
    assert response.tool_calls == [ToolCall("search_notes", {"query": "late penalty"})]


def test_json_string_arguments_are_parsed():
    def handler(request):
        return ollama_reply(tool_call_message('{"query": "late penalty"}'))

    response = client_with(handler).chat(MESSAGES, tools=TOOLS)

    assert response.tool_calls == [ToolCall("search_notes", {"query": "late penalty"})]


@pytest.mark.parametrize("arguments", ["{not json", "[1, 2]"])
def test_broken_arguments_raise_llm_error(arguments):
    def handler(request):
        return ollama_reply(tool_call_message(arguments))

    with pytest.raises(LLMError):
        client_with(handler).chat(MESSAGES, tools=TOOLS)


def test_assistant_message_keeps_tool_calls_in_ollama_format():
    asked = LLMResponse(
        text="",
        model="m",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1.0,
        tool_calls=[ToolCall("search_notes", {"query": "late penalty"})],
    )
    plain = LLMResponse(text="hi", model="m", input_tokens=1, output_tokens=1, latency_ms=1.0)

    assert assistant_message(asked) == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "search_notes", "arguments": {"query": "late penalty"}}}
        ],
    }
    assert assistant_message(plain) == {"role": "assistant", "content": "hi"}


def test_fake_client_copies_messages_and_records_tools():
    asked = [ToolCall("search_notes", {"query": "late penalty"})]
    fake = FakeClient([asked, "done"])
    history = [{"role": "user", "content": "What is the late penalty?"}]

    first = fake.chat(history, tools=TOOLS)
    history.append({"role": "assistant", "content": "changed after the call"})
    second = fake.chat(history)

    assert (first.text, first.tool_calls) == ("", asked)
    assert (second.text, second.tool_calls) == ("done", [])
    assert len(fake.calls[0]) == 1  # the later append did not leak into the stored call
    assert len(fake.calls[1]) == 2
    assert fake.tools_seen == [TOOLS, None]
