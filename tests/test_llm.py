import json

import httpx
import pytest

from coursemate.llm import LLMError, OllamaClient

MESSAGES = [{"role": "user", "content": "What is the late penalty?"}]


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
