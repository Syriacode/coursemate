import re

import pytest

from coursemate.cache import cache_key

SYSTEM = {"role": "system", "content": "You are CourseMate."}


def asking(question):
    return [SYSTEM, {"role": "user", "content": question}]


# the request in the first line of the guide's table (section 2d)
REQUEST = {
    "provider": "ollama",
    "model": "qwen2.5:3b",
    "messages": asking("When is Lab 4 due?"),
    "temperature": 0.0,
    "max_tokens": 512,
}


def key(**changes):
    return cache_key(**{**REQUEST, **changes})


def test_the_same_request_gives_the_same_64_hex_key():
    first = key()
    again = key(messages=asking("When is Lab 4 due?"))  # built afresh, an hour later

    assert first == again
    assert re.fullmatch(r"[0-9a-f]{64}", first)


def test_key_order_inside_a_message_does_not_matter():
    reordered = [{"content": m["content"], "role": m["role"]} for m in REQUEST["messages"]]

    assert key(messages=reordered) == key()


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"model": "llama3.2:3b"}, id="model"),
        pytest.param({"provider": "openai"}, id="provider"),
        pytest.param({"max_tokens": 50}, id="max_tokens"),
        pytest.param({"temperature": 0.7}, id="temperature"),
        pytest.param({"messages": asking("When is Lab 5 due?")}, id="question"),
        pytest.param({"messages": asking("When is Lab 4 due? ")}, id="extra-space"),
    ],
)
def test_anything_that_can_change_the_reply_changes_the_key(change):
    assert key(**change) != key()
