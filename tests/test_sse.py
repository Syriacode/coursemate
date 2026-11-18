import json
from collections.abc import AsyncIterator

import pytest

from coursemate.sse import format_sse, sse_data


async def lines_of(text: str) -> AsyncIterator[str]:
    """A fake network stream: yields the text one line at a time."""
    for line in text.split("\n"):
        yield line


async def collect(text: str) -> list[str]:
    return [data async for data in sse_data(lines_of(text))]


def test_format_sse_matches_the_paper_example():
    assert format_sse("token", {"text": "Hi"}) == 'event: token\ndata: {"text": "Hi"}\n\n'


def test_format_sse_keeps_a_newline_inside_the_text_on_one_data_line():
    # json.dumps writes a newline as the two characters \n, so the event can't break early
    assert format_sse("token", {"text": "a\nb"}) == 'event: token\ndata: {"text": "a\\nb"}\n\n'


@pytest.mark.anyio
async def test_sse_data_reads_back_what_format_sse_writes():
    stream = format_sse("token", {"text": "Hi"}) + format_sse("token", {"text": "a\nb"})
    assert [json.loads(d) for d in await collect(stream)] == [{"text": "Hi"}, {"text": "a\nb"}]


@pytest.mark.anyio
async def test_sse_data_skips_a_keep_alive_comment():
    assert await collect(": keep-alive\n\ndata: x\n\n") == ["x"]


@pytest.mark.anyio
async def test_sse_data_joins_two_data_lines():
    assert await collect("data: a\ndata: b\n\n") == ["a\nb"]
