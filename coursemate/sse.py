import json
from collections.abc import AsyncIterator
from typing import Any


def format_sse(event: str, data: dict[str, Any]) -> str:
    """One server-sent event: a name line, a JSON data line, then a blank line."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def sse_data(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    """Yield the data of each event in an SSE stream.

    An event ends at a blank line. Several data: lines in one event are joined with a newline.
    event:, id:, retry: and ":" comment lines are skipped.
    """
    buffer: list[str] = []
    async for raw in lines:
        line = raw.rstrip("\r\n")
        if line == "":  # a blank line means the event is finished
            if buffer:
                yield "\n".join(buffer)
                buffer = []
        elif line.startswith("data:"):
            value = line[len("data:") :]
            # the SSE spec says to drop one leading space after the colon
            buffer.append(value.removeprefix(" "))
    if buffer:  # the stream ended without a final blank line
        yield "\n".join(buffer)
