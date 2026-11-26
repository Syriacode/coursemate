"""A hand-made MCP client: JSON-RPC messages built and sent by hand."""

import json
import subprocess
import sys
from typing import Any

PROTOCOL_VERSION = "2026-07-28"
REPLY_LIMIT = 300


def mcp_request(
    request_id: int, method: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """One MCP request: a JSON-RPC 2.0 message. Since 2026-07-28 every request says who
    is asking and which protocol version it speaks, in params._meta (no handshake)."""
    meta = {
        "io.modelcontextprotocol/protocolVersion": PROTOCOL_VERSION,
        "io.modelcontextprotocol/clientInfo": {"name": "coursemate-raw", "version": "0.1"},
        "io.modelcontextprotocol/clientCapabilities": {},
    }
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {**(params or {}), "_meta": meta},
    }


class RawClient:
    """MCP over stdio by hand: write one JSON line to the server, read one JSON line back."""

    def __init__(self, command: list[str]) -> None:
        # stderr goes to DEVNULL so the server's logs do not mix with what we print.
        self._process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        self._next_id = 1

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        message = mcp_request(self._next_id, method, params)
        self._next_id += 1
        line = json.dumps(message)
        # Bytes, not text: in text mode Windows would turn "\n" into "\r\n".
        self._process.stdin.write((line + "\n").encode("utf-8"))
        self._process.stdin.flush()
        reply = self._process.stdout.readline()
        if not reply:
            raise RuntimeError("the server closed its stdout")
        text = reply.decode("utf-8").strip()
        print(f">>> {line}")
        print(f"<<< {text[:REPLY_LIMIT]}")
        print()
        return json.loads(text)

    def close(self) -> None:
        self._process.stdin.close()  # end of input tells the server to stop
        self._process.wait()
        self._process.stdout.close()


def main() -> int:
    client = RawClient([sys.executable, "-m", "coursemate.mcp_server", "--fake"])
    try:
        listed = client.request("tools/list")
        print("tools:", [tool["name"] for tool in listed["result"]["tools"]])
        print()

        found = client.request(
            "tools/call", {"name": "search_notes", "arguments": {"query": "When is Lab 4 due?"}}
        )
        print("first text:", repr(found["result"]["content"][0]["text"][:120]))
        print()

        sent = client.request(
            "tools/call", {"name": "send_announcement", "arguments": {"draft_id": 1}}
        )
        print("isError:", sent["result"]["isError"])
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
