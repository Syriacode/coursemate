"""A hand-made MCP client: JSON-RPC messages built and sent by hand."""

from typing import Any

PROTOCOL_VERSION = "2026-07-28"


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
