from coursemate.mcp_raw import PROTOCOL_VERSION, mcp_request

META_KEYS = {
    "io.modelcontextprotocol/protocolVersion",
    "io.modelcontextprotocol/clientInfo",
    "io.modelcontextprotocol/clientCapabilities",
}


def test_tools_call_matches_the_paper_example():
    msg = mcp_request(3, "tools/call", {"name": "search_notes", "arguments": {"query": "Lab 4"}})
    assert msg["jsonrpc"] == "2.0"
    assert msg["id"] == 3
    assert msg["method"] == "tools/call"
    params = msg["params"]
    assert params["name"] == "search_notes"
    assert params["arguments"] == {"query": "Lab 4"}
    meta = params["_meta"]
    assert set(meta) == META_KEYS
    assert meta["io.modelcontextprotocol/protocolVersion"] == "2026-07-28" == PROTOCOL_VERSION
    assert set(meta["io.modelcontextprotocol/clientInfo"]) == {"name", "version"}
    assert meta["io.modelcontextprotocol/clientCapabilities"] == {}


def test_no_params_gives_only_meta():
    msg = mcp_request(1, "tools/list")
    assert set(msg["params"]) == {"_meta"}
