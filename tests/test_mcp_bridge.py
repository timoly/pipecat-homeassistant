"""MCP bridge tests against a real streamable HTTP MCP server on localhost."""

from __future__ import annotations

import socket
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ADDON_ROOT = Path(__file__).resolve().parents[1] / "addons" / "pipecat_assist"
sys.path.insert(0, str(ADDON_ROOT))

import uvicorn  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402

from app.mcp_bridge import CombinedMCPBridge, list_mcp_call_history  # noqa: E402

DATE_TIME = "2026-09-19 20:07"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class LocalMCPServer:
    """Serve one tool over streamable HTTP, like Home Assistant's MCP server."""

    def __init__(self):
        server = MCPServer("test")

        @server.tool()
        def GetDateTime() -> str:
            """Return the current date and time."""
            return DATE_TIME

        self.port = _free_port()
        self.server = uvicorn.Server(
            uvicorn.Config(
                server.streamable_http_app(),
                host="127.0.0.1",
                port=self.port,
                log_level="warning",
            )
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/mcp"

    def start(self) -> None:
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("Local MCP server did not start")
            time.sleep(0.05)

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)


class CombinedMCPBridgeTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.mcp_server = LocalMCPServer()
        cls.mcp_server.start()

    @classmethod
    def tearDownClass(cls):
        cls.mcp_server.stop()

    async def asyncSetUp(self):
        self.bridge = CombinedMCPBridge(
            [{"id": "homeassistant", "name": "Home Assistant MCP", "url": self.mcp_server.url}]
        )
        await self.bridge.start()

    async def asyncTearDown(self):
        await self.bridge.close()

    async def test_registered_handler_calls_the_mcp_tool(self):
        tools = await self.bridge.tools_schema(cache_enabled=False)
        handlers = {}
        llm = SimpleNamespace(register_function=lambda name, handler: handlers.__setitem__(name, handler))
        await self.bridge.register_tools_schema(tools, llm)

        results = []

        async def result_callback(result, **_kwargs):
            results.append(result)

        await handlers["homeassistant__GetDateTime"](
            SimpleNamespace(arguments={}, result_callback=result_callback)
        )

        self.assertEqual([tool.name for tool in tools.standard_tools], ["homeassistant__GetDateTime"])
        self.assertEqual(results, [DATE_TIME])
        latest = list_mcp_call_history()["calls"][0]
        self.assertEqual((latest["tool"], latest["ok"]), ("GetDateTime", True))

    async def test_recording_client_records_pipecat_tool_calls(self):
        _, bridge = self.bridge.bridges[0]

        response = await bridge.client._call_tool_text("GetDateTime", {})

        self.assertEqual(response, DATE_TIME)
        latest = list_mcp_call_history()["calls"][0]
        self.assertEqual((latest["tool"], latest["ok"]), ("GetDateTime", True))


if __name__ == "__main__":
    unittest.main()
