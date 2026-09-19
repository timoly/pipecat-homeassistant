"""OpenAI Live (gpt-live) runtime wiring tests that need no network access."""

from __future__ import annotations

import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ADDON_ROOT = Path(__file__).resolve().parents[1] / "addons" / "pipecat_assist"
sys.path.insert(0, str(ADDON_ROOT))

from app import main  # noqa: E402
from app.config import (  # noqa: E402
    DEFAULT_OPENAI_TEXT_MODEL,
    ConfigStore,
    FlowConfig,
    PipelineStepConfig,
    default_config_from_environment,
    is_openai_live_model,
    is_openai_speech_to_speech_model,
)
from pipecat.adapters.schemas.function_schema import FunctionSchema  # noqa: E402
from pipecat.adapters.schemas.tools_schema import ToolsSchema  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.services.openai.live.llm import OpenAILiveLLMService  # noqa: E402


def _openai_live_flow(**overrides) -> FlowConfig:
    values = {
        "id": "openai-live",
        "name": "OpenAI Live",
        "pipeline_template": "realtime_home",
        "provider_id": "openai",
        "model": "gpt-live-1",
        "voice": "cedar",
        "steps": [
            PipelineStepConfig(id="transport", kind="transport", label="SmallWebRTC"),
            PipelineStepConfig(
                id="llm",
                kind="llm",
                label="Live model",
                integration_id="openai",
                model="gpt-live-1",
            ),
            PipelineStepConfig(id="tools", kind="tools", label="HA MCP tools", integration_id="ha-mcp"),
            PipelineStepConfig(
                id="output",
                kind="output",
                label="Native audio",
                integration_id="openai",
                voice="cedar",
            ),
        ],
    }
    values.update(overrides)
    return FlowConfig(**values)


class OpenAILiveModelTests(unittest.TestCase):
    def test_live_models_are_openai_speech_to_speech_models(self):
        self.assertTrue(is_openai_live_model("gpt-live-1"))
        self.assertFalse(is_openai_live_model("gpt-realtime-2"))
        self.assertTrue(is_openai_speech_to_speech_model("gpt-live-1"))
        self.assertTrue(is_openai_speech_to_speech_model("gpt-realtime-2"))
        self.assertFalse(is_openai_speech_to_speech_model("gpt-5.4-mini"))
        self.assertFalse(is_openai_speech_to_speech_model("models/gpt-live-1"))

    def test_live_model_is_kept_for_openai_realtime_integration(self):
        config = default_config_from_environment()
        flow = _openai_live_flow()

        self.assertEqual(main._model_name(flow, config.integration("openai"), "openai"), "gpt-live-1")
        self.assertTrue(main._filter_openai_model("gpt-live-1", "realtime"))
        self.assertIn(
            "gpt-live-1",
            [item["id"] for item in main._static_models_for(config.integration("openai"), "realtime")],
        )

    def test_config_store_keeps_live_model_after_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ConfigStore(Path(directory) / "config.json")
            config = store.load()
            config.flows.append(_openai_live_flow())
            store.save(config)

            flow = store.load().selected_flow("openai-live")

        self.assertEqual(flow.model, "gpt-live-1")
        self.assertEqual(flow.model_step().model, "gpt-live-1")

    def test_backend_model_skips_non_openai_and_speech_models(self):
        config = default_config_from_environment()
        config.integration("openai-cloud").default_model = ""
        config.text_model = "gemini-3.6-flash"

        self.assertEqual(
            main._openai_live_backend_model(config, _openai_live_flow(text_model="gemini-3.6-flash")),
            DEFAULT_OPENAI_TEXT_MODEL,
        )
        self.assertEqual(
            main._openai_live_backend_model(config, _openai_live_flow(text_model="gpt-live-1")),
            DEFAULT_OPENAI_TEXT_MODEL,
        )
        self.assertEqual(
            main._openai_live_backend_model(config, _openai_live_flow(text_model="gpt-5.5")),
            "gpt-5.5",
        )


class ESPHomeOpenAILiveGuardTests(unittest.TestCase):
    def test_only_live_flows_are_flagged(self):
        config = default_config_from_environment()
        realtime_flow = _openai_live_flow(model="gpt-realtime-2")
        realtime_flow.model_step().model = "gpt-realtime-2"

        self.assertTrue(main._flow_uses_openai_live(config, _openai_live_flow()))
        self.assertFalse(main._flow_uses_openai_live(config, realtime_flow))
        self.assertFalse(main._flow_uses_openai_live(config, config.flows[0]))

    def test_satellite_is_rejected_before_a_live_session_starts(self):
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        with tempfile.TemporaryDirectory() as directory:
            store = ConfigStore(Path(directory) / "config.json")
            config = store.load()
            config.flows.append(_openai_live_flow())
            store.save(config)
            original_store = main.STORE
            main.STORE = store
            try:
                with patch.object(main, "bot") as bot:
                    client = TestClient(main.app)
                    url = f"/api/assist/esphome?token={config.satellite_shared_secret}&flow_id=openai-live"
                    with client.websocket_connect(url) as websocket:
                        hello = json.loads(websocket.receive_text())
                        error = json.loads(websocket.receive_text())
                        with self.assertRaises(WebSocketDisconnect) as closed:
                            websocket.receive_text()
            finally:
                main.STORE = original_store

        bot.assert_not_called()
        self.assertEqual(hello["type"], "hello")
        self.assertEqual(error["code"], "unsupported_pipeline")
        self.assertFalse(error["recoverable"])
        self.assertIn("gpt-live", base64.b64decode(error["message_b64"]).decode())
        self.assertEqual(closed.exception.code, 4400)


class OpenAILiveSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_session_start_delegates_tools_and_opens_with_greeting(self):
        config = default_config_from_environment()
        flow = _openai_live_flow(reasoning_effort="low")
        llm = main._openai_live_service(
            api_key="test-key",
            model="gpt-live-1",
            flow=flow,
            integration=config.integration("openai"),
            backend_model="gpt-5.4-mini",
        )
        self.assertIsInstance(llm, OpenAILiveLLMService)

        sent = []

        async def capture(event):
            sent.append(event.to_payload())

        llm.send_client_event = capture
        tools = ToolsSchema(
            standard_tools=[
                FunctionSchema(
                    name="HassTurnOn",
                    description="Turn on a device",
                    properties={"name": {"type": "string"}},
                    required=["name"],
                )
            ]
        )
        context = LLMContext([{"role": "developer", "content": flow.greeting}], tools)

        await llm._handle_context(context)

        self.assertEqual(len(sent), 1)
        session = sent[0]["session"]
        self.assertEqual(sent[0]["type"], "session.start")
        self.assertEqual(session["model"], "gpt-live-1")
        self.assertEqual(session["audio"]["output"]["voice"], "cedar")
        self.assertIn("Home Assistant", session["instructions"])
        self.assertIsNone(session.get("input"))
        delegation = session["delegation"]
        self.assertEqual(delegation["type"], "responses")
        self.assertEqual(delegation["responses"]["model"], "gpt-5.4-mini")
        self.assertEqual(delegation["responses"]["reasoning"], {"effort": "low"})
        self.assertIn("HassTurnOn", str(delegation["responses"]["tools"]))
        self.assertEqual(llm._opening_instruction, flow.greeting)


if __name__ == "__main__":
    unittest.main()
