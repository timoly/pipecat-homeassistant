"""OpenAI Live (gpt-live) runtime wiring tests that need no network access."""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

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

from app.va_pipecat import SatelliteConversationEndFrame, SatelliteWakeFrame  # noqa: E402


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

    def test_integration_voice_wins_over_the_voice_saved_with_the_pipeline(self):
        config = default_config_from_environment()
        integration = config.integration("openai")
        integration.default_voice = "ash"
        # The UI clears step voices on save and has no pipeline voice field, so
        # flow.voice keeps the integration voice from when the pipeline was made.
        flow = _openai_live_flow(voice="marin")
        flow.steps[-1].voice = ""

        self.assertEqual(main._openai_voice(flow, integration), "ash")

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


class ESPHomeOpenAILiveTests(unittest.TestCase):
    def test_only_live_flows_are_flagged(self):
        config = default_config_from_environment()
        realtime_flow = _openai_live_flow(model="gpt-realtime-2")
        realtime_flow.model_step().model = "gpt-realtime-2"

        self.assertTrue(main._flow_uses_openai_live(config, _openai_live_flow()))
        self.assertFalse(main._flow_uses_openai_live(config, realtime_flow))
        self.assertFalse(main._flow_uses_openai_live(config, config.flows[0]))

    def test_satellite_transport_uses_live_sessions_only_for_live_flows(self):
        config = default_config_from_environment()

        live = main._transport_params(config, _openai_live_flow())["websocket"]()
        gemini = main._transport_params(config, config.flows[0])["websocket"]()

        self.assertTrue(live.serializer.live_sessions)
        self.assertFalse(gemini.serializer.live_sessions)

    def test_satellite_runs_live_pipelines(self):
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as directory:
            store = ConfigStore(Path(directory) / "config.json")
            config = store.load()
            config.flows.append(_openai_live_flow())
            store.save(config)
            original_store = main.STORE
            main.STORE = store
            try:
                with patch.object(main, "bot", new=AsyncMock()) as bot:
                    client = TestClient(main.app)
                    url = f"/api/assist/esphome?token={config.satellite_shared_secret}&flow_id=openai-live"
                    with client.websocket_connect(url) as websocket:
                        hello = json.loads(websocket.receive_text())
            finally:
                main.STORE = original_store

        self.assertEqual(hello["type"], "hello")
        runner_args = bot.await_args.args[0]
        self.assertEqual(runner_args.body["flow_id"], "openai-live")
        self.assertEqual(runner_args.body["transport"], "va-pipecat")

    def test_finnish_goodbyes_end_the_conversation(self):
        self.assertTrue(main._should_end_conversation("Kiitos, siinä kaikki."))
        self.assertTrue(main._should_end_conversation("", "Näkemiin!"))
        self.assertTrue(main._should_end_conversation("Hei hei"))
        self.assertFalse(main._should_end_conversation("Sytytä työhuoneen spotit", "Selvä, sytytin ne."))


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
                    properties={"name": {"type": "string"}, "floor": {"type": "string"}},
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
        tool = next(item for item in delegation["responses"]["tools"] if item["name"] == "HassTurnOn")
        self.assertEqual(tool["parameters"]["properties"]["name"]["type"], "string")
        self.assertEqual(tool["parameters"]["properties"]["floor"]["type"], ["string", "null"])
        self.assertIn(main.OPENAI_LIVE_TOOL_GUIDANCE, delegation["responses"]["instructions"])
        self.assertEqual(llm._opening_instruction, flow.greeting)


class OpenAILiveCaptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_word_pieces_are_captioned_a_sentence_at_a_time(self):
        from pipecat.frames.frames import TTSStoppedFrame, TTSTextFrame

        config = default_config_from_environment()
        llm = main._openai_live_service(
            api_key="test-key",
            model="gpt-live-1",
            flow=_openai_live_flow(),
            integration=config.integration("openai"),
            backend_model="gpt-5.4-mini",
        )
        pushed = []

        async def capture(_service, frame, direction=None):
            pushed.append(frame)

        # Pieces as gpt-live streamed them in the browser test.
        pieces = ["Moik", "ka!", " Mit", "ä halu", "aisit", " te", "hdä?", " Kerro", " vain"]
        with patch.object(OpenAILiveLLMService, "push_frame", capture):
            for piece in pieces:
                await llm._push_assistant_text(piece)
            await llm.push_frame(TTSStoppedFrame())
            await llm._push_assistant_text(" ")
            await llm.push_frame(TTSStoppedFrame())

        captions = [frame for frame in pushed if isinstance(frame, TTSTextFrame)]
        self.assertEqual(
            [frame.text for frame in captions],
            ["Moikka!", " Mitä haluaisit tehdä?", " Kerro vain"],
        )
        self.assertTrue(all(frame.includes_inter_frame_spaces for frame in captions))
        self.assertEqual("".join(frame.text for frame in captions), "".join(pieces))
        self.assertIsInstance(pushed[-1], TTSStoppedFrame)
        self.assertEqual(llm._caption, "")


class OpenAILiveRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def _run_against_fake_live_api(self, scripts: list[str], messages: list[dict]):
        """Run the service against a local Live API stand-in, one script per connection.

        ``moderate`` stops the reply after the session starts and
        ``moderate_at_startup`` stops it before, as the API did in the browser
        tests; any other script closes the session with that reason, and
        ``serve`` keeps the session open until the service speaks.
        """
        from pipecat.frames.frames import LLMContextFrame
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.worker import PipelineParams, PipelineWorker
        from pipecat.workers.runner import WorkerRunner
        from websockets.asyncio.server import serve

        connections: list[list[dict]] = []
        spoke = asyncio.Event()
        moderation_error = {
            "type": "error",
            "error": {
                "type": "content_filter",
                "code": "content_filter",
                "message": "The generation was stopped due to moderation.",
            },
        }

        async def fake_live_api(websocket):
            received: list[dict] = []
            connections.append(received)
            script = scripts[len(connections) - 1]
            async for raw in websocket:
                event = json.loads(raw)
                received.append(event)
                if event["type"] == "session.start":
                    if script != "moderate_at_startup":
                        started = {"type": "session.started", "session": {"id": f"live_{len(connections)}"}}
                        await websocket.send(json.dumps(started))
                    if script == "serve":
                        continue
                    if script.startswith("moderate"):
                        await websocket.send(json.dumps(moderation_error))
                    reason = "content" if script.startswith("moderate") else script
                    await websocket.send(json.dumps({"type": "session.closed", "reason": reason}))
                    await asyncio.sleep(0.2)
                    await websocket.close()
                    return
                if event["type"] == "session.commentary.append" and script == "serve":
                    spoke.set()

        config = default_config_from_environment()
        llm = main._openai_live_service(
            api_key="test-key",
            model="gpt-live-1",
            flow=_openai_live_flow(),
            integration=config.integration("openai"),
            backend_model="gpt-5.4-mini",
        )

        async with serve(fake_live_api, "127.0.0.1", 0) as server:
            port = next(iter(server.sockets)).getsockname()[1]
            llm.base_url = f"ws://127.0.0.1:{port}"
            worker = PipelineWorker(Pipeline([llm]), params=PipelineParams(), idle_timeout_secs=None)
            runner = WorkerRunner(handle_sigint=False)
            await runner.add_workers(worker)
            running = asyncio.create_task(runner.run())
            await worker.queue_frames([LLMContextFrame(LLMContext(messages))])
            try:
                await asyncio.wait_for(spoke.wait(), timeout=10)
                usable = llm.is_usable
            finally:
                await worker.cancel()
                await asyncio.wait_for(running, timeout=10)

        return connections, usable

    async def test_moderation_stop_restarts_without_the_stopped_conversation(self):
        conversation = [
            {"role": "developer", "content": "Greet the user."},
            {"role": "user", "content": "Paljonko kello on?"},
            {"role": "assistant", "content": "Hetki, tarkistan."},
        ]

        connections, usable = await self._run_against_fake_live_api(
            ["moderate", "moderate_at_startup", "serve"],
            conversation,
        )

        self.assertTrue(usable)
        self.assertEqual(len(connections), 3)
        self.assertEqual(len(connections[0][0]["session"]["input"]), 3)
        for restart in connections[1:]:
            self.assertEqual(restart[0]["type"], "session.start")
            self.assertNotIn("input", restart[0]["session"])
        commentary = [event for event in connections[2] if event["type"] == "session.commentary.append"]
        self.assertIn("cut off", commentary[0]["content"])

    async def test_other_session_close_restarts_with_the_conversation(self):
        connections, usable = await self._run_against_fake_live_api(
            ["expired", "serve"],
            [{"role": "developer", "content": "Greet the user."}],
        )

        self.assertTrue(usable)
        self.assertEqual(len(connections), 2)
        restart = connections[1][0]
        self.assertEqual(
            [item["content"][0]["text"] for item in restart["session"]["input"]],
            ["Greet the user."],
        )
        commentary = [event for event in connections[1] if event["type"] == "session.commentary.append"]
        self.assertIn("cut off", commentary[0]["content"])


class FakeLiveAPI:
    """Speak enough of the Live API to start, feed, and close sessions."""

    def __init__(self, start_delay: float = 0.3):
        self.start_delay = start_delay
        self.connections: list[list[dict]] = []
        self.sockets = []
        self.events: asyncio.Queue = asyncio.Queue()

    async def handler(self, websocket):
        from websockets.exceptions import ConnectionClosed

        received: list[dict] = []
        self.connections.append(received)
        self.sockets.append(websocket)
        index = len(self.connections) - 1
        try:
            async for raw in websocket:
                event = json.loads(raw)
                received.append(event)
                await self.events.put((index, event))
                if event["type"] == "session.start":
                    await asyncio.sleep(self.start_delay)
                    await websocket.send(json.dumps({"type": "session.started", "session": {"id": f"live_{index}"}}))
                elif event["type"] == "session.close":
                    await websocket.send(json.dumps({"type": "session.closed", "reason": "client_close"}))
                    await websocket.close()
                    return
        except ConnectionClosed:
            pass

    async def next_event(self, event_type: str, timeout: float = 5.0) -> tuple[int, dict]:
        while True:
            index, event = await asyncio.wait_for(self.events.get(), timeout)
            if event["type"] == event_type:
                return index, event

    async def send(self, index: int, payload: dict):
        await self.sockets[index].send(json.dumps(payload))


class SatelliteOpenAILiveTests(unittest.IsolatedAsyncioTestCase):
    @asynccontextmanager
    async def _satellite(self, context: LLMContext, api: FakeLiveAPI):
        from pipecat.frames.frames import LLMContextFrame
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.worker import PipelineParams, PipelineWorker
        from pipecat.workers.runner import WorkerRunner
        from websockets.asyncio.server import serve

        config = default_config_from_environment()
        llm = main._openai_live_service(
            api_key="test-key",
            model="gpt-live-1",
            flow=_openai_live_flow(),
            integration=config.integration("openai"),
            backend_model="gpt-5.4-mini",
            satellite_config=config,
        )
        async with serve(api.handler, "127.0.0.1", 0) as server:
            llm.base_url = f"ws://127.0.0.1:{next(iter(server.sockets)).getsockname()[1]}"
            worker = PipelineWorker(Pipeline([llm]), params=PipelineParams(), idle_timeout_secs=None)
            runner = WorkerRunner(handle_sigint=False)
            await runner.add_workers(worker)
            running = asyncio.create_task(runner.run())
            # What run_bot queues when the satellite connects.
            await worker.queue_frames([LLMContextFrame(context)])
            try:
                yield llm, worker
            finally:
                await worker.cancel()
                await asyncio.wait_for(running, timeout=10)

    @staticmethod
    def _speech(frames: int) -> list:
        from pipecat.frames.frames import InputAudioRawFrame

        # 100 ms of 16 kHz PCM each, loud enough not to be silence.
        return [
            InputAudioRawFrame(audio=b"\x00\x10" * 1600, sample_rate=16000, num_channels=1)
            for _ in range(frames)
        ]

    async def test_wake_opens_a_session_that_the_device_closes(self):
        api = FakeLiveAPI()
        context = LLMContext(
            [
                {"role": "user", "content": "Sytytä työhuoneen spotit"},
                {"role": "assistant", "content": "Selvä."},
            ]
        )
        async with self._satellite(context, api) as (llm, worker):
            await asyncio.sleep(0.5)
            self.assertEqual(api.connections, [])

            # The device streams the request while the session is still starting.
            await worker.queue_frames([SatelliteWakeFrame(), *self._speech(5)])
            _, start = await api.next_event("session.start")
            appended = bytearray()
            while len(appended) < 0.8 * 5 * 4800:
                _, event = await api.next_event("session.input_audio.append")
                appended += base64.b64decode(event["audio"])

            await worker.queue_frames([SatelliteConversationEndFrame(reason="microphone closed")])
            await api.next_event("session.close")

            await worker.queue_frames([SatelliteWakeFrame()])
            restarted, restart = await api.next_event("session.start")

        self.assertEqual([item["role"] for item in start["session"]["input"]], ["user", "assistant"])
        self.assertEqual(restarted, 1)
        self.assertEqual(len(restart["session"]["input"]), 2)

    async def test_goodbye_closes_the_session(self):
        api = FakeLiveAPI(start_delay=0)
        async with self._satellite(LLMContext(), api) as (llm, worker):
            await worker.queue_frames([SatelliteWakeFrame()])
            index, _ = await api.next_event("session.start")
            await asyncio.sleep(0.2)
            await api.send(index, {"type": "session.input_transcript.delta", "delta": "Kiitos, siinä kaikki."})
            await asyncio.sleep(0.1)
            await api.send(index, {"type": "session.output_transcript.delta", "delta": "Hei hei!"})

            await api.next_event("session.close")

    async def test_silence_closes_the_session(self):
        api = FakeLiveAPI(start_delay=0)
        async with self._satellite(LLMContext(), api) as (llm, worker):
            llm._idle_timeout_secs = 0.3
            await worker.queue_frames([SatelliteWakeFrame()])
            await api.next_event("session.start")

            await api.next_event("session.close", timeout=4)

    async def test_microphone_audio_opens_a_session_without_a_wake_message(self):
        api = FakeLiveAPI(start_delay=0)
        async with self._satellite(LLMContext(), api) as (llm, worker):
            await worker.queue_frames(self._speech(1))

            await api.next_event("session.start")


if __name__ == "__main__":
    unittest.main()
