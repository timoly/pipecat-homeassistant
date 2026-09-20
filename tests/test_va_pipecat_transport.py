"""Runtime contract tests using Pipecat's real frame classes."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ADDON_ROOT = Path(__file__).resolve().parents[1] / "addons" / "pipecat_assist"
sys.path.insert(0, str(ADDON_ROOT))

from app.va_pipecat import (  # noqa: E402
    CHANNELS,
    INPUT_SAMPLE_RATE,
    LIVE_SILENCE_HANGOVER_SECS,
    OUTPUT_PACKET_BYTES,
    OUTPUT_SAMPLE_RATE,
    SatelliteConversationEndFrame,
    SatelliteWakeFrame,
    VaPipecatFrameSerializer,
    websocket_transport_params,
)
from pipecat.frames.frames import (  # noqa: E402
    InputAudioRawFrame,
    InterruptionFrame,
    InterruptionWorkerFrame,
    OutputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
)


def _never_terminal(*_texts: str) -> bool:
    return False


class VaPipecatTransportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.serializer = VaPipecatFrameSerializer(_never_terminal)

    async def test_binary_input_becomes_pcm16_input_frame(self):
        audio = bytes(range(64))
        frame = await self.serializer.deserialize(audio)
        self.assertIsInstance(frame, InputAudioRawFrame)
        self.assertEqual(frame.audio, audio)
        self.assertEqual(frame.sample_rate, INPUT_SAMPLE_RATE)
        self.assertEqual(frame.num_channels, CHANNELS)

    async def test_odd_length_pcm_input_is_rejected(self):
        self.assertIsNone(await self.serializer.deserialize(b"\x00"))

    async def test_output_audio_is_sent_without_reencoding(self):
        audio = bytes(range(128))
        frame = OutputAudioRawFrame(
            audio=audio,
            sample_rate=OUTPUT_SAMPLE_RATE,
            num_channels=CHANNELS,
        )
        self.assertEqual(await self.serializer.serialize(frame), audio)

    async def test_rtvi_phase_is_translated_to_compact_json(self):
        frame = OutputTransportMessageUrgentFrame(
            message={"label": "rtvi-ai", "type": "bot-started-speaking", "data": {}}
        )
        wire = await self.serializer.serialize(frame)
        self.assertEqual(json.loads(wire)["phase"], "speaking")

    async def test_device_interrupt_becomes_pipeline_interruption(self):
        frame = await self.serializer.deserialize('{"type":"interrupt"}')
        self.assertIsInstance(frame, InterruptionWorkerFrame)

    async def test_pipeline_interruption_waits_for_semantic_rtvi_event(self):
        self.assertIsNone(await self.serializer.serialize(InterruptionFrame()))

    def test_websocket_parameters_match_the_device_contract(self):
        params = websocket_transport_params(
            _never_terminal,
            follow_up_ms=12000,
            follow_up_open_delay_ms=150,
            wake_open_delay_ms=50,
            playback_prebuffer_ms=180,
        )
        self.assertEqual(params.audio_in_sample_rate, INPUT_SAMPLE_RATE)
        self.assertEqual(params.audio_out_sample_rate, OUTPUT_SAMPLE_RATE)
        self.assertEqual(params.audio_in_channels, CHANNELS)
        self.assertEqual(params.audio_out_channels, CHANNELS)
        self.assertEqual(params.fixed_audio_packet_size, OUTPUT_PACKET_BYTES)
        hello = json.loads(params.serializer.protocol.hello())
        self.assertEqual(hello["follow_up_ms"], 12000)
        self.assertEqual(hello["follow_up_open_delay_ms"], 150)
        self.assertEqual(hello["wake_open_delay_ms"], 50)
        self.assertEqual(hello["playback_prebuffer_ms"], 180)


class LiveSessionTransportTests(unittest.IsolatedAsyncioTestCase):
    """Satellite transport for models that hold a session per conversation."""

    def setUp(self):
        self.serializer = VaPipecatFrameSerializer(_never_terminal, live_sessions=True)

    @staticmethod
    def _audio(sample: int) -> OutputAudioRawFrame:
        return OutputAudioRawFrame(
            audio=sample.to_bytes(2, "little", signed=True) * (OUTPUT_PACKET_BYTES // 2),
            sample_rate=OUTPUT_SAMPLE_RATE,
            num_channels=CHANNELS,
        )

    async def test_device_messages_become_conversation_frames(self):
        self.assertIsInstance(await self.serializer.deserialize('{"type":"wake"}'), SatelliteWakeFrame)
        stopped = await self.serializer.deserialize('{"type":"stop"}')
        flushed = await self.serializer.deserialize('{"type":"flush"}')
        self.assertIsInstance(stopped, SatelliteConversationEndFrame)
        self.assertIsInstance(flushed, SatelliteConversationEndFrame)
        self.assertIsInstance(await self.serializer.deserialize('{"type":"interrupt"}'), InterruptionWorkerFrame)

    async def test_classic_sessions_keep_their_device_messages(self):
        serializer = VaPipecatFrameSerializer(_never_terminal)

        self.assertIsNone(await serializer.deserialize('{"type":"wake"}'))
        self.assertIsNone(await serializer.deserialize('{"type":"flush"}'))
        self.assertIsInstance(await serializer.deserialize('{"type":"stop"}'), InterruptionWorkerFrame)

    async def test_silence_is_only_sent_right_after_speech(self):
        quiet, speech = self._audio(0), self._audio(4000)

        self.assertIsNone(await self.serializer.serialize(quiet))
        self.assertEqual(await self.serializer.serialize(speech), speech.audio)
        self.assertEqual(await self.serializer.serialize(quiet), quiet.audio)

        self.serializer._last_voiced_audio -= LIVE_SILENCE_HANGOVER_SECS + 0.1
        self.assertIsNone(await self.serializer.serialize(quiet))

    async def test_classic_sessions_send_silence_as_it_comes(self):
        quiet = self._audio(0)

        self.assertEqual(await VaPipecatFrameSerializer(_never_terminal).serialize(quiet), quiet.audio)


if __name__ == "__main__":
    unittest.main()
