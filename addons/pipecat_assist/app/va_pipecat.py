"""Pipecat transport adapter for the ESPHome ``va_pipecat`` component."""

from __future__ import annotations

import array
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger
from pipecat.audio.utils import is_silence
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    InterruptionWorkerFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    SystemFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

from app.va_pipecat_protocol import VaPipecatProtocol

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHANNELS = 1
OUTPUT_PACKET_BYTES = 1920  # 40 ms of mono PCM16 at 24 kHz
# Silence kept after speech in a live session, so pauses between words survive.
LIVE_SILENCE_HANGOVER_SECS = 0.5


@dataclass
class SatelliteWakeFrame(SystemFrame):
    """The satellite heard its wake word and is streaming the user's request."""


@dataclass
class SatelliteConversationEndFrame(SystemFrame):
    """The satellite ended the conversation, so a per-conversation session can close.

    Parameters:
        reason: Why the conversation ended.
    """

    reason: str = ""


class VaPipecatFrameSerializer(FrameSerializer):
    """Serialize raw PCM and compact conversation events for ESPHome.

    With ``live_sessions`` the model holds a session per conversation and
    streams audio continuously, silence included: device wake, stop and flush
    messages become conversation frames, and silence is not sent to the
    device, which waits for its speaker to drain before opening a follow-up.
    """

    def __init__(
        self,
        should_end_conversation: Callable[..., bool],
        *,
        follow_up_ms: int = 30000,
        follow_up_open_delay_ms: int = 80,
        wake_open_delay_ms: int = 0,
        playback_prebuffer_ms: int = 300,
        live_sessions: bool = False,
    ):
        super().__init__(
            params=FrameSerializer.InputParams(ignore_rtvi_messages=False),
            name="VaPipecatFrameSerializer",
        )
        self.protocol = VaPipecatProtocol(
            should_end_conversation=should_end_conversation,
            input_sample_rate=INPUT_SAMPLE_RATE,
            output_sample_rate=OUTPUT_SAMPLE_RATE,
            channels=CHANNELS,
            follow_up_ms=follow_up_ms,
            follow_up_open_delay_ms=follow_up_open_delay_ms,
            wake_open_delay_ms=wake_open_delay_ms,
            playback_prebuffer_ms=playback_prebuffer_ms,
            live_sessions=live_sessions,
        )
        self.live_sessions = live_sessions
        self._last_voiced_audio = float("-inf")
        self._reset_audio_probe()

    def _audible(self, audio: bytes) -> bool:
        now = time.monotonic()
        if not is_silence(audio):
            self._last_voiced_audio = now
            return True
        return now - self._last_voiced_audio <= LIVE_SILENCE_HANGOVER_SECS

    def _reset_audio_probe(self) -> None:
        self._probe_samples = 0
        self._probe_nonzero = 0
        self._probe_peak = 0
        self._probe_energy = 0
        self._probe_window = 0

    def _probe_input_audio(self, data: bytes) -> None:
        if self._probe_window >= 8:
            return

        samples = array.array("h")
        samples.frombytes(data)
        if sys.byteorder != "little":
            samples.byteswap()

        self._probe_samples += len(samples)
        for sample in samples:
            absolute = abs(sample)
            self._probe_peak = max(self._probe_peak, absolute)
            self._probe_nonzero += int(sample != 0)
            self._probe_energy += sample * sample

        if self._probe_samples < INPUT_SAMPLE_RATE:
            return

        rms = math.sqrt(self._probe_energy / self._probe_samples)
        nonzero_pct = 100.0 * self._probe_nonzero / self._probe_samples
        logger.debug(
            "ESPHome audio ingress window={}: samples={} rate={}Hz peak={} rms={:.1f} nonzero={:.1f}%",
            self._probe_window + 1,
            self._probe_samples,
            INPUT_SAMPLE_RATE,
            self._probe_peak,
            rms,
            nonzero_pct,
        )
        self._probe_window += 1
        self._probe_samples = 0
        self._probe_nonzero = 0
        self._probe_peak = 0
        self._probe_energy = 0

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, OutputAudioRawFrame):
            if self.live_sessions and not self._audible(frame.audio):
                return None
            return frame.audio

        if isinstance(frame, InterruptionFrame):
            # The RTVI observer emits bot-interrupted for this same pipeline
            # event and has enough context to distinguish barge-in from an
            # explicit stop. Serializing both produced duplicate interrupts.
            return None

        if isinstance(frame, (EndFrame, CancelFrame)):
            return self.protocol.phase_message("idle", terminal=True)

        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            return self.protocol.on_rtvi_message(frame.message)

        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, bytes):
            if not data or len(data) % 2:
                return None
            self._probe_input_audio(data)
            return InputAudioRawFrame(
                audio=data,
                sample_rate=INPUT_SAMPLE_RATE,
                num_channels=CHANNELS,
            )

        action = self.protocol.client_action(data)
        if action == "wake":
            self._reset_audio_probe()
            if self.live_sessions:
                return SatelliteWakeFrame()
        if self.live_sessions and action in {"stop", "flush"}:
            # flush means the follow-up window or the no-speech watchdog closed the microphone.
            reason = "stopped by the user" if action == "stop" else "microphone closed"
            return SatelliteConversationEndFrame(reason=reason)
        if action in {"interrupt", "stop"}:
            return InterruptionWorkerFrame()
        return None


def websocket_transport_params(
    should_end_conversation: Callable[..., bool],
    *,
    follow_up_ms: int = 30000,
    follow_up_open_delay_ms: int = 80,
    wake_open_delay_ms: int = 0,
    playback_prebuffer_ms: int = 300,
    live_sessions: bool = False,
) -> FastAPIWebsocketParams:
    """Build the fixed PCM transport contract used by ESPHome satellites."""

    return FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_in_sample_rate=INPUT_SAMPLE_RATE,
        audio_in_channels=CHANNELS,
        audio_out_enabled=True,
        audio_out_sample_rate=OUTPUT_SAMPLE_RATE,
        audio_out_channels=CHANNELS,
        audio_out_10ms_chunks=4,
        audio_out_auto_silence=False,
        audio_out_end_silence_secs=0,
        serializer=VaPipecatFrameSerializer(
            should_end_conversation,
            follow_up_ms=follow_up_ms,
            follow_up_open_delay_ms=follow_up_open_delay_ms,
            wake_open_delay_ms=wake_open_delay_ms,
            playback_prebuffer_ms=playback_prebuffer_ms,
            live_sessions=live_sessions,
        ),
        fixed_audio_packet_size=OUTPUT_PACKET_BYTES,
        allowed_origins=[],
    )
