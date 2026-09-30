"""Client half of the va-pipecat satellite protocol, without audio or sockets.

The ESPHome component in ``components/va_pipecat`` is the reference client, and
a Linux satellite has to keep the same promises. Two of them cost money if they
are broken: the add-on opens one OpenAI Live session per conversation, so a
window that closes without speech must be reported with ``flush`` and an
explicit stop with ``stop``, or the session stays open and keeps billing. The
rest is timing the server hands out in its ``hello``.

Keeping the state machine here, free of ALSA and websockets, is what makes it
testable without hardware; ``satellite.py`` executes the actions it returns.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
# The ESPHome satellite gives up seven seconds after a wake word when the
# server never reports hearing anyone, and so does this client.
NO_SPEECH_SECS = 7.0
# A hard cap on one conversation, matching the add-on's own. Every other limit
# here is armed by something the server says, and a server that keeps reporting
# a reply as finished and then speaking again pushes those limits forward
# indefinitely — with a live model, that is billed for as long as it lasts.
MAX_CONVERSATION_SECS = 180.0
DEFAULT_FOLLOW_UP_MS = 30000
DEFAULT_FOLLOW_UP_OPEN_DELAY_MS = 80
DEFAULT_WAKE_OPEN_DELAY_MS = 0
DEFAULT_PLAYBACK_PREBUFFER_MS = 300


@dataclass(frozen=True)
class SendText:
    """Send one control message to the server."""

    payload: str


@dataclass(frozen=True)
class Play:
    """Append assistant audio to the playback jitter buffer."""

    audio: bytes


@dataclass(frozen=True)
class DropPlayback:
    """Throw away audio that is queued but no longer wanted."""

    reason: str


@dataclass(frozen=True)
class Microphone:
    """Start or stop streaming captured audio to the server."""

    streaming: bool
    reason: str = ""


@dataclass(frozen=True)
class PhaseChanged:
    """The server moved the conversation to a new phase."""

    phase: str


@dataclass(frozen=True)
class Transcript:
    """A decoded transcript line, for the console or an LED."""

    role: str
    text: str
    final: bool


@dataclass(frozen=True)
class ServerError:
    """The server reported a pipeline error."""

    code: str
    message: str
    recoverable: bool


Action = (
    SendText | Play | DropPlayback | Microphone | PhaseChanged | Transcript | ServerError
)

_REPLYING_PHASES = {"speaking", "replying"}
_FINISHED_PHASES = {"idle", "thanks"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _decoded(message: dict[str, Any]) -> str:
    """Read a transcript or error payload, which travels base64 encoded."""

    encoded = message.get("text_b64") or message.get("message_b64")
    if isinstance(encoded, str) and encoded:
        try:
            return base64.b64decode(encoded, validate=True).decode("utf-8", "replace")
        except (ValueError, TypeError):
            return ""
    return _text(message.get("text") or message.get("message"))


def _control(message_type: str) -> SendText:
    return SendText(json.dumps({"type": message_type}, separators=(",", ":")))


class SatelliteClient:
    """Track one satellite's conversation and decide what the audio layer does.

    Every method takes the current monotonic time and returns the actions the
    caller should perform, so the clock and the side effects both stay outside.
    """

    def __init__(
        self,
        *,
        barge_in: bool = True,
        no_speech_secs: float = NO_SPEECH_SECS,
        max_conversation_secs: float = MAX_CONVERSATION_SECS,
    ):
        self.barge_in = barge_in
        self.no_speech_secs = no_speech_secs
        self.max_conversation_secs = max_conversation_secs
        self.phase = "idle"
        self.active = False
        self.streaming_microphone = False
        self.follow_up_ms = DEFAULT_FOLLOW_UP_MS
        self.follow_up_open_delay_ms = DEFAULT_FOLLOW_UP_OPEN_DELAY_MS
        self.wake_open_delay_ms = DEFAULT_WAKE_OPEN_DELAY_MS
        self.playback_prebuffer_ms = DEFAULT_PLAYBACK_PREBUFFER_MS
        self._heard = False
        self._mic_open_at: float | None = None
        self._window_secs = 0.0
        self._window_deadline: float | None = None
        self._window_reason = ""
        self._conversation_deadline: float | None = None
        self._request_follow_up = False

    # -- local events ----------------------------------------------------

    def wake(self, now: float) -> list[Action]:
        """Start a conversation, or interrupt the reply that is playing."""

        if self.active and (self.phase in _REPLYING_PHASES or self.phase == "thinking"):
            return [
                _control("interrupt"),
                DropPlayback("barge_in"),
                *self._open_microphone(now, delay_ms=0, window_secs=self.no_speech_secs,
                                       reason="interrupted, waiting for speech"),
            ]

        self.active = True
        self.phase = "listening"
        self._request_follow_up = False
        self._conversation_deadline = now + self.max_conversation_secs
        return [
            _control("wake"),
            PhaseChanged("listening"),
            *self._open_microphone(
                now,
                delay_ms=self.wake_open_delay_ms,
                window_secs=self.no_speech_secs,
                reason="woken, waiting for speech",
            ),
        ]

    def request_stop(self, now: float) -> list[Action]:
        """End the conversation the way the device's stop button does."""

        if not self.active:
            return []
        return [_control("stop"), DropPlayback("stopped"), *self._finish(now, "stopped")]

    def tick(self, now: float) -> list[Action]:
        """Open a delayed microphone and close a window nobody spoke into."""

        actions: list[Action] = []
        if self._mic_open_at is not None and now >= self._mic_open_at:
            self._mic_open_at = None
            self._window_deadline = now + self._window_secs
            if not self.streaming_microphone:
                self.streaming_microphone = True
                actions.append(Microphone(True, self._window_reason))
        if self._window_deadline is not None and now >= self._window_deadline:
            # Live sessions bill by the minute, so a silent window is reported
            # rather than left open: `flush` lets the add-on close the session.
            actions.append(_control("flush"))
            actions.extend(self._finish(now, self._window_reason or "no speech"))
        elif self._conversation_deadline is not None and now >= self._conversation_deadline:
            actions.append(_control("stop"))
            actions.append(DropPlayback("conversation reached its maximum length"))
            actions.extend(self._finish(now, "maximum length"))
        return actions

    # -- server events ---------------------------------------------------

    def on_server_text(self, payload: str, now: float) -> list[Action]:
        """Apply one server message and return what the audio layer must do."""

        try:
            message = json.loads(payload)
        except (TypeError, ValueError):
            return []
        if not isinstance(message, dict):
            return []

        kind = _text(message.get("type")).lower()
        if kind == "hello":
            return self._on_hello(message)
        if kind == "phase":
            return self._on_phase(message, now)
        if kind == "transcript":
            return self._on_transcript(message)
        if kind == "interrupt":
            return self._on_interrupt(message, now)
        if kind == "error":
            return [
                ServerError(
                    _text(message.get("code")) or "pipeline_error",
                    _decoded(message),
                    bool(message.get("recoverable", True)),
                )
            ]
        if kind == "request_follow_up":
            # The model asked a question through its tool and wants the
            # microphone back even if the reply sounds like a goodbye.
            self._request_follow_up = True
        return []

    def on_server_audio(self, audio: bytes, now: float) -> list[Action]:
        """Queue assistant audio, which also proves the reply is under way."""

        if not audio:
            return []
        actions: list[Action] = []
        if self.phase not in _REPLYING_PHASES:
            self.phase = "speaking"
            actions.append(PhaseChanged("speaking"))
            actions.extend(self._microphone_for_reply())
        if self._window_deadline is not None:
            # Audio pushes the window forward rather than cancelling it. A reply
            # must not be cut off mid-sentence, but a window that is cancelled
            # can only be armed again by a phase the server may never send, and
            # the conversation then stays open — and billed — indefinitely.
            self._window_deadline = now + self._window_secs
        actions.append(Play(audio))
        return actions

    def _on_hello(self, message: dict[str, Any]) -> list[Action]:
        for key in (
            "follow_up_ms",
            "follow_up_open_delay_ms",
            "wake_open_delay_ms",
            "playback_prebuffer_ms",
        ):
            value = message.get(key)
            if isinstance(value, int) and value >= 0:
                setattr(self, key, value)
        return []

    def _on_phase(self, message: dict[str, Any], now: float) -> list[Action]:
        phase = _text(message.get("phase")).lower() or "idle"
        terminal = bool(message.get("terminal"))
        follow_up = bool(message.get("follow_up"))
        actions: list[Action] = []

        if message.get("heard"):
            self._note_speech()
        if terminal and self._request_follow_up:
            # The model asked a question, so the reply that reads like a
            # goodbye becomes a follow-up window instead of the end.
            terminal, follow_up = False, True
            if phase in _FINISHED_PHASES:
                phase = "listening"
        if terminal or follow_up:
            self._request_follow_up = False

        if phase != self.phase:
            actions.append(PhaseChanged(phase))
        self.phase = phase

        if phase in _FINISHED_PHASES or terminal:
            actions.extend(self._finish(now, "conversation ended"))
            return actions

        if phase == "listening":
            if follow_up:
                actions.extend(
                    self._open_microphone(
                        now,
                        delay_ms=self.follow_up_open_delay_ms,
                        window_secs=self.follow_up_ms / 1000,
                        reason="follow-up window",
                    )
                )
            elif not self.streaming_microphone and self.active:
                actions.extend(
                    self._open_microphone(
                        now,
                        delay_ms=0,
                        window_secs=self.no_speech_secs,
                        reason="listening again",
                    )
                )
            return actions

        if phase == "thinking":
            self._window_deadline = None
            self._mic_open_at = None
            actions.extend(self._microphone_for_reply())
            return actions

        if phase in _REPLYING_PHASES:
            actions.extend(self._microphone_for_reply())
        return actions

    def _on_transcript(self, message: dict[str, Any]) -> list[Action]:
        role = _text(message.get("role")).lower() or "assistant"
        text = _decoded(message)
        if not text:
            return []
        if role == "user":
            self._note_speech()
        return [Transcript(role, text, bool(message.get("final")))]

    def _on_interrupt(self, message: dict[str, Any], now: float) -> list[Action]:
        reason = _text(message.get("reason")) or "interrupt"
        actions: list[Action] = [DropPlayback(reason)]
        if reason == "stopped":
            actions.extend(self._finish(now, "stopped"))
        return actions

    # -- helpers ---------------------------------------------------------

    def _note_speech(self) -> None:
        """The server heard the user, so no window needs watching."""

        self._heard = True
        self._window_deadline = None

    def _open_microphone(
        self,
        now: float,
        *,
        delay_ms: int,
        window_secs: float,
        reason: str,
    ) -> list[Action]:
        self._heard = False
        self._window_secs = window_secs
        self._window_reason = reason
        self._mic_open_at = now + delay_ms / 1000
        # A zero delay opens in this same tick, which keeps wake latency down.
        return self.tick(now)

    def _microphone_for_reply(self) -> list[Action]:
        """Close the microphone while the assistant speaks, unless echo
        cancellation makes it safe to keep streaming for barge-in."""

        if self.barge_in or not self.streaming_microphone:
            return []
        self.streaming_microphone = False
        return [Microphone(False, "assistant is speaking")]

    def _finish(self, now: float, reason: str) -> list[Action]:
        self.active = False
        self.phase = "idle"
        self._heard = False
        self._mic_open_at = None
        self._window_deadline = None
        self._window_reason = ""
        self._conversation_deadline: float | None = None
        self._request_follow_up = False
        actions: list[Action] = []
        if self.streaming_microphone:
            self.streaming_microphone = False
            actions.append(Microphone(False, reason))
        return actions
