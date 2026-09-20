"""OpenAI Live (gpt-live) services for browser calls and ESPHome satellites."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    AggregationType,
    Frame,
    InputAudioRawFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.live import events as live_events
from pipecat.services.openai.live.llm import OpenAILiveLLMService

from app.va_pipecat import SatelliteConversationEndFrame, SatelliteWakeFrame

MAX_RECOVERIES = 3
RECOVERY_WINDOW_SECS = 120
RECOVERY_INSTRUCTION = (
    "The previous voice session ended before your last answer finished. Briefly tell "
    "the user that your answer was cut off and ask them to repeat their request."
)
TOOL_GUIDANCE = (
    "When you call Home Assistant tools, pass only the arguments the request needs. "
    "Leave every other optional argument null instead of guessing a value: do not "
    "add a floor, area, color, or temperature the user did not ask for, and use "
    "device names exactly as GetLiveContext reports them."
)
# Sentence punctuation that the next caption piece has already followed with a space.
SENTENCE_END_RE = re.compile(r"[.!?…]+[\"')\]»”]*(?=\s)")
CAPTION_MAX_CHARS = 200
# Satellite audio kept while a session starts; the rest of the request streams in after it.
SATELLITE_AUDIO_BACKLOG_SECS = 10
# Backstop for a satellite that never ends its conversation.
SATELLITE_MAX_CONVERSATION_SECS = 600
# Trailing microphone audio after a conversation ends must not reopen it.
SATELLITE_REOPEN_GRACE_SECS = 2.0


def nullable_optional_parameters(tool: dict[str, Any]) -> dict[str, Any]:
    """Let a model leave optional tool arguments null instead of inventing values.

    The OpenAI Live backend fills in every tool argument, so without a null
    option it sends empty strings, zeros, or guesses for the ones it does not
    need. The MCP bridge drops the nulls before calling Home Assistant.
    """

    parameters = tool.get("parameters")
    if tool.get("type") != "function" or not isinstance(parameters, dict):
        return tool
    required = set(parameters.get("required") or [])
    properties = {}
    for name, schema in (parameters.get("properties") or {}).items():
        kind = schema.get("type") if isinstance(schema, dict) else None
        if name not in required and isinstance(kind, str) and kind != "null":
            schema = {**schema, "type": [kind, "null"]}
            if isinstance(schema.get("enum"), list):
                schema["enum"] = [*schema["enum"], None]
        properties[name] = schema
    return {**tool, "parameters": {**parameters, "properties": properties}}


class ResilientOpenAILiveLLMService(OpenAILiveLLMService):
    """Start a new session when the Live API ends one mid-conversation.

    The API closes a session on its own, for example after a moderation
    stop. Pipecat then treats the dropped connection as permanent and the
    call goes silent, so open a new session instead. After a moderation
    stop the new session starts without the earlier turns: replaying them
    makes the model resume the stopped answer and trips moderation again.

    Captions are also passed on a sentence at a time: gpt-live streams its
    transcript in word pieces, which the UI and the Lovelace card would
    otherwise merge as if each piece were a word.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._closing_on_request = False
        self._recovery_times: list[float] = []
        self._recovery_instruction: str | None = None
        self._recovery_drops_history = False
        self._starting_session = False
        self._caption = ""

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, TTSTextFrame):
                self._caption += frame.text
                await self._push_caption(final=False)
                return
            if isinstance(frame, TTSStoppedFrame):
                await self._push_caption(final=True)
        await super().push_frame(frame, direction)

    async def _push_caption(self, *, final: bool):
        text = self._caption
        end = len(text) if final else 0
        if not final:
            sentence_ends = [match.end() for match in SENTENCE_END_RE.finditer(text)]
            if sentence_ends:
                end = sentence_ends[-1]
            elif len(text) > CAPTION_MAX_CHARS:
                end = max(text.rfind(" "), 0)
        if not text[:end].strip():
            if final:
                self._caption = ""
            return
        # Pieces carry their own spaces, so the split keeps them for the
        # assistant aggregator, which joins captions as they are.
        caption, self._caption = text[:end], text[end:]
        frame = TTSTextFrame(caption, aggregated_by=AggregationType.SENTENCE)
        frame.includes_inter_frame_spaces = True
        await super().push_frame(frame)

    async def _close_session(self):
        self._closing_on_request = True
        try:
            await super()._close_session()
        finally:
            self._closing_on_request = False

    async def _send_session_config(self):
        self._starting_session = True
        try:
            await super()._send_session_config()
        finally:
            self._starting_session = False

    def _startup_history(self, history: list[live_events.InputItem]) -> list[live_events.InputItem]:
        """Return the conversation a new session starts from."""

        return history

    def _invocation_params(self):
        params = super()._invocation_params()
        params["tools"] = [nullable_optional_parameters(tool) for tool in params["tools"]]
        if self._starting_session:
            history = self._startup_history(params["input"])
            if self._recovery_instruction:
                instruction, self._recovery_instruction = self._recovery_instruction, None
                # The service speaks a trailing developer item when the session opens.
                history = [
                    *([] if self._recovery_drops_history else history),
                    live_events.InputItem(
                        role="developer",
                        content=[live_events.InputTextContent(text=instruction)],
                    ),
                ]
            params["input"] = history
        return params

    async def _handle_evt_session_started(self, evt):
        output = (evt.session.audio or {}).get("output") or {}
        logger.info(
            "OpenAI Live session voice: requested {}, using {}",
            self._settings.voice,
            output.get("voice") or "unreported",
        )
        await super()._handle_evt_session_started(evt)

    async def _handle_evt_error(self, evt):
        if evt.error.code == "content_filter":
            # session.closed follows and decides whether to recover, so a
            # moderation stop must not mark the service unusable first.
            await self.push_error(
                error_msg=f"OpenAI Live moderation stopped the response: {evt.error.message}"
            )
            return
        await super()._handle_evt_error(evt)

    def _last_user_text(self) -> str:
        if self._user_turn.text.strip():
            return self._user_turn.text
        for message in reversed(self._context.get_messages() if self._context else []):
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content") or "")
        return ""

    async def _handle_evt_session_closed(self, evt):
        await super()._handle_evt_session_closed(evt)
        if self._closing_on_request or self._disconnecting:
            return
        moderated = evt.reason == "content"
        if moderated:
            logger.debug(
                "OpenAI Live moderation stopped the reply {!r} after the user said {!r}",
                self._assistant_turn.text[-300:],
                self._last_user_text()[-300:],
            )
        now = time.monotonic()
        self._recovery_times = [
            started for started in self._recovery_times if now - started < RECOVERY_WINDOW_SECS
        ]
        if len(self._recovery_times) >= MAX_RECOVERIES:
            logger.error(
                "OpenAI Live closed {} sessions within {}s; not reconnecting (reason: {})",
                len(self._recovery_times),
                RECOVERY_WINDOW_SECS,
                evt.reason,
            )
            return
        self._recovery_times.append(now)
        logger.warning(
            "OpenAI Live closed the session ({}); starting a new one{}",
            evt.reason,
            " without the earlier turns" if moderated else "",
        )
        self._recovery_instruction = RECOVERY_INSTRUCTION
        self._recovery_drops_history = moderated
        # The server drops the socket next. Treat that as our own disconnect
        # so it does not mark the service unusable before the reset replaces it.
        self._disconnecting = True
        self.create_task(self.reset_conversation(), "openai-live-recovery")


class SatelliteOpenAILiveLLMService(ResilientOpenAILiveLLMService):
    """Hold an OpenAI Live session only while an ESPHome satellite conversation runs.

    A satellite keeps its connection open from boot, and the Live API bills
    every minute a session is open, silence included. So a session opens when
    the device wakes (or starts streaming its microphone without a wake
    message) and closes when the conversation ends: the device's follow-up
    window runs out, the user stops it, the assistant says goodbye, nothing
    has been said for ``idle_timeout_secs``, or it reaches
    ``SATELLITE_MAX_CONVERSATION_SECS``.

    Microphone audio that arrives while the session starts is kept and sent
    once it is ready. A new session starts from the last ``history_messages``
    of the conversation, or from nothing when the previous conversation ended
    more than ``history_reuse_secs`` ago.
    """

    def __init__(
        self,
        *args,
        should_end_conversation: Callable[..., bool],
        idle_timeout_secs: float,
        history_messages: int,
        history_reuse_secs: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._should_end_conversation = should_end_conversation
        self._idle_timeout_secs = idle_timeout_secs
        self._history_messages = history_messages
        self._history_reuse_secs = history_reuse_secs
        self._conversation_active = False
        self._conversation_lock = asyncio.Lock()
        self._conversation_started_at = 0.0
        self._last_activity = 0.0
        self._last_conversation_end = time.monotonic()
        self._history_is_fresh = False
        self._reopen_blocked_until = 0.0
        self._audio_backlog: list[InputAudioRawFrame] = []
        self._audio_backlog_bytes = 0
        self._sending_backlog = False
        self._last_user_turn_text = ""
        self._watchdog_task = None

    async def setup(self, setup):
        await super().setup(setup)
        self._watchdog_task = self.create_task(self._watch_conversation(), "satellite-live-watchdog")

    async def cleanup(self):
        if self._watchdog_task:
            await self.cancel_task(self._watchdog_task)
            self._watchdog_task = None
        await super().cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, SatelliteWakeFrame):
            self._reopen_blocked_until = 0.0
            await self._start_conversation("wake word")
        elif isinstance(frame, SatelliteConversationEndFrame):
            self.create_task(self._end_conversation(frame.reason), "satellite-live-end")
        elif (
            isinstance(frame, InputAudioRawFrame)
            and not self._conversation_active
            and time.monotonic() >= self._reopen_blocked_until
        ):
            await self._start_conversation("microphone audio")
        await super().process_frame(frame, direction)

    async def _connect(self):
        # Setup connects every Live service; a satellite connects per conversation.
        if self._conversation_active:
            await super()._connect()

    async def _handle_context(self, context):
        # Keep the context for the next conversation; a session opens on wake.
        self._context = context
        if self._conversation_active:
            await super()._handle_context(context)

    def _startup_history(self, history):
        if not self._history_is_fresh or self._history_messages <= 0:
            return []
        history = history[-self._history_messages :]
        # The user speaks first on a satellite, so nothing may open the session.
        while history and history[-1].role == "developer":
            history = history[:-1]
        return history

    async def _start_conversation(self, trigger: str):
        async with self._conversation_lock:
            now = time.monotonic()
            if self._conversation_active:
                self._last_activity = now
                return
            self._conversation_active = True
            self._conversation_started_at = now
            self._last_activity = now
            self._history_is_fresh = now - self._last_conversation_end <= self._history_reuse_secs
            logger.info("Satellite conversation started by {}; opening an OpenAI Live session", trigger)
            await self._connect()
            if self._context is not None:
                self._needs_session_config = True
                await self._send_session_config()

    async def _end_conversation(self, reason: str):
        async with self._conversation_lock:
            if not self._conversation_active:
                return
            now = time.monotonic()
            self._conversation_active = False
            self._last_conversation_end = now
            self._reopen_blocked_until = now + SATELLITE_REOPEN_GRACE_SECS
            self._audio_backlog.clear()
            self._audio_backlog_bytes = 0
            logger.info(
                "Satellite conversation ended ({}) after {:.0f}s; closing the OpenAI Live session",
                reason,
                now - self._conversation_started_at,
            )
            await self._close_open_turns()
            await self._close_session()
            await self._disconnect()
            self._needs_session_config = True

    async def _watch_conversation(self):
        while True:
            await asyncio.sleep(1.0)
            if not self._conversation_active:
                continue
            now = time.monotonic()
            if now - self._conversation_started_at > SATELLITE_MAX_CONVERSATION_SECS:
                await self._end_conversation("maximum conversation length")
            elif now - self._last_activity > self._idle_timeout_secs:
                await self._end_conversation("nothing said")

    async def _send_user_audio(self, frame: InputAudioRawFrame):
        if self._conversation_active and (not self._session_started or self._sending_backlog):
            self._audio_backlog.append(frame)
            self._audio_backlog_bytes += len(frame.audio)
            limit = frame.sample_rate * frame.num_channels * 2 * SATELLITE_AUDIO_BACKLOG_SECS
            while self._audio_backlog_bytes > limit:
                self._audio_backlog_bytes -= len(self._audio_backlog.pop(0).audio)
            return
        await super()._send_user_audio(frame)

    async def _handle_evt_session_started(self, evt):
        await super()._handle_evt_session_started(evt)
        # Frames that arrive while the backlog is sent queue behind it, in order.
        self._sending_backlog = True
        try:
            while self._audio_backlog:
                frame = self._audio_backlog.pop(0)
                self._audio_backlog_bytes -= len(frame.audio)
                await super()._send_user_audio(frame)
        finally:
            self._sending_backlog = False

    async def _handle_evt_transcript_delta(self, evt):
        self._last_activity = time.monotonic()
        await super()._handle_evt_transcript_delta(evt)

    async def _end_turn(self, role: str):
        turn = self._user_turn if role == "user" else self._assistant_turn
        text = turn.text if turn.open else ""
        await super()._end_turn(role)
        if not text.strip():
            return
        self._last_activity = time.monotonic()
        if role == "user":
            self._last_user_turn_text = text
        elif self._conversation_active and self._should_end_conversation(self._last_user_turn_text, text):
            self.create_task(self._end_conversation("goodbye"), "satellite-live-end")
