"""The Linux satellite's session wiring, with fake audio and a fake socket.

``satellite.py`` imports ``websockets`` only when it connects, so this runs on a
machine that has neither that library nor ALSA — which is the whole point: the
wiring between the state machine, the microphone and the speaker is exactly
where a missing ``await`` hides, and a Raspberry Pi is a slow place to find one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIENT_ROOT = ROOT / "clients" / "pi_satellite"
if CLIENT_ROOT.is_dir():
    sys.path.insert(0, str(CLIENT_ROOT))

try:
    import satellite  # noqa: E402
except ImportError:  # The add-on image ships the server, not the Pi client.
    satellite = None

CHUNK = b"\x10\x00" * 320  # 20 ms of quiet PCM16 at 16 kHz
SETTLE_SECS = 0.12


class FakeConnection:
    """A websocket that replays messages and records what the client sends."""

    def __init__(self, messages=()):
        self.sent: list[str] = []
        self.audio: list[bytes] = []
        self._messages = list(messages)
        self._closed = asyncio.Event()

    def __aiter__(self):
        return self._replay()

    async def _replay(self):
        for message in self._messages:
            yield message
        await self._closed.wait()

    async def send(self, payload):
        if isinstance(payload, bytes):
            self.audio.append(payload)
        else:
            self.sent.append(payload)

    def close(self):
        self._closed.set()

    def types(self) -> list[str]:
        return [json.loads(payload).get("type") for payload in self.sent]


class FakeCapture:
    """Hand out one chunk every 20 ms, the way arecord paces itself."""

    def __init__(self):
        self.chunk_bytes = len(CHUNK)

    async def read(self) -> bytes:
        await asyncio.sleep(0.02)
        return CHUNK


class FakePlayback:
    def __init__(self):
        self.prebuffer_ms = 300
        self.written: list[bytes] = []
        self.played: list[bytes] = []
        self.resets = 0

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def write(self, audio: bytes) -> None:
        self.written.append(audio)

    async def play_now(self, audio: bytes) -> None:
        self.played.append(audio)

    async def reset(self) -> None:
        self.resets += 1


class FakeWakeWord:
    """Report the wake word once, after the first full window of audio."""

    def __init__(self, *, windows_before_hit: int = 1):
        self.name = "fake"
        self.resets = 0
        self.heard: list[bytes] = []
        self._remaining = windows_before_hit
        self.windows_fed = 0

    def feed(self, audio: bytes, now: float) -> bool:
        self.heard.append(audio)
        if self._remaining <= 0:
            return False
        self._remaining -= 1
        self.windows_fed += 1
        return self._remaining <= 0

    def reset(self) -> None:
        self.resets += 1


def _commands(*queued: str):
    """Play back console commands, then wait like an idle keyboard."""

    pending = list(queued)

    async def next_command() -> str:
        if pending:
            return pending.pop(0)
        await asyncio.Event().wait()
        return ""

    return next_command


class FakeStderr:
    def __init__(self, lines):
        self._lines = list(lines)

    async def readline(self) -> bytes:
        return self._lines.pop(0) if self._lines else b""


class FakeProcess:
    def __init__(self, lines=()):
        self.stderr = FakeStderr(lines)
        self.returncode = 1


@unittest.skipUnless(satellite, "The Pi satellite client is not available here")
class WakeWordStartupTests(unittest.TestCase):
    def test_the_model_runs_once_before_any_audio_is_captured(self):
        frames = []

        def fake_open(model):
            return (lambda frame: frames.append(len(frame)) or iter(())), lambda: None

        original, satellite.open_wake_word = satellite.open_wake_word, fake_open
        self.addCleanup(setattr, satellite, "open_wake_word", original)

        satellite._wake_word(
            satellite.Config(url="ws://example.invalid", wake_word_model="okay_nabu")
        )

        # The extractor fills eight seconds of history on its first call. Left
        # until audio is flowing, that work blocks the loop draining the
        # microphone and arecord loses what it recorded meanwhile.
        self.assertEqual(frames, [satellite.FRAME_BYTES])


@unittest.skipUnless(satellite, "The Pi satellite client is not available here")
class LevelProbeTests(unittest.TestCase):
    """The probe is what tells echo cancellation from the lack of it."""

    def test_the_level_is_reported_with_what_the_speaker_was_doing(self):
        probe = satellite.LevelProbe(interval_secs=0.0)
        loud = (8000).to_bytes(2, "little", signed=True) * 320

        with self.assertLogs(satellite.logger, logging.DEBUG) as captured:
            probe.add(loud, True, 1.0)

        self.assertIn("peak 8000", captured.output[0])
        self.assertIn("assistant speaking", captured.output[0])

    def test_a_gain_that_clips_says_so(self):
        probe = satellite.LevelProbe(interval_secs=0.0, gain=4.0)
        loud = (28469).to_bytes(2, "little", signed=True) * 320

        with self.assertLogs(satellite.logger, logging.DEBUG) as captured:
            probe.add(loud, False, 1.0)

        # Setting a gain by guesswork is how the loudest part of every sentence
        # ends up flattened before the model ever hears it.
        self.assertIn("CLIPPING", captured.output[0])

    def test_nothing_is_measured_unless_debug_is_on(self):
        records = []
        handler = logging.Handler()
        handler.emit = records.append
        satellite.logger.addHandler(handler)
        satellite.logger.setLevel(logging.INFO)
        self.addCleanup(satellite.logger.removeHandler, handler)
        self.addCleanup(satellite.logger.setLevel, logging.NOTSET)

        satellite.LevelProbe(interval_secs=0.0).add(b"\x40\x1f" * 320, True, 1.0)

        # Measuring every sample in Python is not free on a Pi 3.
        self.assertEqual(records, [])


@unittest.skipUnless(satellite, "The Pi satellite client is not available here")
class AlsaMessageTests(unittest.IsolatedAsyncioTestCase):
    """What arecord and aplay say has to be read, or their pipe fills up."""

    async def test_helper_output_is_kept_and_explains_a_failure(self):
        from collections import deque

        seen = deque(maxlen=5)
        process = FakeProcess([b"overrun!!! (at least 12.345 ms long)\n", b"\n"])

        await satellite._log_stderr(process, "arecord", seen)

        self.assertEqual(list(seen), ["overrun!!! (at least 12.345 ms long)"])
        self.assertIn("overrun", satellite._failure(process, seen))

    async def test_the_silence_between_replies_is_not_a_warning(self):
        from collections import deque

        gap = b"underrun!!! (at least 5250.676 ms long)\n"
        dropout = b"underrun!!! (at least 23.400 ms long)\n"
        process = FakeProcess([gap, dropout])

        with self.assertLogs(satellite.logger, logging.DEBUG) as captured:
            await satellite._log_stderr(
                process,
                "aplay",
                deque(maxlen=5),
                expected=satellite._is_silence_between_replies,
            )

        # Playback runs with an open device, so every gap between replies ends
        # in an underrun. Only a short one happened inside a reply.
        self.assertIn("DEBUG", captured.output[0])
        self.assertIn("WARNING", captured.output[1])

    async def test_a_silent_helper_is_explained_by_its_exit_code(self):
        from collections import deque

        seen = deque(maxlen=5)
        process = FakeProcess()

        await satellite._log_stderr(process, "aplay", seen)

        self.assertEqual(satellite._failure(process, seen), "exit code 1")


@unittest.skipUnless(satellite, "The Pi satellite client is not available here")
class SessionTests(unittest.IsolatedAsyncioTestCase):
    def _session(
        self,
        connection,
        *commands: str,
        no_speech_secs: float = 30.0,
        wake_word=None,
        capture_gain: float = 1.0,
        wake_word_gain: float = 1.0,
        wake_chime: bool = True,
    ):
        config = satellite.Config(
            url="ws://example.invalid/api/assist/esphome",
            capture_gain=capture_gain,
            wake_word_gain=wake_word_gain,
            wake_chime=wake_chime,
        )
        session = satellite.Session(
            config,
            connection,
            FakeCapture(),
            commands=_commands(*commands),
            wake_word=wake_word,
        )
        session.playback = FakePlayback()
        session.client.no_speech_secs = no_speech_secs
        return session

    async def _run(self, session, connection, *, settle: float = SETTLE_SECS):
        """Run the session, let it settle, then close the socket from the server."""

        task = asyncio.create_task(session.run())
        await asyncio.sleep(settle)
        connection.close()
        with self.assertRaises(BaseExceptionGroup) as raised:
            await asyncio.wait_for(task, timeout=2)
        return raised.exception

    async def test_the_microphone_stays_quiet_until_a_conversation_starts(self):
        connection = FakeConnection()
        session = self._session(connection)

        await self._run(session, connection)

        self.assertEqual(connection.audio, [])
        self.assertEqual(connection.types(), [])

    async def test_a_wake_command_streams_the_microphone(self):
        connection = FakeConnection()
        session = self._session(connection, "")

        await self._run(session, connection)

        self.assertEqual(connection.types(), ["wake"])
        self.assertGreaterEqual(len(connection.audio), 2)
        self.assertEqual(connection.audio[0], CHUNK)

    async def test_assistant_audio_reaches_the_speaker(self):
        connection = FakeConnection([b"\x01\x02" * 480])
        session = self._session(connection, "")

        await self._run(session, connection)

        self.assertEqual(session.playback.written, [b"\x01\x02" * 480])

    async def test_a_silent_window_reports_itself_so_the_session_can_close(self):
        connection = FakeConnection()
        session = self._session(connection, "", no_speech_secs=0.05)

        await self._run(session, connection)

        # Without the flush the add-on keeps paying for an open live session.
        self.assertEqual(connection.types(), ["wake", "flush"])
        self.assertFalse(session.client.streaming_microphone)

    async def test_stopping_tells_the_server_and_drops_queued_audio(self):
        connection = FakeConnection()
        session = self._session(connection, "", "s")

        await self._run(session, connection)

        self.assertEqual(connection.types(), ["wake", "stop"])
        self.assertEqual(session.playback.resets, 1)

    async def test_the_hello_timing_reaches_the_jitter_buffer(self):
        hello = json.dumps({"type": "hello", "playback_prebuffer_ms": 120})
        connection = FakeConnection([hello])
        session = self._session(connection)

        await self._run(session, connection)

        self.assertEqual(session.client.playback_prebuffer_ms, 120)
        self.assertEqual(session.playback.prebuffer_ms, 120)

    async def test_the_wake_word_starts_a_conversation(self):
        connection = FakeConnection()
        detector = FakeWakeWord()
        session = self._session(connection, wake_word=detector)

        await self._run(session, connection)

        self.assertEqual(connection.types(), ["wake"])
        self.assertGreaterEqual(len(connection.audio), 1)
        # The chime is the only acknowledgement on the path people actually use.
        self.assertEqual(session.playback.played, [satellite.CHIME])

    async def test_the_wake_word_is_not_listened_for_during_a_conversation(self):
        connection = FakeConnection()
        detector = FakeWakeWord(windows_before_hit=2)
        session = self._session(connection, "", wake_word=detector)

        await self._run(session, connection)

        # The keyboard already opened a conversation, so the detector is reset
        # rather than fed: the assistant must not wake itself.
        self.assertEqual(connection.types(), ["wake"])
        self.assertEqual(detector.windows_fed, 0)
        self.assertGreater(detector.resets, 0)

    async def test_each_listener_gets_its_own_gain(self):
        connection = FakeConnection()
        detector = FakeWakeWord()
        session = self._session(
            connection,
            wake_word=detector,
            capture_gain=4.0,
            wake_word_gain=2.0,
        )

        await self._run(session, connection)

        # Neither model is level invariant, and the amount of gain that suits
        # one is not the amount that suits the other.
        self.assertEqual(connection.audio[0], satellite.amplified(CHUNK, 4.0))
        self.assertEqual(detector.heard[0], satellite.amplified(CHUNK, 2.0))
        self.assertNotEqual(connection.audio[0], detector.heard[0])

    async def test_waking_is_acknowledged_out_loud(self):
        connection = FakeConnection()
        session = self._session(connection, "")

        await self._run(session, connection)

        # The add-on stays silent until the user has said something, so this
        # chime is the only sign the satellite is listening.
        self.assertEqual(session.playback.played, [satellite.CHIME])

    async def test_the_chime_can_be_turned_off(self):
        connection = FakeConnection()
        session = self._session(connection, "", wake_chime=False)

        await self._run(session, connection)

        self.assertEqual(session.playback.played, [])
        self.assertEqual(connection.types(), ["wake"])

    async def test_a_closed_socket_ends_the_session_instead_of_spinning(self):
        connection = FakeConnection()
        session = self._session(connection)

        group = await self._run(session, connection, settle=0.02)

        self.assertIsNotNone(group.subgroup(ConnectionError))


if __name__ == "__main__":
    unittest.main()
