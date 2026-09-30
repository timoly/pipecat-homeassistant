"""The Linux satellite's session wiring, with fake audio and a fake socket.

``satellite.py`` imports ``websockets`` only when it connects, so this runs on a
machine that has neither that library nor ALSA — which is the whole point: the
wiring between the state machine, the microphone and the speaker is exactly
where a missing ``await`` hides, and a Raspberry Pi is a slow place to find one.
"""

from __future__ import annotations

import asyncio
import json
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
