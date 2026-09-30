"""Contract tests for the Linux satellite client, without audio hardware."""

from __future__ import annotations

import base64
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIENT_ROOT = ROOT / "clients" / "pi_satellite"
ADDON_ROOT = ROOT / "addons" / "pipecat_assist"
for path in (CLIENT_ROOT, ADDON_ROOT):
    if path.is_dir():
        sys.path.insert(0, str(path))

from app.va_pipecat_protocol import VaPipecatProtocol  # noqa: E402

try:
    from satellite_protocol import (  # noqa: E402
        DropPlayback,
        Microphone,
        PhaseChanged,
        Play,
        SatelliteClient,
        SendText,
        Transcript,
    )
except ImportError:  # The add-on image ships the server, not the Pi client.
    SatelliteClient = None


def _sent(actions) -> list[str]:
    return [
        json.loads(action.payload)["type"] for action in actions if isinstance(action, SendText)
    ]


def _phase(phase: str, **extra) -> str:
    return json.dumps({"type": "phase", "phase": phase, **extra})


def _transcript(role: str, text: str, final: bool) -> str:
    return json.dumps(
        {
            "type": "transcript",
            "role": role,
            "final": final,
            "text_b64": base64.b64encode(text.encode("utf-8")).decode("ascii"),
        }
    )


@unittest.skipUnless(SatelliteClient, "The Pi satellite client is not available here")
class SatelliteClientTests(unittest.TestCase):
    def setUp(self):
        self.client = SatelliteClient()

    def test_hello_applies_the_timing_the_server_hands_out(self):
        hello = VaPipecatProtocol(lambda *_: False, follow_up_ms=8000).hello()

        self.client.on_server_text(hello, 0.0)

        self.assertEqual(self.client.follow_up_ms, 8000)
        self.assertEqual(self.client.playback_prebuffer_ms, 300)

    def test_wake_opens_the_microphone_and_tells_the_server(self):
        actions = self.client.wake(0.0)

        self.assertEqual(_sent(actions), ["wake"])
        self.assertTrue(self.client.streaming_microphone)
        self.assertIn(Microphone(True, "woken, waiting for speech"), actions)

    def test_a_silent_wake_window_closes_with_flush(self):
        self.client.wake(0.0)

        self.assertEqual(_sent(self.client.tick(6.9)), [])
        actions = self.client.tick(7.1)

        # flush lets the add-on close the paid live session instead of waiting.
        self.assertEqual(_sent(actions), ["flush"])
        self.assertFalse(self.client.streaming_microphone)
        self.assertFalse(self.client.active)

    def test_speech_the_server_heard_cancels_the_window(self):
        self.client.wake(0.0)

        self.client.on_server_text(_phase("listening", heard=True), 1.0)

        self.assertEqual(_sent(self.client.tick(20.0)), [])
        self.assertTrue(self.client.streaming_microphone)

    def test_a_user_transcript_also_counts_as_speech(self):
        self.client.wake(0.0)

        actions = self.client.on_server_text(_transcript("user", "montako astetta", True), 1.0)

        self.assertIn(Transcript("user", "montako astetta", True), actions)
        self.assertEqual(_sent(self.client.tick(20.0)), [])

    def test_a_follow_up_window_reopens_the_microphone_after_its_delay(self):
        client = SatelliteClient(barge_in=False)
        client.wake(0.0)
        client.on_server_text(_phase("thinking"), 1.0)
        client.on_server_text(_phase("speaking"), 2.0)
        self.assertFalse(client.streaming_microphone)

        opened = client.on_server_text(_phase("listening", follow_up=True), 5.0)

        self.assertEqual(opened, [PhaseChanged("listening")])  # the 80 ms delay comes first
        self.assertIn(Microphone(True, "follow-up window"), client.tick(5.2))
        self.assertEqual(_sent(client.tick(30.0)), [])
        self.assertEqual(_sent(client.tick(35.5)), ["flush"])

    def test_a_follow_up_window_lasts_longer_than_a_wake_window(self):
        self.client.wake(0.0)

        self.client.on_server_text(_phase("listening", follow_up=True), 5.0)
        self.client.tick(5.2)

        # Barge-in never closed the microphone, but the longer window still
        # replaces the seven-second watchdog the wake word armed.
        self.assertTrue(self.client.streaming_microphone)
        self.assertEqual(_sent(self.client.tick(30.0)), [])
        self.assertEqual(_sent(self.client.tick(35.5)), ["flush"])

    def test_a_terminal_reply_ends_without_another_message(self):
        self.client.wake(0.0)

        actions = self.client.on_server_text(_phase("thanks", terminal=True), 5.0)

        self.assertEqual(_sent(actions), [])
        self.assertFalse(self.client.active)
        self.assertFalse(self.client.streaming_microphone)

    def test_a_requested_follow_up_survives_a_goodbye(self):
        self.client.wake(0.0)
        self.client.on_server_text(json.dumps({"type": "request_follow_up"}), 4.0)

        self.client.on_server_text(_phase("thanks", terminal=True), 5.0)

        self.assertTrue(self.client.active)
        self.client.tick(5.2)
        self.assertTrue(self.client.streaming_microphone)
        self.assertEqual(_sent(self.client.tick(35.5)), ["flush"])

    def test_barge_in_keeps_the_microphone_open_while_the_assistant_speaks(self):
        self.client.wake(0.0)

        self.client.on_server_text(_phase("speaking"), 3.0)

        self.assertTrue(self.client.streaming_microphone)

    def test_without_echo_cancellation_the_microphone_closes_while_speaking(self):
        client = SatelliteClient(barge_in=False)
        client.wake(0.0)

        actions = client.on_server_text(_phase("speaking"), 3.0)

        self.assertIn(Microphone(False, "assistant is speaking"), actions)
        self.assertFalse(client.streaming_microphone)

    def test_assistant_audio_plays_and_marks_the_reply(self):
        self.client.wake(0.0)

        actions = self.client.on_server_audio(b"\x01\x02" * 8, 3.0)

        self.assertIn(Play(b"\x01\x02" * 8), actions)
        self.assertIn(PhaseChanged("speaking"), actions)

    def test_waking_during_a_reply_interrupts_instead_of_starting_over(self):
        self.client.wake(0.0)
        self.client.on_server_text(_phase("speaking"), 3.0)

        actions = self.client.wake(4.0)

        self.assertEqual(_sent(actions), ["interrupt"])
        self.assertIn(DropPlayback("barge_in"), actions)

    def test_stopping_drops_queued_audio_and_ends_the_conversation(self):
        self.client.wake(0.0)
        self.client.on_server_text(_phase("speaking"), 3.0)

        actions = self.client.request_stop(4.0)

        self.assertEqual(_sent(actions), ["stop"])
        self.assertIn(DropPlayback("stopped"), actions)
        self.assertFalse(self.client.active)

    def test_an_interrupt_the_server_reports_only_drops_audio(self):
        self.client.wake(0.0)
        self.client.on_server_text(_phase("speaking"), 3.0)

        actions = self.client.on_server_text(
            json.dumps({"type": "interrupt", "reason": "barge_in"}), 4.0
        )

        self.assertEqual(actions, [DropPlayback("barge_in")])
        self.assertTrue(self.client.active)

    def test_a_reply_that_ends_without_a_phase_still_closes(self):
        """The bug this guards: audio used to cancel the window outright.

        A live server reports a reply finished and then speaks again, so the
        window was armed and cancelled repeatedly; when the last thing it sent
        was audio, nothing was left to close the conversation and the session
        stayed open, and billed, indefinitely.
        """

        self.client.wake(0.0)
        self.client.on_server_text(_phase("listening", follow_up=True), 5.0)
        self.client.tick(5.2)

        for moment in (6.0, 10.0, 20.0, 30.0):
            self.client.on_server_audio(b"\x01\x02" * 8, moment)
            self.assertEqual(_sent(self.client.tick(moment)), [])

        # Thirty seconds of speech, then the follow-up window from its end.
        self.assertEqual(_sent(self.client.tick(59.0)), [])
        self.assertEqual(_sent(self.client.tick(60.5)), ["flush"])

    def test_a_conversation_cannot_be_kept_open_for_ever(self):
        client = SatelliteClient(max_conversation_secs=20.0)
        client.wake(0.0)

        # A server that keeps reopening the window pushes every other limit
        # forward, so the cap is the one thing it cannot move.
        for moment in range(1, 20):
            client.on_server_text(_phase("listening", follow_up=True), float(moment))
            client.tick(float(moment) + 0.2)

        actions = client.tick(20.5)

        self.assertEqual(_sent(actions), ["stop"])
        self.assertFalse(client.active)

    def test_an_error_is_decoded_rather_than_dropped(self):
        payload = VaPipecatProtocol(lambda *_: False).error_message("boom", "model went away")

        (error,) = self.client.on_server_text(payload, 1.0)

        self.assertEqual(error.code, "boom")
        self.assertEqual(error.message, "model went away")

    def test_garbage_from_the_socket_is_ignored(self):
        self.assertEqual(self.client.on_server_text("not json", 0.0), [])
        self.assertEqual(self.client.on_server_text("[]", 0.0), [])
        self.assertEqual(self.client.on_server_audio(b"", 0.0), [])


@unittest.skipUnless(SatelliteClient, "The Pi satellite client is not available here")
class ServerAcceptsClientMessagesTests(unittest.TestCase):
    """Every control message this client sends must mean something server-side."""

    def setUp(self):
        self.server = VaPipecatProtocol(lambda *_: False)
        self.client = SatelliteClient()

    def _actions_on_server(self, actions) -> list[str]:
        return [
            self.server.client_action(action.payload)
            for action in actions
            if isinstance(action, SendText)
        ]

    def test_wake_flush_stop_and_interrupt_are_all_understood(self):
        self.assertEqual(self._actions_on_server(self.client.wake(0.0)), ["wake"])
        self.assertEqual(self._actions_on_server(self.client.tick(8.0)), ["flush"])

        self.client.wake(10.0)
        self.client.on_server_text(_phase("speaking"), 11.0)
        self.assertEqual(self._actions_on_server(self.client.wake(12.0)), ["interrupt"])
        self.assertEqual(self._actions_on_server(self.client.request_stop(13.0)), ["stop"])


if __name__ == "__main__":
    unittest.main()
