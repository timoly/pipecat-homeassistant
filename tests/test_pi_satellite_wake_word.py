"""Wake word buffering and debouncing, with a stand-in for the model.

The scorer is injected, so these run without pyopen-wakeword or NumPy — neither
belongs in a test of when a frame is scored at all, and which frames are dropped.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

CLIENT_ROOT = Path(__file__).resolve().parents[1] / "clients" / "pi_satellite"
if CLIENT_ROOT.is_dir():
    sys.path.insert(0, str(CLIENT_ROOT))

try:
    from wake_word import NEAR_MISS_EVERY_SECS, FRAME_BYTES, WakeWord  # noqa: E402
except ImportError:  # The add-on image ships the server, not the Pi client.
    WakeWord = None
    FRAME_BYTES = 2048
    NEAR_MISS_EVERY_SECS = 20.0

FRAME = b"\x10\x00" * 1024  # 64 ms at 16 kHz, one call into the detector
HALF_FRAME = b"\x10\x00" * 512


class FakeScorer:
    """Return the next prepared probability, and remember every frame seen."""

    def __init__(self, *scores: float):
        self.scores = list(scores)
        self.frames: list[bytes] = []

    def __call__(self, frame: bytes) -> list[float]:
        self.frames.append(frame)
        return [self.scores.pop(0)] if self.scores else []


@unittest.skipUnless(WakeWord, "The Pi satellite client is not available here")
class WakeWordTests(unittest.TestCase):
    def test_a_partial_frame_is_buffered_rather_than_scored(self):
        scorer = FakeScorer(1.0)
        detector = WakeWord(scorer)

        self.assertFalse(detector.feed(HALF_FRAME, 0.0))
        self.assertEqual(scorer.frames, [])

        self.assertTrue(detector.feed(HALF_FRAME, 0.0))
        self.assertEqual(len(scorer.frames), 1)
        self.assertEqual(len(scorer.frames[0]), FRAME_BYTES)

    def test_a_frame_that_yields_nothing_yet_reports_nothing(self):
        detector = WakeWord(FakeScorer(), threshold=0.5)

        self.assertFalse(detector.feed(FRAME, 0.0))

    def test_a_quiet_frame_reports_nothing(self):
        detector = WakeWord(FakeScorer(0.2), threshold=0.5)

        self.assertFalse(detector.feed(FRAME, 0.0))

    def test_the_threshold_is_inclusive(self):
        detector = WakeWord(FakeScorer(0.5), threshold=0.5)

        self.assertTrue(detector.feed(FRAME, 0.0))

    def test_every_frame_in_one_batch_is_scored(self):
        scorer = FakeScorer(0.1, 0.1, 0.9)
        detector = WakeWord(scorer, threshold=0.5)

        self.assertTrue(detector.feed(FRAME * 3, 0.0))
        self.assertEqual(len(scorer.frames), 3)

    def test_the_cooldown_swallows_a_second_hit(self):
        scorer = FakeScorer(1.0, 1.0)
        detector = WakeWord(scorer, cooldown_secs=2.0)

        self.assertTrue(detector.feed(FRAME, 10.0))
        self.assertFalse(detector.feed(FRAME, 11.0))

        # The frame was dropped instead of queued, so the score meant for the
        # next attempt is still waiting.
        self.assertEqual(len(scorer.frames), 1)
        self.assertTrue(detector.feed(FRAME, 12.5))
        self.assertEqual(len(scorer.frames), 2)

    def test_a_phrase_cannot_wake_the_satellite_from_the_models_memory(self):
        """Observed live: woken every seven seconds in a silent room.

        The model keeps ten seconds of audio and judges the last 775 ms of it.
        Nothing is fed while a conversation is open, so its view of the room
        stops there — and the phrase that started the conversation was still in
        its window afterwards.
        """

        class StickyModel:
            """Keeps answering about the phrase it heard until told to forget."""

            def __init__(self):
                self.heard = False

            def score(self, frame: bytes):
                self.heard = self.heard or frame == FRAME
                return [1.0 if self.heard else 0.0]

            def forget(self):
                self.heard = False

        model = StickyModel()
        detector = WakeWord(model.score, forget=model.forget, threshold=0.5)
        self.assertTrue(detector.feed(FRAME, 0.0))

        # A conversation runs, during which nothing is fed, and then ends.
        detector.reset()

        silence = b"\x00\x00" * 1024
        self.assertFalse(detector.feed(silence, 10.0))

    def test_buffered_audio_does_not_survive_a_conversation(self):
        scorer = FakeScorer(1.0)
        detector = WakeWord(scorer)
        detector.feed(HALF_FRAME, 0.0)

        detector.reset()

        self.assertFalse(detector.feed(HALF_FRAME, 1.0))
        self.assertEqual(scorer.frames, [])

    def test_audio_the_model_almost_accepted_is_handed_over(self):
        misses = []
        detector = WakeWord(
            FakeScorer(0.31),
            threshold=0.5,
            on_near_miss=lambda audio, score: misses.append((len(audio), score)),
        )

        with self.assertLogs("satellite.wake_word", "DEBUG"):
            detector.feed(FRAME, 100.0)

        # Near misses are the only cases worth a second opinion: the same audio
        # through the model on its own says whether it was ever detectable.
        self.assertEqual(misses, [(FRAME_BYTES, 0.31)])

    def test_a_score_far_from_the_threshold_is_not_handed_over(self):
        misses = []
        detector = WakeWord(
            FakeScorer(0.02),
            threshold=0.5,
            on_near_miss=lambda audio, score: misses.append(score),
        )

        with self.assertLogs("satellite.wake_word", "DEBUG"):
            detector.feed(FRAME, 100.0)

        self.assertEqual(misses, [])

    def test_near_misses_are_not_saved_on_every_attempt(self):
        misses = []
        detector = WakeWord(
            FakeScorer(0.31, 0.31, 0.31),
            threshold=0.5,
            on_near_miss=lambda audio, score: misses.append(score),
        )

        with self.assertLogs("satellite.wake_word", "DEBUG"):
            detector.feed(FRAME, 100.0)
            detector.feed(FRAME, 102.0)
            detector.feed(FRAME, 100.0 + NEAR_MISS_EVERY_SECS + 1)

        self.assertEqual(len(misses), 2)

    def test_resetting_an_untouched_detector_costs_nothing(self):
        detector = WakeWord(FakeScorer())
        detector.feed(HALF_FRAME, 0.0)

        detector.reset()

        # reset() runs on every captured chunk while a conversation is open, so
        # it must be free once there is nothing left to drop.
        self.assertFalse(detector._fed)
        detector.reset()


if __name__ == "__main__":
    unittest.main()
