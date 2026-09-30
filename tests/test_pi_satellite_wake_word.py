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
    from wake_word import FRAME_BYTES, WakeWord  # noqa: E402
except ImportError:  # The add-on image ships the server, not the Pi client.
    WakeWord = None
    FRAME_BYTES = 320

FRAME = b"\x10\x00" * 160  # 10 ms at 16 kHz, one call into the detector
HALF_FRAME = b"\x10\x00" * 80


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

    def test_a_detection_also_clears_the_model_history(self):
        forgotten = []
        detector = WakeWord(FakeScorer(1.0), forget=lambda: forgotten.append(True))

        detector.feed(FRAME, 0.0)

        # Frames dropped during the cooldown would leave the model with a gap.
        self.assertEqual(len(forgotten), 1)

    def test_buffered_audio_does_not_survive_a_conversation(self):
        scorer = FakeScorer(1.0)
        detector = WakeWord(scorer)
        detector.feed(HALF_FRAME, 0.0)

        detector.reset()

        self.assertFalse(detector.feed(HALF_FRAME, 1.0))
        self.assertEqual(scorer.frames, [])

    def test_resetting_an_untouched_detector_costs_nothing(self):
        forgotten = []
        detector = WakeWord(FakeScorer(), forget=lambda: forgotten.append(True))

        detector.reset()
        detector.reset()

        # reset() runs on every captured chunk while a conversation is open.
        self.assertEqual(forgotten, [])


if __name__ == "__main__":
    unittest.main()
