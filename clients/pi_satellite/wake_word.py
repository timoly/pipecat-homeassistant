"""Local wake word detection for the Linux satellite.

Detection is a function from a 10 ms window of captured audio to the
probabilities it produced, which keeps pyopen-wakeword — and the NumPy and
TensorFlow Lite library it brings — out of everything else.

pyopen-wakeword is Rhasspy's openWakeWord, the one Home Assistant's own wake
word add-on runs. It matters here for two reasons: it ships ``okay_nabu``, so a
Pi satellite answers to the same phrase as a Voice PE, and it carries its own
compiled TensorFlow Lite library, so nothing has to be found for the Pi's
architecture. A model trained for a phrase of your own is a file path instead of
a name, and nothing else changes.
"""

from __future__ import annotations

import array
import logging
import sys
from collections import deque
from collections.abc import Callable, Iterable

# 1024 samples, 64 ms at 16 kHz: what pyopen-wakeword's own live command reads
# from stdin and what Home Assistant's Wyoming server streams. Its README shows
# 160 samples, but that path is not the one either of them runs, and smaller
# chunks also shift the extractor's ten-second buffer six times as often.
FRAME_BYTES = 2048
# How often the best recent score is reported, so tuning has something to read
# and a detector that hears nothing at all can be told from one that is idle.
REPORT_SECS = 1.0
# Audio kept for a near miss, which is long enough to hold the phrase and the
# moment before it.
NEAR_MISS_SECS = 3.0
NEAR_MISS_EVERY_SECS = 20.0
BUILTIN_MODELS = ("okay_nabu", "hey_jarvis", "hey_mycroft", "alexa", "hey_rhasspy")

Scorer = Callable[[bytes], Iterable[float]]

logger = logging.getLogger("satellite.wake_word")


def open_wake_word(model: str) -> tuple[Scorer, Callable[[], None]]:
    """Load a wake word, importing pyopen-wakeword only when one is asked for.

    ``model`` is one of ``BUILTIN_MODELS`` or the path to a ``.tflite`` model
    trained for another phrase. Returns the scorer and a way to forget the audio
    it has heard so far.
    """

    from pyopen_wakeword import Model, OpenWakeWord, OpenWakeWordFeatures

    features = OpenWakeWordFeatures.from_builtin()
    if model in BUILTIN_MODELS:
        detector = OpenWakeWord.from_builtin(Model(model))
    else:
        detector = OpenWakeWord.from_model(model)

    def score(frame: bytes) -> Iterable[float]:
        # The features run per frame and the model per embedding, so one frame
        # yields a probability only every eighth call or so.
        for embedding in features.process_streaming(frame):
            yield from detector.process_streaming(embedding)

    def forget() -> None:
        features.reset()
        detector.reset()

    return score, forget


def _peak(frame: bytes) -> int:
    samples = array.array("h")
    samples.frombytes(frame[: len(frame) - len(frame) % 2])
    if sys.byteorder != "little":
        samples.byteswap()
    return max((abs(sample) for sample in samples), default=0)


class WakeWord:
    """Collect captured audio into whole frames and debounce what it hears."""

    def __init__(
        self,
        score: Scorer,
        *,
        threshold: float = 0.5,
        cooldown_secs: float = 2.0,
        name: str = "",
        on_near_miss: Callable[[bytes, float], None] | None = None,
        near_miss_score: float = 0.15,
    ):
        self.score = score
        self.threshold = threshold
        self.cooldown_secs = cooldown_secs
        self.name = name
        self.on_near_miss = on_near_miss
        self.near_miss_score = near_miss_score
        self._buffer = bytearray()
        self._quiet_until = 0.0
        self._fed = False
        self._best = 0.0
        self._best_peak = 0
        self._reported_at = 0.0
        self._near_miss_at = 0.0
        self._recent: deque[bytes] = deque(
            maxlen=max(1, int(NEAR_MISS_SECS * 16000 * 2 / FRAME_BYTES))
        )

    def feed(self, audio: bytes, now: float) -> bool:
        """Return True when the wake word was heard in the audio so far."""

        self._buffer.extend(audio)
        self._fed = True
        while len(self._buffer) >= FRAME_BYTES:
            frame = bytes(self._buffer[:FRAME_BYTES])
            del self._buffer[:FRAME_BYTES]
            if now < self._quiet_until:
                # Inside the cooldown the frame is dropped rather than queued,
                # so a reply's worth of audio cannot pile up behind it.
                continue
            if logger.isEnabledFor(logging.DEBUG):
                self._recent.append(frame)
                self._best_peak = max(self._best_peak, _peak(frame))
            for value in self.score(frame):
                self._best = max(self._best, value)
                if value >= self.threshold:
                    logger.info("Wake word %s: %.2f", self.name or "model", value)
                    self._quiet_until = now + self.cooldown_secs
                    self._clear()
                    return True
            self._report(now)
        return False

    def _report(self, now: float) -> None:
        if now - self._reported_at < REPORT_SECS:
            return
        self._reported_at = now
        logger.debug(
            "Wake word %s: best %.3f at peak %d in the last second (threshold %.2f)",
            self.name or "model",
            self._best,
            self._best_peak,
            self.threshold,
        )
        if (
            self.on_near_miss is not None
            and self.near_miss_score <= self._best < self.threshold
            and now - self._near_miss_at >= NEAR_MISS_EVERY_SECS
        ):
            # Close enough to be worth hearing again: saved so the same audio
            # can be put through the model on its own, which separates a model
            # that cannot hear the phrase from a stream that arrives damaged.
            self._near_miss_at = now
            self.on_near_miss(b"".join(self._recent), self._best)
        self._best = 0.0
        self._best_peak = 0

    def reset(self) -> None:
        """Drop part-built frames, so a conversation leaves nothing behind.

        The model's own history is deliberately left alone. Resetting it sets
        its feature extractor back to filling eight seconds, which makes the
        next call after a conversation carry eight seconds of work — enough to
        stall the loop that drains the microphone, and arecord then loses what
        it recorded meanwhile. The cooldown is what keeps one phrase from waking
        the satellite twice; by the time it is over, the phrase has fallen out of
        the model's 775 ms window anyway.
        """

        if not self._fed:
            return
        self._clear()

    def _clear(self) -> None:
        self._fed = False
        self._best = 0.0
        self._best_peak = 0
        self._buffer.clear()
        self._recent.clear()
