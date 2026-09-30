#!/usr/bin/env python3
"""A Linux voice satellite for the Pipecat Assist add-on.

Runs on a Raspberry Pi with a USB speakerphone and speaks the same ``va-pipecat``
protocol as the ESPHome component: raw PCM16 up at 16 kHz, down at 24 kHz, with
compact JSON for control. The audio itself goes through ``arecord`` and
``aplay``, so talking to the add-on needs nothing but ``websockets``; a wake word
is optional and brings openWakeWord with it.

    python3 satellite.py --config ~/.config/pipecat-satellite.toml

Press Enter to start a conversation, ``s`` to stop one, ``q`` to quit. A wake
word can replace the keyboard later; the protocol does not care which one woke
the satellite.
"""

from __future__ import annotations

import argparse
import array
import asyncio
import contextlib
import logging
import math
import os
import sys
import time
import tomllib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wake_word import WakeWord, open_wake_word  # noqa: E402

from satellite_protocol import (  # noqa: E402
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    DropPlayback,
    Microphone,
    PhaseChanged,
    Play,
    SatelliteClient,
    SendText,
    ServerError,
    Transcript,
)

CHUNK_MS = 20
TICK_SECS = 0.025
# Assistant audio arriving after a longer gap starts a new reply, which is
# primed again so a late first packet cannot stutter the opening word.
PLAYBACK_GAP_SECS = 0.4
RECONNECT_MIN_SECS = 1.0
RECONNECT_MAX_SECS = 30.0
# A satellite says nothing when it wakes: the add-on suppresses the greeting
# for a live session, because the user spoke first and every second of an
# answer nobody asked for is billed. A local chime is the acknowledgement, the
# way a Voice PE lights its ring, and it costs nothing.
CHIME_TONES = (880, 1175)
CHIME_TONE_MS = 70
CHIME_FADE_MS = 5
CHIME_LEVEL = 0.2
# Trailing silence, so the tone is not left waiting in aplay's read buffer for
# audio that only arrives when the assistant answers.
CHIME_TAIL_MS = 60
# aplay hands ALSA one period at a time, and a default period is longer than the
# chime. A short one plays it at once and costs a Pi 3 nothing measurable.
PLAYBACK_PERIOD_US = 20000
PLAYBACK_BUFFER_US = 400000
# Left alone, aplay asks ALSA to start only once the whole buffer is full, so a
# sound shorter than the buffer waits for unrelated audio to push it out — or
# for the end of the stream, which is why the same bytes play from a pipe that
# closes. This starts playback as soon as there is anything to play; the jitter
# buffer in this client, not the one in ALSA, is what covers a late packet.
PLAYBACK_START_DELAY_US = 1

logger = logging.getLogger("satellite")


@dataclass
class Config:
    """Everything the satellite needs to reach the add-on and the hardware."""

    url: str
    capture_device: str = "default"
    playback_device: str = "default"
    capture_gain: float = 1.0
    barge_in: bool = True
    wake_chime: bool = True
    wake_word_model: str = ""
    wake_word_gain: float = 1.0
    wake_word_threshold: float = 0.5
    wake_word_cooldown_secs: float = 2.0
    log_level: str = "INFO"

    @classmethod
    def load(cls, path: Path | None) -> Config:
        values: dict[str, object] = {}
        if path is not None:
            with path.open("rb") as handle:
                values = tomllib.load(handle)
        url = str(os.environ.get("PIPECAT_SATELLITE_URL") or values.get("url") or "")
        if not url:
            raise SystemExit(
                "No websocket URL. Copy config.example.toml, fill in the URL "
                "from the add-on's Runtime tab, or set PIPECAT_SATELLITE_URL."
            )
        return cls(
            url=url,
            capture_device=str(values.get("capture_device", "default")),
            playback_device=str(values.get("playback_device", "default")),
            capture_gain=float(values.get("capture_gain", 1.0)),
            barge_in=bool(values.get("barge_in", True)),
            wake_chime=bool(values.get("wake_chime", True)),
            wake_word_model=str(values.get("wake_word_model", "")),
            wake_word_gain=float(values.get("wake_word_gain", 1.0)),
            wake_word_threshold=float(values.get("wake_word_threshold", 0.5)),
            wake_word_cooldown_secs=float(values.get("wake_word_cooldown_secs", 2.0)),
            log_level=str(values.get("log_level", "INFO")).upper(),
        )


def amplified(audio: bytes, gain: float) -> bytes:
    """Scale PCM16 samples, clipping rather than wrapping around.

    A USB speakerphone with its own mixer already at maximum can still be far
    too quiet, and a loud sample multiplied without a bound would wrap into the
    opposite polarity and sound like a click.
    """

    if gain == 1.0 or not audio:
        return audio
    samples = array.array("h")
    samples.frombytes(audio[: len(audio) - len(audio) % 2])
    if sys.byteorder != "little":
        samples.byteswap()
    for index, sample in enumerate(samples):
        samples[index] = max(-32768, min(32767, int(sample * gain)))
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


def chime_audio() -> bytes:
    """Build the two-tone blip played when a conversation opens."""

    samples = array.array("h")
    fade = OUTPUT_SAMPLE_RATE * CHIME_FADE_MS // 1000
    for frequency in CHIME_TONES:
        length = OUTPUT_SAMPLE_RATE * CHIME_TONE_MS // 1000
        for index in range(length):
            # Without the fades each tone would start and end on a step, which
            # a speaker reproduces as a click.
            envelope = min(1.0, index / fade, (length - index) / fade)
            value = math.sin(2 * math.pi * frequency * index / OUTPUT_SAMPLE_RATE)
            samples.append(int(value * envelope * CHIME_LEVEL * 32767))
    samples.extend([0] * (OUTPUT_SAMPLE_RATE * CHIME_TAIL_MS // 1000))
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


CHIME = chime_audio()


class AlsaCapture:
    """Keep one ``arecord`` running for the whole session.

    Opening the device per conversation costs a few hundred milliseconds and
    swallows the first word — USB speakerphones ramp their own processing when
    the stream starts — so capture runs continuously and unwanted audio is
    dropped instead.

    Audio comes out as the device recorded it. Gain belongs to whoever wants it:
    the model on the other end of the socket does, and the wake word does not.
    """

    def __init__(self, device: str):
        self.device = device
        self.chunk_bytes = INPUT_SAMPLE_RATE * 2 * CHUNK_MS // 1000
        self._process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        self._process = await asyncio.create_subprocess_exec(
            "arecord",
            "-q",
            "-D",
            self.device,
            "-t",
            "raw",
            "-f",
            "S16_LE",
            "-r",
            str(INPUT_SAMPLE_RATE),
            "-c",
            "1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        logger.info("Capturing from %s at %d Hz", self.device, INPUT_SAMPLE_RATE)

    async def read(self) -> bytes:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("Capture has not been started")
        try:
            return await process.stdout.readexactly(self.chunk_bytes)
        except asyncio.IncompleteReadError as error:
            # A wrong device name is the common case here, and arecord explains
            # it on stderr — the reason belongs in the log, not just the exit code.
            raise RuntimeError(f"arecord stopped: {await _failure(process)}") from error

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def ensure_running(self) -> None:
        """Bring capture back after the USB device was unplugged or reset."""

        if self.running:
            return
        if self._process is not None:
            logger.warning("Capture stopped: %s", await _failure(self._process))
            await self.stop()
        await self.start()

    async def stop(self) -> None:
        await _terminate(self._process)
        self._process = None


class AlsaPlayback:
    """Feed assistant audio to ``aplay`` through a small jitter buffer."""

    def __init__(self, device: str, *, prebuffer_ms: int = 300):
        self.device = device
        self.prebuffer_ms = prebuffer_ms
        self._process: asyncio.subprocess.Process | None = None
        self._buffer = bytearray()
        self._primed = False
        self._prime_started = 0.0
        self._last_write = 0.0

    async def start(self) -> None:
        self._buffer.clear()
        self._primed = False
        self._last_write = 0.0
        self._process = await asyncio.create_subprocess_exec(
            "aplay",
            "-q",
            "-D",
            self.device,
            "-t",
            "raw",
            "-f",
            "S16_LE",
            "-r",
            str(OUTPUT_SAMPLE_RATE),
            "-c",
            "1",
            f"--period-time={PLAYBACK_PERIOD_US}",
            f"--buffer-time={PLAYBACK_BUFFER_US}",
            f"--start-delay={PLAYBACK_START_DELAY_US}",
            stdin=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        logger.info("Playing to %s at %d Hz", self.device, OUTPUT_SAMPLE_RATE)

    async def write(self, audio: bytes) -> None:
        now = time.monotonic()
        if now - self._last_write > PLAYBACK_GAP_SECS:
            self._buffer.clear()
            self._primed = False
            self._prime_started = now
        self._last_write = now

        if not self._primed:
            self._buffer.extend(audio)
            target = OUTPUT_SAMPLE_RATE * 2 * self.prebuffer_ms // 1000
            waited = now - self._prime_started
            if len(self._buffer) < target and waited < self.prebuffer_ms / 1000:
                return
            audio = bytes(self._buffer)
            self._buffer.clear()
            self._primed = True

        process = self._process
        if process is None or process.stdin is None:
            return
        try:
            process.stdin.write(audio)
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            logger.warning("Playback stopped: %s", await _failure(process))
            await self.reset()

    async def play_now(self, audio: bytes) -> None:
        """Play a local sound without waiting for the jitter buffer to fill."""

        process = self._process
        if process is None or process.stdin is None:
            return
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            process.stdin.write(audio)
            await process.stdin.drain()

    async def reset(self) -> None:
        """Drop audio that is queued in ALSA as well as in this buffer.

        There is no way to flush a pipe that ``aplay`` has already read, so the
        process is replaced. That costs tens of milliseconds and is what makes
        an interruption sound immediate.
        """

        await self.stop()
        await self.start()

    async def stop(self) -> None:
        await _terminate(self._process)
        self._process = None


async def _failure(process: asyncio.subprocess.Process | None) -> str:
    """Explain why an ALSA helper stopped, in its own words when it has any."""

    if process is None:
        return "no process"
    if process.stderr is not None:
        with contextlib.suppress(Exception):
            # Bounded, because a helper that closed its pipes without exiting
            # would otherwise keep this read waiting for an EOF that never comes.
            raw = await asyncio.wait_for(process.stderr.read(), timeout=1)
            message = raw.decode("utf-8", "replace").strip()
            if message:
                return message
    return f"exit code {process.returncode}"


async def _terminate(process: asyncio.subprocess.Process | None) -> None:
    if process is None or process.returncode is not None:
        return
    if process.stdin is not None:
        with contextlib.suppress(Exception):
            process.stdin.close()
    with contextlib.suppress(ProcessLookupError):
        process.kill()
    with contextlib.suppress(Exception):
        await process.wait()


class Session:
    """One websocket connection, its audio devices and the state machine."""

    def __init__(
        self,
        config: Config,
        connection,
        capture: AlsaCapture,
        *,
        commands: Callable[[], Awaitable[str]] | None = None,
        wake_word: WakeWord | None = None,
    ):
        self.config = config
        self.connection = connection
        self.capture = capture
        self.wake_word = wake_word
        self.client = SatelliteClient(barge_in=config.barge_in)
        self.playback = AlsaPlayback(config.playback_device)
        # Conversations start from the console today and from a wake word or a
        # button later; the protocol cannot tell the difference.
        self._next_command = commands or ConsoleCommands()
        self._send_lock = asyncio.Lock()

    async def run(self) -> None:
        self.playback.prebuffer_ms = self.client.playback_prebuffer_ms
        await self.playback.start()
        try:
            async with asyncio.TaskGroup() as tasks:
                tasks.create_task(self._receive())
                tasks.create_task(self._send_microphone())
                tasks.create_task(self._tick())
                tasks.create_task(self._read_commands())
        finally:
            await self.playback.stop()

    async def _apply(self, actions) -> None:
        for action in actions:
            if isinstance(action, SendText):
                async with self._send_lock:
                    await self.connection.send(action.payload)
            elif isinstance(action, Play):
                await self.playback.write(action.audio)
            elif isinstance(action, DropPlayback):
                logger.info("Dropping queued audio (%s)", action.reason)
                await self.playback.reset()
            elif isinstance(action, Microphone):
                logger.info(
                    "Microphone %s (%s)",
                    "open" if action.streaming else "closed",
                    action.reason,
                )
            elif isinstance(action, PhaseChanged):
                logger.info("Phase: %s", action.phase)
            elif isinstance(action, Transcript):
                marker = "" if action.final else "…"
                print(f"{action.role}: {action.text}{marker}", flush=True)
            elif isinstance(action, ServerError):
                logger.error("Server error %s: %s", action.code, action.message)

    async def _wake(self, now: float) -> None:
        """Start or interrupt a conversation, and say so out loud locally."""

        await self._apply(self.client.wake(now))
        if self.config.wake_chime:
            # After the actions, because interrupting a reply drops queued
            # audio and would take the chime with it.
            await self.playback.play_now(CHIME)

    async def _receive(self) -> None:
        async for message in self.connection:
            now = time.monotonic()
            if isinstance(message, bytes):
                await self._apply(self.client.on_server_audio(message, now))
            else:
                await self._apply(self.client.on_server_text(message, now))
                self.playback.prebuffer_ms = self.client.playback_prebuffer_ms
        # A closed socket ends the whole session: the other tasks would happily
        # keep ticking and capturing against a connection that is gone.
        raise ConnectionError("The add-on closed the connection")

    async def _send_microphone(self) -> None:
        gain = self.config.capture_gain
        wake_gain = self.config.wake_word_gain
        while True:
            chunk = await self.capture.read()
            if self.client.streaming_microphone:
                async with self._send_lock:
                    await self.connection.send(amplified(chunk, gain))
            if self.wake_word is None:
                continue
            if self.client.active:
                # Listening for the wake word during a conversation would only
                # let the assistant's own voice trigger it.
                self.wake_word.reset()
            # The wake word gets its own gain. Its model is not level
            # invariant, and the amount that suits it is not the amount that
            # suits the model on the other end of the socket.
            elif self.wake_word.feed(amplified(chunk, wake_gain), time.monotonic()):
                await self._wake(time.monotonic())

    async def _tick(self) -> None:
        while True:
            await asyncio.sleep(TICK_SECS)
            await self._apply(self.client.tick(time.monotonic()))

    async def _read_commands(self) -> None:
        while True:
            command = (await self._next_command()).strip().lower()
            now = time.monotonic()
            if command in {"q", "quit", "exit"}:
                raise SystemExit(0)
            if command in {"s", "stop"}:
                await self._apply(self.client.request_stop(now))
            else:
                await self._wake(now)


class ConsoleCommands:
    """Read keyboard lines through the event loop rather than a thread.

    A thread parked in ``readline`` cannot be cancelled, and shutting the
    interpreter down would then wait for a keypress that is not coming. Under
    systemd there is no keyboard at all: stdin reports end of file, the reader
    is dropped, and conversations start from the server or a wake word instead.
    """

    def __init__(self):
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._attached = False

    async def __call__(self) -> str:
        if not self._attached:
            self._attached = True
            try:
                asyncio.get_running_loop().add_reader(sys.stdin, self._readable)
            except (OSError, ValueError):
                logger.info("No console input available; waiting for the server")
        return await self._queue.get()

    def _readable(self) -> None:
        line = sys.stdin.readline()
        if not line:
            with contextlib.suppress(Exception):
                asyncio.get_running_loop().remove_reader(sys.stdin)
            logger.info("Console input closed; waiting for the server")
            return
        self._queue.put_nowait(line)


def _wake_word(config: Config) -> WakeWord | None:
    """Load the configured wake word, or fail with the reason it could not be."""

    if not config.wake_word_model:
        return None
    try:
        score, forget = open_wake_word(config.wake_word_model)
    except Exception as error:
        # A configured wake word that cannot load is a broken satellite, not
        # something to start quietly without.
        raise SystemExit(
            f"Cannot load the wake word {config.wake_word_model!r}: {error}\n"
            "Install it with: pip install pyopen-wakeword"
        ) from error
    logger.info(
        "Wake word %s ready (threshold %.2f, gain %.1f)",
        config.wake_word_model,
        config.wake_word_threshold,
        config.wake_word_gain,
    )
    return WakeWord(
        score,
        forget=forget,
        threshold=config.wake_word_threshold,
        cooldown_secs=config.wake_word_cooldown_secs,
        name=config.wake_word_model,
    )


async def run(config: Config) -> None:
    import websockets  # Imported here so the session layer stays testable.

    capture = AlsaCapture(config.capture_device)
    console = ConsoleCommands()
    wake_word = _wake_word(config)
    delay = RECONNECT_MIN_SECS
    try:
        while True:
            try:
                await capture.ensure_running()
                async with websockets.connect(
                    config.url,
                    max_size=None,
                    ping_interval=20,
                    ping_timeout=20,
                ) as connection:
                    logger.info("Connected to %s", config.url.split("?")[0])
                    delay = RECONNECT_MIN_SECS
                    await Session(
                        config,
                        connection,
                        capture,
                        commands=console,
                        wake_word=wake_word,
                    ).run()
            except BaseExceptionGroup as group:
                # The task group reports whatever ended the session; quitting
                # from the console is a request, not a failure to retry.
                if group.subgroup(SystemExit) is not None:
                    raise SystemExit(0) from None
                logger.warning("Session ended: %s", group)
            except Exception as error:
                logger.warning("Session ended: %s: %s", type(error).__name__, error)
            logger.info("Reconnecting in %.0f s", delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX_SECS)
    finally:
        await capture.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path.home() / ".config" / "pipecat-satellite.toml",
        help="TOML file holding the websocket URL and the audio devices",
    )
    arguments = parser.parse_args()
    path = arguments.config if arguments.config.is_file() else None
    config = Config.load(path)
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    # DEBUG is for tuning the wake word and the microphone level; the websocket
    # library logs every audio frame at that level and buries everything else.
    logging.getLogger("websockets").setLevel(logging.INFO)
    print("Enter = talk, s = stop, q = quit", flush=True)
    with contextlib.suppress(KeyboardInterrupt, SystemExit):
        asyncio.run(run(config))


if __name__ == "__main__":
    main()
