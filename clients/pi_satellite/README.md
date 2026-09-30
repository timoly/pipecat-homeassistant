# Raspberry Pi voice satellite

A Linux satellite for the Pipecat Assist add-on. It speaks the same
`va-pipecat` protocol as the ESPHome component in `components/va_pipecat`, so
the add-on treats it exactly like a Voice PE: raw PCM16 up at 16 kHz, down at
24 kHz, compact JSON for control, one OpenAI Live session per conversation.

Tested target: Raspberry Pi 3 A+ with Raspberry Pi OS Lite and an Anker
PowerConf USB speakerphone. Audio goes through `arecord` and `aplay`, so talking
to the add-on needs nothing but `websockets`; a wake word is optional and brings
openWakeWord with it.

## Install

```bash
sudo apt update && sudo apt install -y python3-venv alsa-utils
mkdir -p ~/pipecat-satellite && cd ~/pipecat-satellite
python3 -m venv .venv && .venv/bin/pip install websockets
```

Copy `satellite.py`, `satellite_protocol.py` and `config.example.toml` onto the
Pi, then write your own configuration:

```bash
cp config.example.toml ~/.config/pipecat-satellite.toml
chmod 600 ~/.config/pipecat-satellite.toml
```

The websocket URL is on the add-on's **Runtime** tab under *ESPHome satellite*.
It contains the shared secret, so it belongs in that file and nowhere else.

Find the ALSA device names — never the card numbers, which move between boots:

```bash
arecord -l && aplay -l
```

A USB speakerphone shows up as something like `card 2: PowerConf`, which is
`plughw:CARD=PowerConf` in the configuration.

## Run

```bash
~/pipecat-satellite/.venv/bin/python ~/pipecat-satellite/satellite.py
```

Press Enter to start talking, `s` to stop the conversation, `q` to quit.
Transcripts print as they arrive. This keyboard control exists so the audio
chain can be verified before a wake word is added; the protocol does not care
what woke the satellite, so a wake word or a button will call the same
`wake()`.

## Wake word

Without one, conversations start from the keyboard. To use one:

```bash
.venv/bin/pip install pyopen-wakeword
```

Then set `wake_word_model` and restart. The built-in phrases are `okay_nabu`,
`hey_jarvis`, `hey_mycroft`, `alexa` and `hey_rhasspy`. **`okay_nabu` is the one
a Home Assistant Voice PE answers to**, so a house with both kinds of satellite
keeps a single wake word.

`pyopen-wakeword` is Rhasspy's openWakeWord, the one Home Assistant's own wake
word add-on uses. It carries its own compiled TensorFlow Lite library and its
models, so there is no ONNX or TFLite runtime to find for the Pi. Its wheels are
built for `aarch64`, so a 64-bit Raspberry Pi OS is required — check with
`uname -m`.

Tune it from the log. With `log_level = "DEBUG"` the detector reports its best
score once a second, so a model that hears nothing can be told from one that is
not running at all. Speak normally from where you will stand and watch the
number: it should reach 0.7 or more. If it sits near 0.02 and only a raised
voice gets through, raise `wake_word_gain` — an Anker PowerConf needs about 4.
Too much clips a raised voice into distortion and costs detections. Only once
the number is right is `wake_word_threshold` worth touching: raise it if the
satellite wakes on its own, lower it if it still misses you.

To check the model and the microphone without this client in the way, record
yourself and run pyopen-wakeword's own command over the file:

```bash
arecord -D plughw:CARD=PowerConf -f S16_LE -r 16000 -c 1 -d 6 /tmp/wake.wav
.venv/bin/python -m pyopen_wakeword --model okay_nabu /tmp/wake.wav
```

**A phrase of your own**, for example a Finnish one, means training a model.
openWakeWord's training notebook generates thousands of synthetic samples with
Piper text-to-speech, which has Finnish voices, and produces a `.tflite` file
that goes straight into `wake_word_model`. Two things are worth knowing before
starting: short phrases false-trigger much more often, so three syllables is
about the floor, and a model trained here does not run on a Voice PE — ESP32
uses microWakeWord, a separate pipeline.

`wake_word.py` only needs a function that scores a 10 ms frame, so another
engine can replace this one without touching the rest.

## Tuning

**Microphone level.** Speakerphones vary wildly in how hot their USB capture
is, and ALSA may have no gain left to give:

```bash
amixer -c PowerConf                     # is `Mic` already at 100%?
arecord -D plughw:CARD=PowerConf -f S16_LE -r 16000 -c 1 -V mono -d 15 /dev/null
```

If the meter sits near a few percent while you speak from a couple of metres,
raise `capture_gain`. It applies only to the audio sent to the add-on. The wake
word has its own `wake_word_gain`, because neither model is level invariant and
the amount that suits one is not the amount that suits the other. The add-on logs what it actually receives once per second
as `ESPHome audio ingress window=... peak=... rms=...`; aim for a peak of
3000–10000 when speaking normally. The Anker PowerConf needs about `4.0`.

**Barge-in.** `barge_in = true` keeps the microphone open while the assistant
speaks, so you can interrupt it. That only works when the speakerphone cancels
its own echo; otherwise the assistant hears itself and interrupts itself. Test
it before trusting it:

```bash
arecord -D plughw:CARD=PowerConf -f S16_LE -r 16000 -c 1 -d 8 /tmp/echo.wav & \
  sleep 1; aplay -D plughw:CARD=PowerConf /tmp/some-speech.wav; wait
```

If the recording contains the playback at full volume, set `barge_in = false`.

**Playback volume** is an ALSA control on the device, so the same mixer applies:

```bash
amixer -c PowerConf sset "PCM",0 80%
```

## Why the assistant says nothing first

The add-on suppresses a flow's greeting for a satellite running a live model,
on purpose: the user spoke first, and a greeting nobody asked for is billed
live audio. A satellite also holds its connection from boot, so a greeting on
connect would play once, at night, to an empty room.

The acknowledgement is local instead. `wake_chime` plays a short blip the
moment the wake word lands, the way a Voice PE lights its ring, and it costs
nothing because it never reaches the server.

## Cost

The add-on opens one Live session per conversation and bills by the minute
while it is open, so this client reports every window that closes without
speech (`flush`) and every explicit stop (`stop`). It also respects the
follow-up window the server advertises in its `hello`. Lowering *Follow-up
listening (ms)* in the add-on to around 8000 is the single most effective
saving if the satellite sits in a room with background noise.

## Running at boot

`pipecat-satellite.service` is a systemd user service, so it runs as the user
who owns the audio device and the configuration in `~/.config`, with no root
anywhere. It needs a wake word to be useful — a service has no keyboard — so set
`wake_word_model` first.

```bash
mkdir -p ~/.config/systemd/user
cp ~/pipecat-homeassistant/clients/pi_satellite/pipecat-satellite.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now pipecat-satellite
sudo loginctl enable-linger "$USER"
```

That last line is what keeps it running with nobody logged in. Afterwards:

```bash
systemctl --user status pipecat-satellite
journalctl --user -u pipecat-satellite -f
```

After a `git pull` that changes `satellite.py`, restart it:

```bash
systemctl --user restart pipecat-satellite
```

## Files

| File | Purpose |
| --- | --- |
| `satellite_protocol.py` | The conversation state machine: no audio, no sockets, fully testable |
| `satellite.py` | ALSA capture and playback, the websocket, the keyboard |
| `config.example.toml` | Template for `~/.config/pipecat-satellite.toml` |
| `pipecat-satellite.service` | systemd unit for unattended operation |

`tests/test_pi_satellite_protocol.py` in the repository root runs the state
machine against the add-on's own server-side protocol, so a change that breaks
the contract fails without any hardware.
