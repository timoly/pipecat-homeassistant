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

Change a setting by editing the line that is already there. TOML rejects a key
that appears twice, so appending one that the file already has stops the
satellite from starting at all — check a file you are unsure of with:

```bash
python3 -c "import tomllib,pathlib; tomllib.loads(pathlib.Path.home().joinpath('.config/pipecat-satellite.toml').read_text()); print('ok')"
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
number: it should reach 0.7 or more. If it sits near 0.02, check the capture
level before reaching for `wake_word_gain` — gain is not a free improvement, and
a score that falls when you raise it is clipping, not a threshold problem. Only
once the number is right is `wake_word_threshold` worth touching: raise it if the
satellite wakes on its own, lower it if it still misses you.

The report also carries the level the detector saw, so a score can be read
against the loudness that produced it:

```
Wake word okay_nabu: best 0.180 at peak 4212 in the last second (threshold 0.40)
```

When a phrase scores close but not close enough, set
`wake_word_save_near_misses = true` and the audio behind each near miss is
written to a WAV in the temporary directory, at most one every twenty seconds.
Putting that file through the model on its own is what separates a model that
cannot hear the phrase from a stream that reached it damaged:

```bash
.venv/bin/python -m pyopen_wakeword --model okay_nabu /tmp/wake-miss-*.wav
```

`detected` there and a miss in the stream means the problem is on the way in —
look for `overrun` in the log. `not-detected` both ways means the model needs the
phrase said differently, or a longer one trained for it.

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

**Microphone level.** Leave both gains at 1.0 until a measurement says
otherwise. A USB speakerphone runs its own automatic gain, and an Anker
PowerConf reaches 20000–28000 of full scale from normal speech across a room —
any multiplier on top of that clips the loudest part of every sentence, which
costs a model far more than quiet audio does. With `log_level = "DEBUG"` the
client reports the level once a second and says so:

```
Capture level: peak 28469 x4.0 = 32767 CLIPPING rms 4528 (assistant quiet)
```

Speak normally from where you will stand — no wake word needed, the level is
measured whether or not a conversation is open — and read the peaks:

```bash
journalctl --user-unit=pipecat-satellite --since "-2min" | grep -o "peak [0-9]*" | sort -k2 -rn | head -5
```

Aim for a peak between 8000 and 20000 after gain. Raise `capture_gain` only if
the device really is quiet, and never past the point where CLIPPING appears.

**Barge-in.** `barge_in = true` keeps the microphone open while the assistant
speaks, so you can interrupt it. That only works when the speakerphone cancels
its own output; otherwise the model hears itself, treats it as the user's turn,
and the conversation never settles — phases flap between speaking and listening,
and the assistant answers itself.

There is nothing to tune inside the cancellation itself: it runs in the
speakerphone's own processing. What you decide is whether to trust it. With
`log_level = "DEBUG"` the client reports the captured level once a second and
says what the speaker was doing:

```
Capture level: peak 210 rms 48 (assistant quiet)
Capture level: peak 265 rms 61 (assistant speaking)
```

Those two lines are the answer. Ask for a long answer, stay silent while it
plays, and compare. A level while the assistant speaks that stays near the
quiet-room level means cancellation is working and `barge_in = true` is safe. A
level that jumps by a factor of ten or more means the microphone is hearing the
reply, and `barge_in = false` closes it for the duration — you then interrupt
with the wake word instead of by talking over it.

Three things help when cancellation is marginal, in this order: lower the
playback volume, which is the strongest lever since the echo scales with it;
lower `capture_gain`, which amplifies the echo along with the speech; and stand
the device on a hard flat surface away from a wall that reflects its own output
back into it.

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
journalctl --user-unit=pipecat-satellite -f
```

`--user-unit=` rather than `--user -u`: Raspberry Pi OS keeps the journal in
memory and does not split it per user, so the second form reports that no
journal files were found. To keep logs across reboots, create the directory the
journal becomes persistent in:

```bash
sudo mkdir -p /var/log/journal
sudo systemd-tmpfiles --create --prefix /var/log/journal
sudo systemctl restart systemd-journald
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
