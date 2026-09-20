#!/usr/bin/env python3
"""Derive a Home Assistant Voice PE configuration that talks to Pipecat Assist.

The official configuration is a single file whose wake word starts Home
Assistant Assist. ESPHome packages can add to a configuration but not replace
one of its automations, so the file is derived instead: this script downloads
it and applies the four edits below, keeping everything else — the LED ring,
buttons, timers, media player and the built-in assistant — as it is.

    python3 derive_voice_pe_config.py > home-assistant-voice-pe.yaml

Upstream: https://github.com/esphome/home-assistant-voice-pe (MIT for YAML).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request

OFFICIAL_URL = (
    "https://raw.githubusercontent.com/esphome/home-assistant-voice-pe/{ref}/home-assistant-voice.yaml"
)

HEADER = """# Home Assistant Voice: Preview Edition, talking to Pipecat Assist.
#
# Derived from the official configuration ({url})
# with components/va_pipecat/examples/derive_voice_pe_config.py. The wake word
# starts the Pipecat satellite instead of Home Assistant Assist; the LED ring,
# buttons, timers and media player are untouched. The center button still
# starts the built-in Home Assistant Assist pipeline.
#
# Keep your own `api: encryption: key:` and Wi-Fi settings from the
# configuration ESPHome Builder generated when you took control of the device.
# The official firmware can be restored at any time from
# https://esphome.github.io/home-assistant-voice-pe/
"""

# The wake word runs the Pipecat conversation instead of Home Assistant Assist.
WAKE_WORD_BEFORE = """                            - voice_assistant.start:
                                wake_word: !lambda return wake_word;
"""
WAKE_WORD_AFTER = """                            - va_pipecat.start: pipecat_va
"""

# The add-on provisions the satellite through a dynamic native API action.
API_BEFORE = """api:
  id: api_id
"""
API_AFTER = """api:
  id: api_id
  custom_services: true
"""

EXTERNAL_COMPONENTS_BEFORE = """external_components:
"""
EXTERNAL_COMPONENTS_AFTER = """external_components:
  - source: github://{repo}@{component_ref}
    components:
      - va_pipecat
"""

# Phases the satellite reports drive the LED ring the official scripts own.
VA_PIPECAT_BLOCK = """
va_pipecat:
  id: pipecat_va
  auto_provision: true
  microphone:
    microphone: i2s_mics
    channels: 0  # The XMOS chip puts the echo-cancelled audio on channel 0.
  speaker: announcement_resampling_speaker
  barge_in: true
  playback_buffer_size: 2MB
  on_phase:
    - lambda: |-
        if (phase == "listening") {
          id(voice_assistant_phase) = ${voice_assist_listening_for_command_phase_id};
        } else if (phase == "thinking") {
          id(voice_assistant_phase) = ${voice_assist_thinking_phase_id};
        } else if (phase == "speaking" || phase == "replying") {
          id(voice_assistant_phase) = ${voice_assist_replying_phase_id};
        } else {
          id(voice_assistant_phase) = ${voice_assist_idle_phase_id};
        }
    - script.execute: control_leds
    # Quiet the music while the assistant speaks, as the official pipeline does.
    - if:
        condition:
          lambda: return phase == "idle" || phase == "thanks";
        then:
          - mixer_speaker.apply_ducking:
              id: media_mixing_input
              decibel_reduction: 0
              duration: 1.0s
        else:
          - mixer_speaker.apply_ducking:
              id: media_mixing_input
              decibel_reduction: 20
              duration: 0.0s
  on_error:
    - lambda: id(voice_assistant_phase) = ${voice_assist_error_phase_id};
    - script.execute: control_leds
    - delay: 2s
    - lambda: id(voice_assistant_phase) = ${voice_assist_idle_phase_id};
    - script.execute: control_leds
"""


def derive(official: str, *, url: str, repo: str, component_ref: str) -> str:
    """Return the official configuration with the Pipecat satellite wired in."""

    edits = (
        (WAKE_WORD_BEFORE, WAKE_WORD_AFTER),
        (API_BEFORE, API_AFTER),
        (
            EXTERNAL_COMPONENTS_BEFORE,
            EXTERNAL_COMPONENTS_AFTER.format(repo=repo, component_ref=component_ref),
        ),
    )
    derived = official
    for before, after in edits:
        if derived.count(before) != 1:
            raise SystemExit(
                f"The official configuration changed: expected exactly one\n{before}"
            )
        derived = derived.replace(before, after)
    return HEADER.format(url=url) + "\n" + derived.rstrip("\n") + "\n" + VA_PIPECAT_BLOCK


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="dev", help="Voice PE branch or tag to derive from")
    parser.add_argument(
        "--repo",
        default="timoly/pipecat-homeassistant",
        help="Repository that provides the va_pipecat component",
    )
    parser.add_argument("--component-ref", default="main", help="Branch or tag of that repository")
    args = parser.parse_args()

    url = OFFICIAL_URL.format(ref=args.ref)
    with urllib.request.urlopen(url) as response:
        official = response.read().decode("utf-8")
    sys.stdout.write(derive(official, url=url, repo=args.repo, component_ref=args.component_ref))


if __name__ == "__main__":
    main()
