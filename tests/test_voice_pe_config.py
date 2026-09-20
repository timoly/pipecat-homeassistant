"""The derived Home Assistant Voice PE configuration keeps its Pipecat wiring."""

from __future__ import annotations

import unittest
from pathlib import Path

try:
    import yaml
except ImportError:  # The lightweight CI job runs without PyYAML.
    yaml = None

CONFIG = (
    Path(__file__).resolve().parents[1]
    / "components"
    / "va_pipecat"
    / "examples"
    / "home-assistant-voice-pe.yaml"
)


def _load(text: str) -> dict:
    """Load ESPHome YAML, keeping !lambda and friends out of the way."""

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_multi_constructor("!", lambda loader, suffix, node: None)
    return yaml.load(text, Loader)


@unittest.skipUnless(yaml and CONFIG.is_file(), "PyYAML or the example config is unavailable")
class VoicePeConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = _load(CONFIG.read_text(encoding="utf-8"))

    def test_the_wake_word_starts_the_pipecat_satellite(self):
        wake_word = str(self.config["micro_wake_word"]["on_wake_word_detected"])

        self.assertIn("va_pipecat.start", wake_word)
        self.assertNotIn("voice_assistant.start", wake_word)

    def test_the_satellite_is_provisioned_over_the_native_api(self):
        self.assertTrue(self.config["api"]["custom_services"])
        self.assertTrue(self.config["va_pipecat"]["auto_provision"])
        sources = [str(item.get("source")) for item in self.config["external_components"]]
        self.assertTrue(any("pipecat-homeassistant" in source for source in sources))

    def test_the_satellite_uses_the_echo_cancelled_microphone_and_speaker(self):
        satellite = self.config["va_pipecat"]

        self.assertEqual(satellite["microphone"], {"microphone": "i2s_mics", "channels": 0})
        self.assertEqual(satellite["speaker"], "announcement_resampling_speaker")
        self.assertTrue(satellite["barge_in"])

    def test_phases_drive_the_official_led_ring(self):
        phase_actions = str(self.config["va_pipecat"]["on_phase"])

        self.assertIn("voice_assistant_phase", phase_actions)
        self.assertIn("control_leds", phase_actions)
        self.assertIn("apply_ducking", phase_actions)

    def test_the_official_hardware_configuration_is_kept(self):
        for section in ("esp32", "psram", "micro_wake_word", "speaker", "microphone", "light"):
            self.assertIn(section, self.config)


if __name__ == "__main__":
    unittest.main()
