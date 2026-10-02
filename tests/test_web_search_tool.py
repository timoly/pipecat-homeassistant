"""The web search prompt, which decides what "now" means to a search model."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ADDON_ROOT = Path(__file__).resolve().parents[1] / "addons" / "pipecat_assist"
if ADDON_ROOT.is_dir():
    sys.path.insert(0, str(ADDON_ROOT))

try:
    from app.web_search_tool import search_prompt
except ImportError:  # httpx and the OpenAI SDK live in the add-on image.
    search_prompt = None


@unittest.skipUnless(search_prompt, "the add-on's dependencies are not available here")
class SearchPromptTests(unittest.TestCase):
    def test_the_prompt_says_what_day_it_is(self):
        """Asked for the current top story, a search model answered with one from
        three weeks earlier. It had searched; nothing told it when now was, so it
        read "right now" against its own training cutoff."""

        when = datetime(2026, 10, 2, 7, 55, tzinfo=timezone(timedelta(hours=3)))

        prompt = search_prompt("Iltasanomat tämän hetken pääuutinen", now=when)

        self.assertIn("2026-10-02", prompt)
        self.assertIn("07:55", prompt)

    def test_the_question_survives_the_instructions(self):
        prompt = search_prompt("paljonko bitcoin on euroissa")

        self.assertTrue(prompt.endswith("paljonko bitcoin on euroissa"))
        self.assertIn("2 short sentences", prompt)


if __name__ == "__main__":
    unittest.main()
