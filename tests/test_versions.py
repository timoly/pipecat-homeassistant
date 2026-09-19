"""Keep the add-on, Python package, and UI versions released together."""

from __future__ import annotations

import json
import re
import tomllib
import unittest
from pathlib import Path

ADDON_ROOT = Path(__file__).resolve().parents[1] / "addons" / "pipecat_assist"


@unittest.skipUnless(ADDON_ROOT.is_dir(), "add-on sources are not mounted next to the tests")
class VersionTests(unittest.TestCase):
    def test_package_versions_match_the_add_on_version(self):
        config = (ADDON_ROOT / "config.yaml").read_text(encoding="utf-8")
        addon_version = re.search(r'^version: "([^"]+)"$', config, re.MULTILINE).group(1)
        pyproject = tomllib.loads((ADDON_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        package = json.loads((ADDON_ROOT / "ui-src" / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(pyproject["project"]["version"], addon_version)
        self.assertEqual(package["version"], addon_version)


if __name__ == "__main__":
    unittest.main()
