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
    def addon_version(self) -> str:
        config = (ADDON_ROOT / "config.yaml").read_text(encoding="utf-8")
        return re.search(r'^version: "([^"]+)"$', config, re.MULTILINE).group(1)

    def test_package_versions_match_the_add_on_version(self):
        addon_version = self.addon_version()
        pyproject = tomllib.loads((ADDON_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        package = json.loads((ADDON_ROOT / "ui-src" / "package.json").read_text(encoding="utf-8"))

        self.assertEqual(pyproject["project"]["version"], addon_version)
        self.assertEqual(package["version"], addon_version)

    def test_the_built_ui_carries_the_add_on_version(self):
        """The bundle is committed rather than built by the image.

        Bumping the version files alone therefore ships a UI that reports the
        previous release, which is what happened to 0.1.87: run
        ``npm run build`` in ui-src after every bump.
        """

        bundle = ADDON_ROOT / "app" / "ui" / "index.js"
        page = ADDON_ROOT / "app" / "ui" / "index.html"
        if not (bundle.is_file() and page.is_file()):
            self.skipTest("the UI has not been built")
        addon_version = self.addon_version()

        rebuild = "run `npm run build` in ui-src"
        self.assertTrue(
            addon_version in bundle.read_text(encoding="utf-8"),
            f"index.js does not mention {addon_version}; {rebuild}",
        )
        # The cache-busting query keeps a browser from serving the old bundle.
        self.assertTrue(
            f"?v={addon_version}" in page.read_text(encoding="utf-8"),
            f"index.html does not ask for ?v={addon_version}; {rebuild}",
        )


if __name__ == "__main__":
    unittest.main()
