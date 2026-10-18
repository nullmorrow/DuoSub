"""R2b display-name fallback boundaries (Qt-free)."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subkit.core.people import Character, NameForm, character_display_name


class CharacterDisplayNames(unittest.TestCase):
    def test_fallback_chain_and_empty_forms(self):
        ch = Character("chr_internal", {"en": "English", "ja": "日本名", "zh-Hant": "繁體名"}, aliases=["别名"])
        self.assertEqual(character_display_name(ch, "zh-Hans", "标签"), "繁體名")
        self.assertEqual(character_display_name(ch, "en"), "English")
        self.assertEqual(character_display_name(ch, "fr"), "日本名")
        ch.names.pop("ja")
        self.assertEqual(character_display_name(ch, "fr"), "English")
        ch.names = {"ja": NameForm()}
        self.assertEqual(character_display_name(ch, "zh-Hans"), "别名")
        ch.aliases.clear()
        self.assertEqual(character_display_name(ch, "zh-Hans", "原标签"), "原标签")
        self.assertEqual(character_display_name(ch, "zh-Hans"), "（未命名人物）")
        self.assertEqual(character_display_name(ch, "zh-Hans", "原标签", "label"), "原标签")

    def test_short_or_missing_part_uses_same_language_full_name(self):
        ch = Character("chr_internal", {"ja": NameForm(given="枝織", short="family"), "en": "Shiori"})
        for format in ("short", "family", "given", "full"):
            with self.subTest(format=format):
                self.assertEqual(character_display_name(ch, "zh-Hans", format=format), "枝織")


if __name__ == "__main__":
    unittest.main()
