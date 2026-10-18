"""Inline reset targets survive preview/release pruning and source aliases."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests import test_stage0 as stage0
from subkit.ass import AssDocument, Style
from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.model.document import Document, Group, Member, Run
from subkit.styles_default import default_styles_ass


class ExportStyles(unittest.TestCase):
    def test_named_reset_targets_with_spaces_multiple_and_bare_reset(self):
        template = AssDocument.parse(default_styles_ass())
        template.styles.extend([Style.new("Other"), Style.new("With spaces"), Style.new("Unused")])
        text = r"{\fs80}A{\r}B{\rOther}X{\rWith spaces\b1}Y"
        doc = Document(episode="E1")
        doc.members["m"] = Member("m", "g", "en", "original", "dialogue", 0, 1000,
                                  style="JP", runs=[Run(text, "text", ins="i", b=len(text))])
        doc.groups = [Group("g", "dialogue", ["m"])]
        path = Path(tempfile.mkdtemp(prefix="subkit_export_styles_"))
        cache = ExportCache()
        for mode in ("preview", "preview", "release"):
            out, _ = build_ass(doc, {}, template, path, ExportOptions(mode=mode, ruby=False), cache=cache)
            self.assertEqual({s.name for s in out.styles}, {"JP", "Other", "With spaces"})
            self.assertEqual(out.events[0].text, text)

    def test_aliases_with_common_prefix_preserve_block_bytes(self):
        from subkit.core.export import _alias_resets
        self.assertEqual(_alias_resets(r"{note\rOther\rOtherTwo\b1}X", {"Other": "src.Other", "OtherTwo": "src.OtherTwo"}),
                         r"{note\rsrc.Other\rsrc.OtherTwo\b1}X")

    def test_plain_text_reset_like_string_does_not_keep_style(self):
        from subkit.core.export import _reset_styles
        self.assertEqual(_reset_styles(r"outside \rOther {plain comment}"), set())


class SourceResetStyles(unittest.TestCase):
    setUp = stage0.Stage0.setUp
    tearDown = stage0.Stage0.tearDown
    merged = stage0.Stage0.merged

    def test_source_reset_collision_is_aliased_and_source_unchanged(self):
        text = r"{\rDefault}招牌{\rSign}字"
        self.cn.events[3].text = text
        # The byte copy is what the exporter reads for source styles.
        path = self.ep_dir / self.cn.copy
        import os, stat
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        source = AssDocument.load(path)
        source.events[3].text = text
        source.save(path)
        doc = self.merged()
        template = AssDocument.parse(default_styles_ass())
        template.styles.append(Style.new("Default"))
        cache = ExportCache()
        before = path.read_bytes()
        for mode in ("preview", "preview", "release"):
            out, _ = build_ass(doc, self.sources, template, self.ep_dir, ExportOptions(mode=mode, ruby=False), cache=cache)
            event = next(e for e in out.events if "招牌" in e.text)
            self.assertIn(r"\rsrc.Default", event.text)
            self.assertIn(r"\rSign", event.text)
            self.assertIsNotNone(out.style("src.Default"))
            self.assertIsNotNone(out.style("Sign"))
            self.assertEqual(doc.members[next(m.id for m in doc.members.values() if "招牌" in m.text)].text, text)
            self.assertEqual(path.read_bytes(), before)
