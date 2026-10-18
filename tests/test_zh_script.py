"""Bidirectional Chinese display: real OpenCC/libass and offscreen Qt."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from subkit import templates as T
from subkit.ass.document import AssDocument
from subkit.ass.text import VisibleText
from subkit.core.display import DisplayRules, display_name, display_text
from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.core.ops import Editor
from subkit.core.people import Character, CharacterBook
from subkit.core.review import build_packet
from subkit.core.review_import import Answer, apply, match
from subkit.core.workspace import Project
from tests.test_pinyin_export import document


def rules(direction, variant="tw", phrases=False, **scope):
    return DisplayRules.from_settings({"display": {"zh_script": direction,
        "zh_traditional": {"variant": variant, "phrases": phrases}, "zh_scope": scope}})


class ChineseScript(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="zh-script-")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.styles = T.build_styles(T.FontSet("test", "test", "Microsoft YaHei", "Microsoft YaHei", "Arial",
                                              pinyin_font="Arial"), T.palette("classic"))
        self.template = AssDocument.parse(self.styles)

    def export(self, doc, display, *, cache=None, **kw):
        return build_ass(doc, {}, self.template, self.path, ExportOptions(display=display, **kw),
                         measure_cache=self.path / "render.sqlite3", cache=cache)

    def test_episode_direction_override_preserves_settings_and_scope(self):
        project = Project.create(self.path / "override", "两份", template_text=self.styles)
        project.data.setdefault("settings", {})["display"] = {
            "zh_script": "off", "zh_traditional": {"variant": "hk", "phrases": True},
            "zh_scope": {"dialogue": False, "gloss": True, "raw": False}}
        ep = project.add_episode("E1")
        ep.document = document("软件猫{@注:猫=软件}{@Raw=软件}")
        before = copy.deepcopy(project.settings())
        for direction, note in (("simplified", "软件"), ("traditional", "軟體")):
            ass, rep = ep.build_export(zh_direction=direction)
            self.assertEqual(VisibleText(ass.events[0].text).text, "软件猫软件")
            self.assertEqual(VisibleText(ass.events[-1].text).text, note)
            self.assertEqual(rep.zh_script, direction)
            self.assertEqual(project.settings(), before)
        with self.assertRaises(ValueError):
            ep.build_export(zh_direction="off")

    def test_all_chinese_variants_and_inserted_text(self):
        for lang in ("zh-Hant", "zh-Hans", "zh", "zh-TW", "zh-HK", "zh-CN"):
            with self.subTest(lang=lang):
                doc = document("这里是", lang=lang)
                before = copy.deepcopy(doc)
                ass, rep = self.export(doc, rules("traditional"))
                self.assertEqual(ass.events[0].text, "這裡是")
                self.assertEqual((rep.zh_lines, rep.zh_script), (1, "traditional"))
                self.assertEqual(doc, before)
        ass, _ = self.export(document("這裡是軟體"), rules("simplified", phrases=True))
        self.assertEqual(ass.events[0].text, "这里是软件")
        ass, _ = self.export(document("這裡是軟體與资讯", lang="zh-Hant"), rules("traditional"))
        self.assertEqual(ass.events[0].text, "這裡是軟體與資訊")

    def test_non_chinese_and_off_preserve_bytes(self):
        for lang in ("ja", "en", "yue"):
            doc = document("这里是軟體", lang=lang)
            baseline, _ = self.export(doc, rules("off"))
            for direction in ("traditional", "simplified"):
                ass, rep = self.export(doc, rules(direction, phrases=True))
                self.assertEqual(ass.serialize(), baseline.serialize())
                self.assertEqual(rep.zh_lines, 0)
        text = r"{\fn软件}这里是軟體\N资讯"
        self.assertEqual(display_text(text, rules("off"))[0], text)
        self.assertEqual(rules("off").convert(text), text)

    def test_legacy_settings_and_explicit_off_precedence(self):
        legacy = {"display": {"zh_traditional": {"enabled": True, "variant": "hk", "phrases": True}}}
        old = DisplayRules.from_settings(legacy)
        self.assertEqual((old.direction, old.zh_traditional_variant, old.zh_traditional_phrases),
                         ("traditional", "hk", True))
        self.assertEqual(old, rules("traditional", "hk", True))
        legacy["display"]["zh_script"] = "off"
        self.assertFalse(DisplayRules.from_settings(legacy).converted_for(document().members["m_1"]))
        self.assertEqual(DisplayRules.from_settings({}).direction, "off")
        legacy["display"]["zh_script"] = "invalid"
        self.assertEqual(DisplayRules.from_settings(legacy).direction, "off")

    def test_regional_configs_and_traditional_regression_bytes(self):
        for variant, glyphs in (("tw", "裡面軟件"), ("hk", "裏面軟件"), ("std", "裏面軟件")):
            for phrases in (False, True):
                with self.subTest(variant=variant, phrases=phrases):
                    expected = glyphs if not phrases else glyphs.replace("軟件", "軟體")
                    ass, _ = self.export(document("里面软件"), rules("traditional", variant, phrases))
                    reference, _ = self.export(document(expected), rules("off"))
                    self.assertEqual(ass.serialize(), reference.serialize())
                    simplified = rules("simplified", variant, phrases).convert("這裡是軟體")
                    self.assertEqual(simplified, "这里是软件" if variant == "tw" and phrases else "这里是软体")

    def test_phrase_bounds_and_pinyin_anchors_in_both_directions(self):
        cases = (("traditional", "互联网猫", "網際網路貓", [0, 1, 3, 4, 5]),
                 ("simplified", "網際網路貓", "互联网猫", [0, 1, 2, 2, 3, 4]))
        for direction, source, shown, bounds in cases:
            with self.subTest(direction=direction):
                rule = rules(direction, phrases=True)
                self.assertEqual(display_text(source, rule), (shown, bounds))
                doc = document(source, lang="zh-Hant" if direction == "simplified" else "zh-Hans")
                doc.annotations = {k: a for k, a in doc.annotations.items() if a.base == source[-1]}
                self.assertEqual(len(doc.annotations), 1)
                before = copy.deepcopy(doc)
                ass, rep = self.export(doc, rule, pinyin_mode="PY")
                reference = document(shown)
                reference.annotations = copy.deepcopy(doc.annotations)
                for ann in reference.annotations.values():
                    anchor = ann.anchor[0]
                    anchor["a"], anchor["b"] = bounds[anchor["a"]], bounds[anchor["b"]]
                    ann.base = shown[anchor["a"]:anchor["b"]]
                expected, expected_rep = self.export(reference, rules("off"), pinyin_mode="PY")
                rows = [e.text for e, o in zip(ass.events, rep.owners) if o["kind"] == "pinyin"]
                expected_rows = [e.text for e, o in zip(expected.events, expected_rep.owners) if o["kind"] == "pinyin"]
                self.assertEqual(rep.pinyin_lines, 1)
                self.assertEqual(rows, expected_rows)
                self.assertFalse(any("找不到底字" in w for w in rep.warnings))
                self.assertEqual(doc, before)

    def test_cache_changes_direction(self):
        doc, cache = document("这里是軟體"), ExportCache()
        output = []
        for direction in ("traditional", "simplified", "off"):
            ass, rep = self.export(doc, rules(direction, phrases=True), cache=cache)
            self.assertEqual(rep.laid_out, 1)
            output.append(ass.events[0].text)
        self.assertEqual(output, ["這裡是軟體", "这里是软件", "这里是軟體"])

    def test_simplified_preserves_tags_drawings_and_raw_regions(self):
        source = r"{\fn軟體}軟體{\p1}m 0 0 l 1 1 軟體{\p0}網際網路{@Raw=軟體}"
        expected = r"{\fn軟體}软件{\p1}m 0 0 l 1 1 軟體{\p0}互联网{@Raw=軟體}"
        rule = rules("simplified", phrases=True)
        self.assertEqual(display_text(source, rule)[0], expected)
        ass, _ = self.export(document(source, lang="zh-Hant"), rule)
        self.assertEqual(ass.events[0].text, expected.replace("{@Raw=軟體}", "軟體"))

    def test_gloss_anchor_after_phrase_contraction(self):
        doc = document("網際網路貓{@注:貓=animal}", lang="zh-Hant")
        ass, rep = self.export(doc, rules("simplified", phrases=True))
        reference, _ = self.export(document("互联网猫{@注:猫=animal}"), rules("off"))
        self.assertEqual(ass.serialize(), reference.serialize())
        self.assertFalse(rep.warnings)
        self.assertEqual(len(ass.events), 2)

    def test_review_packet_and_inverse_preserve_member_script(self):
        cases = (("zh-Hant", "simplified", "這裡是軟體", "软件", "硬件", "這裡是硬體"),
                 ("zh-Hans", "simplified", "这里是软件", "软件", "硬件", "这里是硬件"),
                 ("zh-Hant", "traditional", "這裡是軟體", "軟體", "硬體", "這裡是硬體"),
                 ("zh-Hans", "traditional", "这里是软件", "軟體", "硬體", "这里是硬体"))
        for lang, direction, source, target, new, stored in cases:
            with self.subTest(lang=lang, direction=direction):
                doc, rule = document(source, lang=lang), rules(direction, phrases=True)
                _, packet = build_packet(doc, episode="E1", display=rule)
                self.assertIn(target, json.loads(packet)["translation"][0]["text"])
                item = Answer("g_1", "zh", "text", target, target, new, "复查")
                suggestion = match(doc, {}, rule, item)
                self.assertEqual(suggestion.status, "ok")
                editor = Editor(doc, {})
                self.assertTrue(apply(editor, suggestion))
                self.assertEqual(doc.members["m_1"].text, stored)
                self.assertEqual(match(doc, {}, rule, item).status, "already_done")
                editor.history.undo()
                self.assertEqual(doc.members["m_1"].text, source)

    def test_names_follow_direction_and_prefer_explicit_form(self):
        book = CharacterBook(self.path / "characters.json")
        ch = Character("chr_1", {"zh-Hant": "張軟體"})
        book.characters = [ch]
        member = document("台詞", lang="zh-Hant").members["m_1"]
        member.speaker = {"character": "chr_1", "label": "張"}
        self.assertEqual(display_name(member, rules("simplified", phrases=True), book), "张软件")
        from subkit_app.views.people import PeoplePanel
        project = Mock(languages={"translation": "zh-Hant"})
        project.settings.return_value = {"display": {"zh_script": "simplified", "zh_traditional": {"phrases": True}}}
        self.assertEqual(PeoplePanel.display(Mock(project=project), ch), "张软件")
        ch.names["zh-Hans"] = Character("chr_2", {"zh-Hans": "专用简中名"}).names["zh-Hans"]
        self.assertEqual(display_name(member, rules("simplified", phrases=True), book), "专用简中名")
        self.assertEqual(PeoplePanel.display(Mock(project=project), ch), "专用简中名")

    def test_export_report_direction_text(self):
        from subkit_app import mainwindow as mw
        for direction, expected in (("traditional", "中文显示为繁体：1 行（台湾用字）"),
                                    ("simplified", "中文显示为简体：1 行")):
            _, rep = self.export(document("這裡是" if direction == "simplified" else "这里是"), rules(direction))
            with patch.object(mw, "report_lines", return_value=[]):
                lines = mw.MainWindow._export_report_lines(Mock(private_enabled=False), rep, None)
            self.assertIn(expected, lines)

    def test_body_scope_for_every_kind_and_lane(self):
        for role, scope in (("dialogue", "dialogue"), ("top", "dialogue"), ("alt", "dialogue"),
                            ("song_cn", "song"), ("song_jp", "song"), ("song", "song"),
                            ("sign", "sign"), ("comment", "sign")):
            for lane in ("translation", "original"):
                with self.subTest(role=role, lane=lane):
                    doc = document("这里是", lane=lane)
                    doc.members["m_1"].role = role
                    on, rep = self.export(doc, rules("traditional"), include_source_comments=True)
                    self.assertEqual(on.events[0].text, "這裡是")
                    self.assertEqual(rep.zh_lines, 1)
                    off, rep = self.export(doc, rules("traditional", **{scope: False}), include_source_comments=True)
                    self.assertEqual(off.events[0].text, "这里是")
                    self.assertEqual(rep.zh_lines, 0)

    def test_raw_opt_in_bounds_expansion_contraction_and_japanese(self):
        cases = (("traditional", "互联网猫", "網際網路貓", [0, 1, 3, 4, 5]),
                 ("simplified", "網際網路貓", "互联网猫", [0, 1, 2, 2, 3, 4]))
        for direction, source, shown, bounds in cases:
            with self.subTest(direction=direction):
                raw = "{@Raw=" + source + "}"
                self.assertEqual(display_text(raw, rules(direction, phrases=True)),
                                 (raw, list(range(len(source) + 1))))
                rule = rules(direction, phrases=True, raw=True)
                self.assertEqual(display_text(raw, rule), ("{@Raw=" + shown + "}", bounds))
                ass, rep = self.export(document(raw), rule)
                self.assertEqual(ass.events[0].text, shown)
                self.assertEqual(rep.zh_lines, 1)
                ass, rep = self.export(document(raw, lang="ja", lane="original", style="JP"), rule)
                self.assertEqual(ass.events[0].text, source)
                self.assertEqual(rep.zh_lines, 0)
                ass, rep = self.export(document(raw), rules(direction, phrases=True, raw=True, dialogue=False))
                self.assertEqual(ass.events[0].text, source)
                self.assertEqual(rep.zh_lines, 0)

    def test_raw_phrase_gloss_anchor_punctuation_and_preview(self):
        from subkit.core.display import PunctuationRules
        source = r"{@Raw=互联网}猫{@注:猫=互联网}\N这里是"
        expected = r"網際網路貓{@注:貓=網際網路}  這裡是"
        for punctuation in (False, True):
            rule = rules("traditional", phrases=True, raw=True)
            rule.merge_lines = (("translation", "dialogue"),)
            if punctuation:
                rule.punctuation = (("translation", PunctuationRules(comma="spaces")),)
            doc = document(source)
            before = copy.deepcopy(doc)
            ass, rep = self.export(doc, rule)
            reference, _ = self.export(document(expected), rules("off"))
            self.assertEqual(ass.serialize(), reference.serialize())
            self.assertFalse(rep.warnings)
            preview, _ = self.export(doc, rule, mode="preview")
            self.assertEqual(preview.events, ass.events)
            self.assertEqual(doc, before)

    def test_chinese_gloss_scope_independent_of_body_scope(self):
        for direction, source, note in (("traditional", "互联网", "網際網路"),
                                        ("simplified", "網際網路", "互联网")):
            doc = document("猫{@注:猫=" + source + ";font=Microsoft YaHei;size=30}")
            for body in (False, True):
                for enabled in (False, True):
                    with self.subTest(direction=direction, body=body, gloss=enabled):
                        rule = rules(direction, phrases=True, dialogue=body, gloss=enabled)
                        ass, rep = self.export(doc, rule)
                        gloss = next(e for e, owner in zip(ass.events, rep.owners) if owner["kind"] == "gloss")
                        self.assertEqual(VisibleText(gloss.text).text, note if enabled else source)
                        self.assertEqual(rep.zh_lines, int(enabled or body and direction == "traditional"))
                        self.assertFalse(rep.warnings)

    def test_japanese_gloss_kana_protection_and_opt_in(self):
        for note in ("软件", "软件かな", "软件カナ", "软件ｶﾅ", "软件ㇰ", "软件𛀀"):
            for enabled in (False, True):
                with self.subTest(note=note, enabled=enabled):
                    doc = document("国语{@注:国语=" + note + "}", lang="ja", lane="original", style="JP")
                    ass, rep = self.export(doc, rules("traditional", phrases=True, gloss_ja=enabled))
                    self.assertEqual(VisibleText(ass.events[0].text).text, "国语")
                    gloss = next(e for e, owner in zip(ass.events, rep.owners) if owner["kind"] == "gloss")
                    changed = enabled and note == "软件"
                    self.assertEqual(VisibleText(gloss.text).text, "軟體" if changed else note)
                    self.assertEqual(rep.zh_lines, int(changed))
                    self.assertFalse(rep.warnings)
        doc = document("word{@注:word=软件}", lang="en")
        ass, rep = self.export(doc, rules("traditional", gloss_ja=True))
        self.assertEqual(VisibleText(ass.events[-1].text).text, "软件")
        self.assertEqual(rep.zh_lines, 0)

    def test_report_counts_changes_and_scope_invalidates_cache(self):
        for text in ("這裡是", "English", "{@Raw=这里是}", "貓{@注:貓=animal}"):
            ass, rep = self.export(document(text), rules("traditional"))
            self.assertEqual(rep.zh_lines, 0, text)
        for role, end in (("comment", 3000), ("dialogue", 0)):
            doc = document("貓{@注:貓=软件}")
            doc.members["m_1"].role = role
            doc.members["m_1"].end = end
            ass, rep = self.export(doc, rules("traditional"), include_source_comments=True)
            self.assertEqual(len(ass.events), 1)
            self.assertEqual(rep.zh_lines, 0)
        doc, cache = document("{@Raw=互联网}猫{@注:猫=软件}"), ExportCache()
        results = []
        for raw, gloss, body in ((False, False, False), (True, True, True), (False, False, False)):
            ass, rep = self.export(doc, rules("traditional", phrases=True, raw=raw, gloss=gloss, dialogue=body), cache=cache)
            results.append([VisibleText(e.text).text for e in ass.events])
        self.assertEqual(results, [["互联网猫", "软件"], ["網際網路貓", "軟體"], ["互联网猫", "软件"]])
        rule = rules("traditional", phrases=True)
        before, _ = self.export(doc, rule, cache=cache)
        rule.zh_scope["raw"] = True
        after, _ = self.export(doc, rule, cache=cache)
        self.assertEqual(VisibleText(before.events[0].text).text, "互联网貓")
        self.assertEqual(VisibleText(after.events[0].text).text, "網際網路貓")

    def test_offscreen_menu_roundtrip_labels_preview_and_help(self):
        from PySide6.QtCore import QSettings, QRect
        from PySide6.QtWidgets import QApplication, QLabel
        from subkit_app import mainwindow as mw, help as help_ui
        from subkit_app.session import EpisodeSession
        from subkit_app.watch import InlineTextEditor
        from tests.test_convert_ui import PlayerDouble
        app = QApplication.instance() or QApplication([])
        project = Project.create(self.path / "project", "中文显示", template_text=self.styles)
        episode = project.add_episode("E1")
        episode.document = document("这里是軟體", lang="zh-Hant")
        project.data.setdefault("settings", {})["display"] = {
            "zh_traditional": {"enabled": True, "variant": "hk", "phrases": True}}
        settings = QSettings(str(self.path / "settings.ini"), QSettings.Format.IniFormat)
        with patch.object(mw, "MpvWidget", PlayerDouble), patch.object(mw, "QSettings", return_value=settings), \
                patch.object(mw.MainWindow, "_open_last", Mock()):
            window = mw.MainWindow()
        session = EpisodeSession(project, episode)
        window.project, window.session = project, session
        try:
            window._apply_rules()
            self.assertTrue(window.zh_script_actions["traditional"].isChecked())
            expected_labels = ("对白", "歌词", "屏幕字等", "旁注（中文行）", "日文行上不含假名的旁注", "{@Raw=…} 里的文字")
            self.assertEqual(tuple(a.text() for a in window.zh_scope_actions.values()), expected_labels)
            for key, action in window.zh_scope_actions.items():
                default = key not in ("gloss_ja", "raw")
                self.assertEqual(action.isChecked(), default)
                self.assertTrue(action.toolTip())
                self.assertEqual(action.toolTip(), action.statusTip())
                for expected in (not default, default):
                    with patch.object(session, "request_preview") as preview:
                        action.trigger()
                        preview.assert_called_once()
                    reopened = Project.open(project.root)
                    self.assertEqual(DisplayRules.from_settings(reopened.settings()).zh_scope[key], expected)
                    project.data["settings"]["display"]["zh_scope"][key] = not expected
                    window._apply_rules()
                    self.assertEqual(action.isChecked(), not expected)
                    project.data["settings"] = reopened.settings()
                    with patch.object(session, "request_preview") as preview:
                        window._apply_rules()
                        preview.assert_not_called()
                    self.assertEqual(action.isChecked(), expected)
                self.assertEqual(help_ui.resolve_anchor(window, focus=window.zh_script_menu), "readings")
            window.editor.set_group(episode.document, episode.document.groups[0], None, None)
            for direction in ("simplified", "traditional", "off"):
                with patch.object(session, "request_preview") as preview:
                    window.zh_script_actions[direction].trigger()
                    preview.assert_called_once()
                state = DisplayRules.from_settings(Project.open(project.root).settings())
                self.assertEqual(state.direction, direction)
                self.assertEqual((state.zh_traditional_variant, state.zh_traditional_phrases), ("hk", True))
                self.assertNotIn("enabled", project.settings()["display"]["zh_traditional"])
                self.assertEqual(sum(a.isChecked() for a in window.zh_script_actions.values()), 1)
                self.assertTrue(window.zh_script_actions[direction].isChecked())
                notes = [l for l in window.editor.findChildren(QLabel) if l.property("traditional_member")]
                self.assertTrue(notes)
                self.assertTrue(all(l.isHidden() == (direction == "off") for l in notes))
                if direction != "off":
                    self.assertTrue(all(l.text() == state.script_label for l in notes))
                inline = InlineTextEditor(episode.document, ["m_1"], "m_1", QRect(0, 0, 600, 60), 20, state, window)
                try:
                    self.assertEqual(inline.edit.text(), "这里是軟體")
                    self.assertEqual(inline.traditional_note.isHidden(), direction == "off")
                    self.assertEqual(inline.traditional_note.text(), state.script_label)
                finally:
                    inline._cancel = True
                    inline.close()
                    inline.deleteLater()
            for action in window.zh_script_actions.values():
                self.assertEqual(action.statusTip(), "只影响预览和导出，工程原文不变")
            self.assertEqual(help_ui.resolve_anchor(window, focus=window.zh_script_menu), "readings")
        finally:
            session.close()
            window.session = None
            window.close()
            window.deleteLater()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
