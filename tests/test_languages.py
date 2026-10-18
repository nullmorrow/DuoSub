"""C3 language rules and regression against the pre-C3 soma copy."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.ruby_policy import legacy_ruby_layout
from subkit.core.importer import import_source
from subkit.core.merge import MergeSettings, merge_episode
from subkit.core.ops import Editor, phrase_breaks
from subkit.core.readings import annotate_document, annotate_member, suggest_readings
from subkit.core.sdh_en import analyze_event
from subkit.core.units import build_units, is_interjection, split_range
from subkit.core.workspace import Project
from subkit.model.document import Document, Group, Member, Run
from subkit.model.serialize import document_to_json
from subkit.model.validate import validate
from subkit.reading.engine import ReadingEngine


def member(text, lang="ja", lane="original"):
    return Member("m_1", "g_1", lang, lane, "dialogue", 1000, 3000, style="JP",
                  runs=[Run(text, ins="i_1", b=len(text))])


def document(m):
    return Document("E1", groups=[Group("g_1", members=[m.id])], members={m.id: m})


class LanguageRules(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="c3-language-")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def source(self, bodies, lang="en", lane="original", sid="o1"):
        raw = "\n\n".join(f"{i+1}\n00:00:{i*4+1:02},000 --> 00:00:{i*4+3:02},000\n{body}"
                            for i, body in enumerate(bodies)) + "\n"
        path = self.path / f"{sid}.srt"
        path.write_text(raw, encoding="utf-8")
        episode_dir = self.path / ("episode-" + lang)
        src = import_source(path, src_id=sid, lang=lang, lane=lane, episode_dir=episode_dir)
        self.assertEqual(path.read_bytes(), (episode_dir / src.copy).read_bytes())
        return src

    def test_english_sdh_spans_merge_speakers_and_balanced_ledger(self):
        bodies = ["JOHN: Hello there.", "[John] Come here.", "(JOHN) I agree.",
                  "[door creaks]", "(laughs)", "♪", "♪ We sing together ♪",
                  "- <i>Hello there</i>\n- {\\an8}Good morning", "[music playing]", "[John]\nHello again."]
        src = self.source(bodies)
        self.assertEqual(src.analyzer, {"name": "en-sdh", "version": 1})
        self.assertEqual({s.kind for s in src.spans}, {"speaker_label", "sfx", "music", "marker", "format"})
        self.assertEqual([s.value for s in src.spans if s.kind == "speaker_label"], ["JOHN", "John", "JOHN", "John"])
        self.assertEqual(len([s for s in src.spans if (s.data or {}).get("turn")]), 2)
        for e, body in zip(src.events, bodies):
            self.assertEqual(e.text, body)
            spans = src.spans_of(e.i)
            self.assertTrue(all(a.b <= b.a for a, b in zip(spans, spans[1:])))
        tl = self.source(["你好"] * len(bodies), "zh-Hans", "translation", "t1")
        doc = merge_episode("E1", tl, src).document
        rep = validate(doc, {src.id: src, tl.id: tl})
        self.assertTrue(rep.ok, rep.summary())
        self.assertEqual((rep.ledger[src.id]["missing"], rep.ledger[src.id]["duplicate"]), (0, 0))
        self.assertEqual({m.speaker["label"] for m in doc.members.values() if m.speaker}, {"JOHN", "John"})
        self.assertTrue(any(u.lyric for u in build_units(src)))

    def test_uncertain_patterns_and_single_dashes_remain_dialogue(self):
        for text in ("[maybe tomorrow]", "(not really)", "Time: to go", "hello: there", "unclosed [laughs",
                     "- One line only", "- One line\nOther line"):
            with self.subTest(text=text):
                result = analyze_event(0, text)
                self.assertFalse(result.spans)
                self.assertTrue(all(line.kind == "speech" for line in result.lines))

    def test_generic_format_dashes_without_speaker_or_sound_guessing(self):
        src = self.source(["- <b>Bonjour tout le monde</b>\n- Au revoir", "[John] (laughs) ♪"], "fr")
        self.assertEqual(src.analyzer, {"name": "generic", "version": 1})
        self.assertEqual({s.kind for s in src.spans}, {"format", "marker"})
        self.assertEqual([u.text for u in build_units(src)], ["Bonjour tout le monde", "Au revoir", "[John] (laughs) ♪"])
        tl = self.source(["你好", "再见"], "zh-Hans", "translation", "t1")
        rep = validate(merge_episode("E1", tl, src).document, {src.id: src, tl.id: tl})
        self.assertTrue(rep.ok, rep.summary())

    def test_reanalysis_uses_original_language_and_keeps_source_bytes(self):
        for lang, name in (("en-GB", "en-sdh"), ("fr", "generic"), ("ja", "ja-cc")):
            with self.subTest(lang=lang):
                project = Project.create(self.path / lang, lang)
                ep = project.add_episode("E1")
                path = self.path / "reanalyze.srt"
                path.write_text("1\n00:00:01,000 --> 00:00:03,000\nJOHN: Hello\n", encoding="utf-8")
                src = ep.add_source(path, lang=lang, lane="original")
                expected = src.events_json()
                raw = (ep.dir / src.copy).read_bytes()
                src.spans = []
                result = ep.reanalyze_sources()
                self.assertIn(src.id, result)
                self.assertEqual(src.analyzer["name"], name)
                self.assertEqual(src.events_json(), expected)
                self.assertEqual((ep.dir / src.copy).read_bytes(), raw)
                self.assertTrue(list((ep.dir / "sources").glob("*.events.v*.json")))

    def test_phrase_boundaries_follow_language_and_keep_words(self):
        cases = [("ja", "今日は いい天気。帰ろう", ["今日は", "いい天気。", "帰ろう"]),
                 ("zh-Hant", "你好，世界。回家！好吗？行；走、快 来", ["你好，", "世界。", "回家！", "好吗？", "行；", "走、", "快", "来"]),
                 ("en", "Hello world, come here; go home. All done!", ["Hello world,", "come here;", "go home.", "All done!"]),
                 ("fr", "Bonjour tout le monde", ["Bonjour tout le monde"]),
                 ("en", "Value 3.14 is fine. Yes", ["Value 3.14 is fine.", "Yes"])]
        for lang, text, expected in cases:
            with self.subTest(lang=lang, text=text):
                pieces = split_range(text, 0, len(text), lang)
                self.assertEqual([text[a:b] for a, b, _ in pieces], expected)
                self.assertEqual(phrase_breaks(member(text, lang)), [a for a, _, _ in pieces[1:]])
        m = member("Hello world  Good morning", "en")
        m.runs = [Run("Hello world", ins="i_1", b=11), Run("  ", "sep"), Run("Good morning", ins="i_2", b=12)]
        self.assertEqual(phrase_breaks(m), [13])
        self.assertEqual(phrase_breaks(member("  Hello world", "en")), [])

    def test_filler_tables_and_language_boundaries(self):
        for text in "啊 嗯 哦 呃 欸 唉 哎 哈 呀 喔 噢 嘿".split():
            self.assertTrue(is_interjection(text * 3 + "！？", "zh-Hant"))
        for text in ("uh", "UM", "Oh", "aaah", "hmmmm", "Huh?!", "ERRR…", "eh", "mmm"):
            self.assertTrue(is_interjection(text, "en-US"), text)
        for text, lang in (("嗯", "en"), ("uh", "zh"), ("oh", "fr"), ("hello", "en"), ("嗯好的", "zh"), ("あっ", "fr")):
            self.assertFalse(is_interjection(text, lang))
        self.assertTrue(is_interjection("あっ…", "ja"))
        self.assertFalse(is_interjection("はい", "ja"))

    def test_compact_mode_uses_only_supported_language_fillers(self):
        for lang, text, hidden in (("en", "Uhhh!", True), ("zh-Hans", "嗯嗯！", True), ("fr", "Uhhh!", False)):
            with self.subTest(lang=lang):
                src = self.source(["Hello", text, "Bye"], lang)
                tl = self.source(["你好", "", "再见"], "zh-Hans", "translation", "t1")
                # Remove the empty translation cue from alignment, keeping its source record auditable.
                tl.roles[1] = "sign"
                doc = merge_episode("E1", tl, src, MergeSettings(filler_mode="compact")).document
                self.assertEqual(any(c.kind == "hidden_filler" for c in doc.comments.values()), hidden)
                rep = validate(doc, {src.id: src, tl.id: tl})
                self.assertTrue(rep.ok, rep.summary())

    def test_reading_generation_and_regeneration_follow_language_in_both_lanes(self):
        engine = ReadingEngine()
        for lang, lane, eligible in (("ja", "original", True), ("ja-JP", "translation", True),
                                     ("en", "original", False), ("zh-Hans", "translation", False)):
            with self.subTest(lang=lang, lane=lane):
                m = member("迷宮都市", lang, lane)
                doc = document(m)
                self.assertEqual(bool(suggest_readings(m, engine)), eligible)
                self.assertEqual(bool(annotate_document(doc, {}, engine)), eligible)
                before = document_to_json(doc)
                n = Editor(doc, {}).regenerate_readings(m.id, 0, 4, engine)
                self.assertEqual(bool(n), eligible)
                if not eligible:
                    self.assertEqual(document_to_json(doc), before)
                    self.assertEqual(annotate_member(doc, m, engine), 0)

    def test_editor_rows_and_session_refresh_follow_language(self):
        from PySide6.QtWidgets import QApplication
        from subkit_app.session import EpisodeSession
        from subkit_app.views.group_editor import GroupEditor, RubyLine

        QApplication.instance() or QApplication([])
        project = Project.create(self.path / "ui", "Languages")
        for lang, lane in (("en", "original"), ("ja", "translation"), ("ja", "original")):
            m = member("迷宮都市", lang, lane)
            ep = project.add_episode(lang + lane)
            ep.document = document(m)
            session = EpisodeSession(project, ep)
            editor = GroupEditor()
            try:
                with patch.object(session, "request_preview"), patch.object(session, "_reannotate") as refresh:
                    session.editor.replace_text(m.id, "迷宮都市の朝")
                    self.assertEqual(refresh.call_count, int(lang == "ja"))
                editor.set_group(ep.document, ep.document.groups[0], None, None)
                rows = [r for r in editor.findChildren(RubyLine) if r.member is not None and r.kind == "ruby"]
                self.assertEqual(len(rows), int(lang == "ja"))
            finally:
                editor.close()
                session.close()

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires real ffmpeg/libass")
    def test_ruby_export_and_cache_use_language_in_both_lanes(self):
        from subkit.core.export import ExportCache, build_ass

        project = Project.create(self.path / "export", "Languages")
        m = member("迷宮都市", "ja", "translation")
        doc = document(m)
        annotate_member(doc, m, ReadingEngine())
        cache = ExportCache()
        def export():
            return build_ass(doc, {}, project.template(), self.path, cache=cache,
                             measure_cache=self.path / "measure.sqlite3")
        self.assertGreater(export()[1].ruby_lines, 0)
        self.assertEqual(export()[1].laid_out, 0)
        m.lang = "en"
        self.assertEqual(export()[1].ruby_lines, 0)
        m.lang, m.lane = "ja", "original"
        self.assertGreater(export()[1].ruby_lines, 0)

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires real ffmpeg/libass")
    @legacy_ruby_layout()
    def test_soma_export_bytes_and_merge_document_match_pre_c3(self):
        path = self.path / "soma"
        shutil.copytree(ROOT / "test_projects/soma", path, copy_function=shutil.copyfile)
        project = Project.open(path)
        # Historical byte fingerprints use the legacy layout; X11 defaults to width limits.
        project.data.setdefault("settings", {}).setdefault("ruby", {})["max_width"] = {
            "dialogue": {"enabled": False}, "song": {"enabled": False}}
        ep = project.episode(project.episode_ids()[0])
        ass, _ = ep.build_export(name_format="label")
        ass.save(self.path / "before.ass")
        self.assertEqual(hashlib.sha256((self.path / "before.ass").read_bytes()).hexdigest(),
                         "f4e1d14b93d23e27ef2de064cc075cc0c8dc0b252aa938c157bf455e4b45af8e")  # 0.20.0: song translation lines one layer above the original (was 0.10.8)
        count = itertools.count(1)
        def ids(prefix):
            return prefix + format(next(count), "020x")
        with patch("subkit.core.merge.new_id", side_effect=ids), patch("subkit.core.readings.new_id", side_effect=ids):
            doc = ep.merge()
        # D1a adds exact character links and bumps only those members' revisions.
        # Keep the C3 fingerprint for every other field, and verify the new fields.
        baseline = document_to_json(doc)
        self.assertEqual(baseline.pop("coordinate_playres"), [1920, 1080])
        book = project.character_book()
        linked = 0
        for m in baseline["members"].values():
            if not m.get("speaker"):
                continue
            expected = book.resolve(m["speaker"]["label"])
            self.assertEqual(m["speaker"]["character"], expected)
            if expected is not None:
                self.assertEqual(m.pop("rev"), 1)
                m["speaker"]["character"] = None
                linked += 1
        self.assertGreater(linked, 0)
        raw = json.dumps(baseline, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.assertEqual(hashlib.sha256(raw).hexdigest(), "2be39bb0e885b3d0eccebf9024dbb985e239999c3102387fd4525b6d3094032e")
        ass, _ = ep.build_export(name_format="label")
        ass.save(self.path / "merged.ass")
        data = (self.path / "merged.ass").read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), "9dfc755c010b327f817a47dea1b8818ab4f2f731b73ca7e107df9f9a838abdcd")  # 0.20.0: song translation lines one layer above the original (was 0.10.8)
        print(f"C3 soma: unchanged document apart from {linked} D1a speaker links "
              f"({len(doc.groups)} groups/{len(doc.members)} members); "
              f"merged export {len(data)} bytes, SHA256={hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    unittest.main()
