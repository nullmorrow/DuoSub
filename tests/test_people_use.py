"""D1b: book-backed readings, language names, review persona and soma fidelity."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.core.ops import Editor
from subkit.core.readings import annotate_document, suggest_readings
from subkit.core.review import build_packet
from subkit.core.workspace import Project
from subkit.model.document import Document, Group, Member, Run
from subkit.model.serialize import document_to_json
from subkit.reading.characters import CharacterTable
from subkit.reading.dictionary import ReadingDict
from subkit.reading.engine import ReadingEngine


def member(mid, text="私と俺", *, lang="ja", cid=None, label="太郎"):
    return Member(mid, "g_1", lang, "original" if lang == "ja" else "translation", "dialogue", 1000, 2000,
                  style="JP" if lang == "ja" else "CN", speaker={"character": cid, "label": label},
                  runs=[Run(text, ins="i_" + mid, b=len(text))])


def document(*members):
    return Document("E1", groups=[Group("g_1", members=[m.id for m in members])],
                    members={m.id: m for m in members})


def annotations(doc):
    return sorted((a["member"], json.dumps({k: v for k, v in a.items() if k != "id"},
                                          ensure_ascii=False, sort_keys=True))
                  for a in document_to_json(doc)["annotations"].values())


class PeopleUseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="d1b-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.project = Project.create(self.root / "project", "人物")
        self.book = self.project.character_book()
        self.ch = self.book.add_character({"ja": "太郎", "ja-kana": "たろう", "zh-Hans": "太郎简", "en": "Taro"},
                                          self_refs=[{"surface": "私", "reading": "わたくし", "scope": {}}])

    def engine(self, episode="E1", season="S1"):
        return self.project.reading_engine(episode=episode, season=season)

    def test_scoped_self_references_precedence_is_per_surface_not_list_order(self):
        refs = [{"surface": "私", "reading": reading, "scope": scope} for reading, scope in
                [("あたし", {"line": "m_line"}), ("わし", {"episodes": ["E1", "E2"]}),
                 ("わたし", {"season": "S1"}), ("わたくし", {})]]
        refs.append({"surface": "俺", "reading": "おいら", "scope": {}})
        self.book.update_character(self.ch.id, self_refs=refs)
        for mid, episode, season, expected in [("m_line", "E1", "S1", "あたし"),
                                               ("m_episode", "E2", "S1", "わし"),
                                               ("m_season", "E3", "S1", "わたし"),
                                               ("m_all", "E3", "S2", "わたくし")]:
            with self.subTest(mid=mid):
                out = suggest_readings(member(mid, cid=self.ch.id), self.engine(episode, season))
                by_base = {a.base: a for a in out}
                self.assertEqual(by_base["私"].text, expected)
                self.assertEqual(by_base["俺"].text, "おいら")
                self.assertEqual(by_base["私"].origin, "dict")

    def test_no_label_guess_for_unlinked_crowd_not_person_or_missing_id(self):
        baseline = {a.base: a.text for a in suggest_readings(member("m", cid=None), ReadingEngine())}
        for cid, label in [(None, "太郎"), ("crowd", "太郎"), ("chr_missing", "太郎"), (None, "効果音")]:
            with self.subTest(cid=cid, label=label):
                actual = {a.base: a.text for a in suggest_readings(member("m", cid=cid, label=label), self.engine())}
                self.assertEqual(actual, baseline)
        out = suggest_readings(member("m", cid=self.ch.id, label="別人"), self.engine())
        self.assertEqual(next(a.text for a in out if a.base == "私"), "わたくし")

    def test_episode_context_and_new_json_name_parts_and_project_words_win(self):
        ep = self.project.add_episode("E1", season="S1")
        self.book.add_character({"ja": "山田 花子", "ja-kana": "やまだ はなこ"})
        self.book.add_character({"ja": "司", "ja-kana": "つかさ"})
        self.project.dict_path("readings").write_text(json.dumps({"words": {"山田": "やまなか"}}, ensure_ascii=False), encoding="utf-8")
        engine = ep.reading_engine()
        self.assertEqual((engine.episode, engine.season), ("E1", "S1"))
        entries = engine.analyze("山田花子 花子 山田 寿司")
        full = next(e for e in entries if e.surface == "山田花子")
        self.assertEqual([p.reading for p in full.parts], ["やまだ", "はなこ"])
        self.assertTrue(full.locked)
        self.assertEqual(next(e.reading for e in entries if e.surface == "山田"), "やまなか")
        self.assertNotIn("司", engine.dict.words)

    def test_renamed_converted_names_do_not_use_stale_parts(self):
        self.book.update_character(self.ch.id, name_parts=["太郎"], kana_parts=["たろう"])
        self.book.update_character(self.ch.id, names={"ja": "花子", "ja-kana": "はなこ"})
        self.assertEqual(self.engine().dict.words["花子"].reading, "はなこ")
        self.assertNotIn("太郎", self.engine().dict.words)

    def test_existing_book_ignores_later_tsv_changes(self):
        self.project.dict_path("characters").write_text("太郎\tべつよみ\t\t\tあたし\n", encoding="utf-8")
        engine = self.engine()
        self.assertEqual(engine.dict.words["太郎"].reading, "たろう")
        self.assertEqual(next(a.text for a in suggest_readings(member("m", text="私", cid=self.ch.id), engine)), "わたくし")

    def test_book_edit_refreshes_only_auto_and_one_undo_restores_document_file(self):
        ms = [member(f"m{i}", text="私", cid=self.ch.id) for i in range(5)]
        doc = document(*ms)
        annotate_document(doc, {}, self.engine())
        for m, status, origin in zip(ms, ["confirmed", "hidden", "auto", "auto"],
                                     ["dict", "dict", "manual", "exception"]):
            a = next(a for a in doc.annotations.values() if a.member == m.id)
            a.status, a.origin, a.text = status, origin, "保留"
        before = copy.deepcopy(document_to_json(doc))
        raw = self.book.path.read_bytes()
        editor = Editor(doc, {}, character_book=self.book, reading_engine_factory=self.engine)
        editor.edit_characters(lambda book: book.update_character(self.ch.id,
                                self_refs=[{"surface": "私", "reading": "あたし", "scope": {}}]))
        self.assertEqual(len(editor.history.done), 1)
        self.assertEqual([a.text for a in doc.annotations.values() if a.member != ms[-1].id], ["保留"] * 4)
        self.assertEqual(next(a.text for a in doc.annotations.values() if a.member == ms[-1].id), "あたし")
        after = annotations(doc)
        editor.history.undo()
        self.assertEqual(annotations(doc), annotations_from_json(before))
        self.assertEqual(self.book.path.read_bytes(), raw)
        editor.history.redo()
        self.assertEqual(annotations(doc), after)

    def test_speaker_pick_and_relink_refresh_and_undo_without_changing_raw_label(self):
        m = member("m", text="私")
        doc = document(m)
        annotate_document(doc, {}, self.engine())
        before = annotations(doc)
        editor = Editor(doc, {}, character_book=self.book, reading_engine_factory=self.engine)
        editor.set_speaker([m.id], self.ch.id, whole_group=False)
        self.assertEqual(next(iter(doc.annotations.values())).text, "わたくし")
        self.assertEqual(m.speaker["label"], "太郎")
        editor.history.undo()
        self.assertEqual(annotations(doc), before)
        editor.relink_speakers()
        self.assertEqual(next(iter(doc.annotations.values())).text, "わたくし")

    def test_name_export_fallbacks_include_actor_and_cache_on_rename(self):
        self.book.update_character(self.ch.id, names={"ja": "太郎", "zh-Hant": "太郎繁", "zh": "太郎中", "en": "Taro"})
        ms = [member(str(i), text="hello", lang=lang, cid=self.ch.id) for i, lang in
              enumerate(["ja", "zh-Hans", "zh-Hant", "zh", "en", "fr"])]
        ms += [member("unknown", text="hello", lang="en", label="原标签"),
               member("crowd", text="hello", lang="en", cid="crowd", label="群众")]
        doc = document(*ms)
        cache = ExportCache()
        def export(include_actor=True):
            return build_ass(doc, {}, self.project.template(), self.root, ExportOptions(ruby=False, include_actor=include_actor),
                             character_book=self.book, cache=cache)
        out, _ = export()
        self.assertEqual([e.name for e in out.events], ["太郎", "太郎繁", "太郎繁", "太郎中", "Taro", "太郎", "原标签", "群众"])
        self.assertEqual(export()[1].laid_out, 0)
        self.book.rename_character(self.ch.id, "en", "New Taro")
        out, report = export()
        self.assertEqual(report.laid_out, 1)
        self.assertEqual(out.events[4].name, "New Taro")
        self.assertTrue(all(e.name == "" for e in export(False)[0].events))
        self.book.update_character(self.ch.id, names={"zh": "通用名"})
        self.assertEqual(self.book.display_name(self.ch.id, "zh-Hans", "原"), "通用名")
        self.assertEqual(self.book.display_name(self.ch.id, "en", "原"), "通用名")
        self.book.update_character(self.ch.id, names={"zh-Hans": "简体名", "zh-Hant": "繁體名", "ja": "日本名"})
        self.assertEqual(self.book.display_name(self.ch.id, "zh-Hans"), "简体名")
        self.assertEqual(self.book.display_name(self.ch.id, "zh-Hant"), "繁體名")
        self.assertEqual(self.book.display_name(self.ch.id, "fr", "原标签"), "日本名")

    def test_episode_export_passes_book_and_review_has_all_persona_and_scopes(self):
        persona = dict(zip(["self", "address", "catchphrases", "tone", "note_ja", "note_tr"],
                           ["俺", "君", "行くぞ", "丁寧", "原注|二\n行", "译注"]))
        self.book.update_character(self.ch.id, persona=persona)
        ja, en = member("ja", text="私", cid=self.ch.id), member("en", text="hello", lang="en", cid=self.ch.id)
        doc = document(ja, en, member("unknown", text="私", label="太郎"))
        ep = self.project.add_episode("E1")
        ep.document = doc
        with patch("subkit.core.export.build_ass", return_value=(None, None)) as build:
            ep.build_export()
            self.assertEqual(build.call_args.kwargs["character_book"].to_json(), self.book.to_json())
        md, jl = build_packet(doc, episode="E1", chars=self.book, trans_lang="en")
        for value in ["君", "行くぞ", "丁寧", "原注\\|二<br>行", "译注", "私=わたくし", "Taro"]:
            self.assertIn(value, md)
        rec = json.loads(jl.splitlines()[0])
        self.assertEqual(rec["original"][0]["speaker"], {"label": "太郎", "character": self.ch.id, "name": "太郎"})
        self.assertEqual(rec["translation"][0]["speaker"]["name"], "Taro")
        self.assertNotIn("character", rec["original"][1]["speaker"])

    def test_session_invalidates_cached_engine_after_book_edit_and_undo(self):
        from PySide6.QtWidgets import QApplication
        from subkit_app.session import EpisodeSession

        QApplication.instance() or QApplication([])
        ep = self.project.add_episode("E1")
        ep.document = document(member("m", text="私", cid=self.ch.id))
        session = EpisodeSession(self.project, ep)
        self.addCleanup(session.close)
        with patch.object(session, "request_preview"):
            session._reannotate(ep.document.members["m"])
            self.assertIsNotNone(session.engine)
            session.editor.edit_characters(lambda book: book.update_character(self.ch.id,
                                           self_refs=[{"surface": "私", "reading": "あたし", "scope": {}}]))
            self.assertIsNone(session.engine)
            self.assertEqual(next(iter(ep.document.annotations.values())).text, "あたし")
            session.editor.history.undo()
            self.assertEqual(next(iter(ep.document.annotations.values())).text, "わたくし")
            self.assertIsNone(session.engine)

    def test_soma_fresh_annotations_tsv_vs_book_are_identical(self):
        source = ROOT / "test_projects/soma"
        original = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in source.rglob("*") if p.is_file()}
        target = self.root / "soma"
        shutil.copytree(source, target, copy_function=shutil.copyfile)
        project = Project.open(target)
        episode = project.episode(project.episode_ids()[0])
        old_dict = ReadingDict.load(project.root / "dict/_no_global.json", project.dict_path("readings"))
        table = CharacterTable.load(project.dict_path("characters"))
        old_dict.merge_characters(table)
        self.assertTrue(set(table.words()).issubset(set(project.character_book().words())))
        old, new = copy.deepcopy(episode.document), copy.deepcopy(episode.document)
        old.annotations.clear()
        new.annotations.clear()
        project.character_book().link_document(new)
        annotate_document(old, episode.sources, ReadingEngine(old_dict))
        annotate_document(new, episode.sources, episode.reading_engine(new))
        a, b = annotations(old), annotations(new)
        differences = len(set(a) ^ set(b))
        print(f'D1b soma: {len(new.members)} members; {len(a)} TSV annotations / {len(b)} book annotations; diff={differences}; exact-link exceptions=[]')
        self.assertEqual(a, b)
        self.assertEqual(original, {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in source.rglob("*") if p.is_file()})


def annotations_from_json(data):
    from subkit.model.serialize import document_from_json
    return annotations(document_from_json(data))


if __name__ == "__main__":
    unittest.main()
