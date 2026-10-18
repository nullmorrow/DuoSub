"""Pinyin export, shared preview cache and real libass geometry (Windows host)."""

from __future__ import annotations

import copy
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from subkit import templates as T
from subkit.ass.document import AssDocument
from subkit.ass.text import VisibleText
from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.core.layout_index import LayoutIndex
from subkit.core.ops import Editor
from subkit.core.pinyin_notes import annotate_pinyin
from subkit.core.workspace import Project
from subkit.model.document import Document, Group, Member, Run
from subkit.proc import run


def document(text="银行", *, lang="zh-Hans", lane="translation", style="CN"):
    m = Member("m_1", "g_1", lang, lane, "dialogue", 0, 3000, layer=1, style=style,
               runs=[Run(text, ins="i_1", b=len(text))])
    doc = Document("E1", groups=[Group("g_1", members=[m.id])], members={m.id: m})
    annotate_pinyin(doc, m)
    return doc


def pinyin_rows(ass, report):
    return [(event, owner) for event, owner in zip(ass.events, report.owners) if owner["kind"] == "pinyin"]


@unittest.skipUnless(shutil.which("ffmpeg"), "requires real ffmpeg/libass")
class PinyinExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="c1b-export-")
        cls.path = Path(cls.tmp.name)
        fonts = T.FontSet("test", "test", "Microsoft YaHei", "Microsoft YaHei", "Arial", pinyin_font="Arial")
        cls.styles = T.build_styles(fonts, T.palette("classic"))
        cls.template = AssDocument.parse(cls.styles)
        cls.measure_cache = cls.path / "render.sqlite3"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def export(self, doc, mode="PY", *, template=None, cache=None, **kw):
        return build_ass(doc, {}, template or self.template, self.path, ExportOptions(pinyin_mode=mode, **kw),
                         measure_cache=self.measure_cache, cache=cache)

    def test_modes_convert_stored_readings_without_mutating_them(self):
        doc = document("行绿了")
        ed = Editor(doc, {})
        for ann, reading in zip(doc.annotations.values(), ("hang2", "lv4", "le")):
            ed.set_reading(ann.id, reading)
        saved = copy.deepcopy(doc.annotations)
        expected = {"PY": ["háng", "lǜ", "le"], "ZY": ["ㄏㄤˊ", "ㄌㄩˋ", "ㄌㄜ˙"], "WG": ["hang2", "lü4", "lê"]}
        for mode, syllables in expected.items():
            with self.subTest(mode=mode):
                ass, rep = self.export(doc, mode)
                self.assertEqual([VisibleText(e.text).text for e, _ in pinyin_rows(ass, rep)], syllables)
                self.assertEqual(rep.pinyin_lines, 3)
                self.assertEqual(rep.ruby_lines, 0)
                self.assertEqual(doc.annotations, saved)

    def test_language_controls_both_lanes_and_template_style(self):
        for lang, lane in [("zh-Hans", "original"), ("zh-Hant", "translation")]:
            with self.subTest(lang=lang, lane=lane):
                ass, rep = self.export(document(lang=lang, lane=lane))
                self.assertEqual(rep.pinyin_lines, 2)
                self.assertTrue(all(e.style == "CN.Pinyin" for e, _ in pinyin_rows(ass, rep)))
        doc = document()
        for lang in ("ja", "en"):
            doc.members["m_1"].lang = lang
            self.assertEqual(self.export(doc)[1].pinyin_lines, 0)

    def test_hidden_empty_comments_and_untimed_are_not_drawn(self):
        doc = document("银行了")
        anns = list(doc.annotations.values())
        anns[0].status = "hidden"
        anns[1].text = ""
        ass, rep = self.export(doc)
        self.assertEqual([o["ann"] for _, o in pinyin_rows(ass, rep)], [anns[2].id])
        for role, start, end in [("comment", 0, 3000), ("dialogue", 1000, 1000), ("dialogue", -2000, -1000)]:
            doc.members["m_1"].role = role
            doc.members["m_1"].start, doc.members["m_1"].end = start, end
            self.assertEqual(self.export(doc, include_source_comments=True)[1].pinyin_lines, 0)

    def test_cache_edit_hide_mode_and_language_invalidation(self):
        doc = document()
        cache = ExportCache()
        before, _ = self.export(doc, cache=cache)
        hit, rep = self.export(doc, cache=cache)
        self.assertEqual(rep.laid_out, 0)
        self.assertEqual(hit.serialize(), before.serialize())
        ed = Editor(doc, {})
        ann = next(a for a in doc.annotations.values() if a.base == "行")
        ed.set_reading(ann.id, "xing2")
        changed, rep = self.export(doc, cache=cache)
        self.assertEqual(rep.laid_out, 1)
        self.assertIn("xíng", changed.serialize())
        zy, rep = self.export(doc, "ZY", cache=cache)
        self.assertEqual(rep.laid_out, 1)
        self.assertIn("ㄒㄧㄥˊ", zy.serialize())
        ed.hide_annotation(ann.id)
        self.assertEqual(self.export(doc, "ZY", cache=cache)[1].pinyin_lines, 1)
        self.assertEqual(self.export(doc, "off", cache=cache)[1].pinyin_lines, 0)
        doc.members["m_1"].lang = "ja"
        self.assertEqual(self.export(doc, cache=cache)[1].pinyin_lines, 0)

    def test_style_fallback_missing_style_and_cached_style_restoration(self):
        doc = document(style="CN.Top")
        cache = ExportCache()
        ass, rep = self.export(doc, cache=cache)
        self.assertEqual(rep.pinyin_lines, 2)
        self.assertIsNotNone(ass.style("CN.Top.Pinyin"))
        self.assertTrue(any("CN.Top.Pinyin" in w and "使用 CN.Pinyin" in w for w in rep.warnings))
        cached, rep2 = self.export(doc, cache=cache)
        self.assertEqual(cached.serialize(), ass.serialize())
        self.assertEqual(rep2.warnings, rep.warnings)
        custom = copy.deepcopy(self.template)
        st = custom.style("CN.Pinyin").copy("CN.Top.Pinyin")
        st.set("Fontsize", 19)
        custom.styles.append(st)
        ass, rep = self.export(doc, template=custom)
        self.assertEqual(ass.style("CN.Top.Pinyin").fontsize, 19)
        self.assertFalse(rep.warnings)
        custom.styles = [s for s in custom.styles if not s.name.endswith(".Pinyin")]
        ass, rep = self.export(doc, template=custom)
        self.assertEqual(rep.pinyin_lines, 0)
        self.assertTrue(any("未导出拼音" in w for w in rep.warnings))

    def test_geometry_and_hit_keep_pinyin_owner_with_transforms(self):
        for overrides in ({}, {"raw": r"\frz15"}):
            with self.subTest(overrides=overrides):
                doc = document("行")
                doc.members["m_1"].overrides = overrides
                ass, rep = self.export(doc)
                if overrides:
                    self.assertIn(r"\frz15", pinyin_rows(ass, rep)[0][0].text)
                index = LayoutIndex(ass, rep.owners, measure_cache=self.measure_cache)
                try:
                    row_indices = [i for i, o in enumerate(rep.owners) if o["kind"] == "pinyin"]
                    geoms = index.geometry(row_indices)
                    self.assertEqual(len(geoms), 1)
                    geom = geoms[0]
                    self.assertEqual((geom.kind, geom.ann), ("pinyin", next(iter(doc.annotations))))
                    self.assertEqual(index.hit(geoms, geom.rect.cx, geom.rect.cy).line.ann, geom.ann)
                finally:
                    index.close()

    def test_release_and_preview_entry_points_use_project_mode(self):
        from PySide6.QtWidgets import QApplication
        from subkit_app.session import EpisodeSession, PreviewJob

        QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory(prefix="c1b-project-") as tmp:
            project = Project.create(Path(tmp), "Pinyin", template_text=self.styles)
            episode = project.add_episode("E1")
            episode.document = document()
            project.set_pinyin_mode("WG")
            ass, rep = episode.build_export()
            self.assertEqual(rep.pinyin_lines, 2)
            self.assertIn("hang2", ass.serialize())
            session = EpisodeSession(project, episode)
            try:
                job = PreviewJob(session, 0, Path(tmp) / "preview.ass")
                done, errors = [], []
                job.signals.done.connect(lambda *args: done.append(args))
                job.signals.failed.connect(lambda *args: errors.append(args))
                job.run()
                self.assertFalse(errors, errors)
                self.assertEqual(len(done), 1)
                self.assertEqual(done[0][3].pinyin_lines, 2)
                self.assertEqual([e.text for e, _ in pinyin_rows(done[0][2], done[0][3])],
                                 [e.text for e, _ in pinyin_rows(ass, rep)])
            finally:
                session.close()

    def test_libass_positions_match_template_scene(self):
        text = T.preview_sentence("zh-Hans", 0)[0]
        ass, rep = self.export(document(text))
        rows = pinyin_rows(ass, rep)
        expected = AssDocument.parse(T._HEAD + self.styles + T._EVH +
                                     "\n".join(T.scene_events(self.styles, "dialogue", pinyin=True,
                                                            measure_cache=self.measure_cache)))
        expected.events = [e for e in expected.events if e.style == "CN.Pinyin"]
        self.assertEqual(len(rows), len(expected.events))
        deltas = []
        for (event, owner), preview in zip(rows, expected.events):
            x, y = map(float, re.search(r"\\pos\(([^,]+),([^)]+)\)", preview.text).groups())
            deltas.append(max(abs(owner["x"] - x), abs(owner["y"] - y)))
            self.assertEqual(VisibleText(event.text).text, VisibleText(preview.text).text)
        self.assertLessEqual(max(deltas), 3.0)
        ass.events = [e for e, _ in rows]
        frames, boxes, logs = [], [], []
        for name, doc in (("export", ass), ("template", expected)):
            doc.save(self.path / f"{name}.ass")
            result = run(["ffmpeg", "-v", "info", "-f", "lavfi", "-i", "color=black:s=1920x1080:d=2",
                          "-vf", f"ass={name}.ass", "-ss", "1", "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "gray", "pipe:1"], cwd=self.path, capture_output=True, check=True)
            frame = np.frombuffer(result.stdout, dtype=np.uint8).reshape(1080, 1920)
            yy, xx = np.where(frame > 8)
            self.assertGreater(len(xx), 0)
            boxes.append((int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())))
            frames.append(frame)
            logs.append(result.stderr.decode("utf-8", errors="replace"))
        self.assertLessEqual(max(abs(a-b) for a, b in zip(*boxes)), 3)
        self.assertTrue(np.array_equal(*frames))
        versions = [line for line in logs[0].splitlines() if "ffmpeg version" in line or "libass API version" in line]
        print(f"C1b libass 1920x1080: {versions}; {len(rows)} pinyin rows; export/template ink boxes={boxes}; "
              f"max anchor delta={max(deltas):.3f}px; differing pixels={np.count_nonzero(frames[0] != frames[1])}")

    def test_soma_copy_off_preserves_bytes_with_retained_pinyin(self):
        with tempfile.TemporaryDirectory(prefix="c1b-soma-test-") as tmp:
            path = Path(tmp) / "soma"
            shutil.copytree(ROOT / "test_projects/soma", path, copy_function=shutil.copyfile)
            project = Project.open(path)
            episode = project.episode(project.episode_ids()[0])
            before, _ = episode.build_export()
            editor = Editor(episode.document, episode.sources)
            editor.set_pinyin_mode(project, "PY")
            self.assertTrue(any(a.kind == "pinyin" for a in episode.document.annotations.values()))
            editor.set_pinyin_mode(project, "off")
            after, rep = episode.build_export()
            before.save(Path(tmp) / "before.ass")
            after.save(Path(tmp) / "after.ass")
            self.assertEqual((Path(tmp) / "after.ass").read_bytes(), (Path(tmp) / "before.ass").read_bytes())
            self.assertEqual(rep.pinyin_lines, 0)


if __name__ == "__main__":
    unittest.main()
