"""Hide/mark transactions, common render path, neighbours and saved-project review."""
from __future__ import annotations

import copy
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tests import test_convert_ui as ui
from tests.test_autotime_doc import document, mapping
from subkit.ass import AssDocument
from subkit.core.autotime_doc import Settings, _clamp_neighbours, estimation_intervals, plan
from subkit.core.batch import BatchOptions, adopt_suggestions
from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.core.ops import Editor
from subkit.core.review_list import collect_review, write_review
from subkit.model.serialize import load_document, save_document
from subkit.styles_default import default_styles_ass
from subkit_app.views.group_table import C_TAGS, GroupFilter, GroupTableModel
from subkit_app.views.waveform import WaveformView
from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QDialog, QFileDialog, QInputDialog, QMessageBox


class HideMarkCore(unittest.TestCase):
    def setUp(self):
        self.doc = document(3)
        self.editor = Editor(self.doc, {})
        self.path = Path(tempfile.mkdtemp(prefix="hide_mark_"))

    def test_multiselect_undo_redo_preserves_every_member(self):
        members = copy.deepcopy(self.doc.members)
        self.doc.groups[0].flags = ["order_exception"]
        self.editor.hide_many(["g0", "g1", "g0", "missing"])
        self.assertEqual([g.id for g in self.doc.groups if "hidden" in g.flags], ["g0", "g1"])
        self.editor.history.undo()
        self.assertFalse(any("hidden" in g.flags for g in self.doc.groups))
        self.editor.history.redo()
        self.assertIn("order_exception", self.doc.groups[0].flags)
        self.editor.mark_many(["g0", "g1"])
        self.editor.set_mark_note(["g0", "g1"], "复查 <&>")
        self.editor.history.undo()
        self.assertNotIn("mark_note", self.doc.groups[0].extra)
        self.editor.history.redo()
        self.assertEqual(self.doc.groups[1].extra["mark_note"], "复查 <&>")
        self.assertEqual(self.doc.members, members)
        save_document(self.doc, self.path / "document.json")
        restored = load_document(self.path / "document.json")
        self.assertEqual(restored.groups, self.doc.groups)
        self.assertEqual(restored.members, members)
        self.editor.hide_many(["g0", "g1"], False)
        self.editor.mark_many(["g0", "g1"], False)
        self.editor.set_mark_note(["g0", "g1"], "")
        self.assertFalse(any("marked" in g.flags or "hidden" in g.flags or "mark_note" in g.extra for g in self.doc.groups))
        self.editor.history.undo()
        self.assertEqual(self.doc.groups[0].extra["mark_note"], "复查 <&>")
        self.editor.history.undo()
        self.assertIn("marked", self.doc.groups[0].flags)
        self.editor.history.undo()
        self.assertIn("hidden", self.doc.groups[0].flags)

    def test_render_skip_and_byte_identical_restore_with_cache(self):
        template = AssDocument.parse(default_styles_ass())
        cache = ExportCache()
        for mode in ("preview", "release"):
            opts = ExportOptions(mode=mode, ruby=False)
            before, _ = build_ass(self.doc, {}, template, self.path, opts, cache=cache)
            before.save(self.path / "before.ass")
            original = (self.path / "before.ass").read_bytes()
            self.editor.hide_many(["g0", "g1"])
            hidden, report = build_ass(self.doc, {}, template, self.path, opts, cache=cache)
            self.assertEqual(report.hidden_groups, 2)
            self.assertEqual(len(hidden.events), 2)
            self.assertEqual({o["group"] for o in report.owners}, {"g2"})
            self.editor.hide_many(["g0", "g1"], False)
            self.editor.mark_many(["g0"])
            restored, _ = build_ass(self.doc, {}, template, self.path, opts, cache=cache)
            restored.save(self.path / "restored.ass")
            self.assertEqual((self.path / "restored.ass").read_bytes(), original)

    def test_autotime_hidden_even_include_protected(self):
        self.editor.hide_many(["g1"])
        self.assertEqual(len(estimation_intervals(self.doc, {})), 4)
        result = plan(self.doc, {}, None, Settings(include_protected=True), mapping_override=mapping())
        self.assertEqual({g.group_id for g in result.changes}, {"g0", "g2"})
        self.editor.apply_autotime(result)
        self.assertEqual(self.doc.members["original1"].start, 31000)

    def test_neighbour_overlap_and_waveform_ignore_hidden(self):
        self.editor.hide_many(["g1"])
        mapped = {"g0": [(0, 100)], "g1": [(110, 120)], "g2": [(300, 400)]}
        planned = {"g0": [(0, 200)], "g1": [(110, 120)], "g2": [(300, 400)]}
        _clamp_neighbours(self.doc, mapped, planned)
        self.assertEqual(planned["g0"], [(0, 200)])
        neighbours = WaveformView._hint_neighbours(SimpleNamespace(doc=self.doc), self.doc.groups[0])
        self.assertEqual(neighbours, (None, self.doc.group_time(self.doc.groups[2])))
        result = WaveformView._compute_group_hints(SimpleNamespace(), self.doc.groups[1])
        self.assertIsNone(result["suggested"])

    def test_batch_suggestions_skip_hidden_neighbour(self):
        self.editor.hide_many(["g1"])
        features = SimpleNamespace(snap_activity=object())
        with patch("subkit.core.batch.suggest_group_times", return_value=None) as suggest:
            adopt_suggestions(self.editor, features, BatchOptions())
        self.assertEqual(suggest.call_count, 2)
        self.assertEqual(suggest.call_args_list[0].kwargs["following"], self.doc.group_time(self.doc.groups[2]))

    def test_review_reads_two_saved_episodes_escaped_html_bom_csv_and_empty(self):
        project = SimpleNamespace(root=self.path, data={"episodes": []})
        self.assertEqual(collect_review(project), [])
        self.assertEqual(write_review([], self.path / "empty.html"), 0)
        self.assertFalse((self.path / "empty.html").exists())
        for eid in ("E01", "E02"):
            directory = self.path / eid
            directory.mkdir()
            project.data["episodes"].append({"id": eid, "title": "标题 <&>", "path": eid})
            self.doc.groups[1].flags = ["marked", "hidden"]
            self.doc.groups[1].extra["mark_note"] = '待复查 <script> & "备注"'
            self.doc.members["translation1"].runs[0].text = '翻译 <script>alert(1)</script>&'
            save_document(self.doc, directory / "document.json")
        episodes = collect_review(project)
        self.assertEqual(len(episodes), 2)
        self.assertEqual(write_review(episodes, self.path / "review.html"), 2)
        page = (self.path / "review.html").read_text(encoding="utf-8")
        self.assertIn("&lt;script&gt;", page)
        self.assertNotIn("<script>", page)
        self.assertIn("prefers-color-scheme:dark", page)
        self.assertTrue((self.path / "review.csv").read_bytes().startswith(b"\xef\xbb\xbf"))
        with (self.path / "review.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1][0], "E01")
        self.assertEqual(rows[2][0], "E02")
        self.assertEqual(rows[1][-2], "标记+隐藏")
        self.assertIn("<script>", rows[1][-1])


class HideMarkUI(unittest.TestCase):
    setUpClass = ui.ConversionUI.__dict__["setUpClass"]
    setUp = ui.ConversionUI.setUp
    _dispose = ui.ConversionUI._dispose
    group = ui.ConversionUI.group
    select = ui.ConversionUI.select
    context_menu = ui.ConversionUI.context_menu

    def test_buttons_filters_badges_tooltip_and_undo(self):
        self.group("a")
        self.group("b")
        self.window.select_group("a")
        self.window._on_action("hidden")
        self.window._on_action("marked")
        self.session.editor.set_mark_note(["a"], "检查这个字")
        self.window._flush_group_selection()
        self.window.select_group("a")
        self.assertTrue(self.window.editor.review_buttons["hidden"].isChecked())
        self.assertTrue(self.window.editor.review_buttons["marked"].isChecked())
        model = self.window.model
        tags = model.data(model.index(0, C_TAGS))
        self.assertIn("已隐藏", tags)
        self.assertIn("★ 标记", tags)
        self.assertIn("检查这个字", model.data(model.index(0, C_TAGS), Qt.ItemDataRole.ToolTipRole))
        self.assertEqual(model.data(model.index(0, C_TAGS), Qt.ItemDataRole.ForegroundRole).color().name(), "#8c959f")
        for mode in ("只看已标记", "只看已隐藏"):
            self.window.proxy.set_mode(mode)
            self.assertEqual(self.window.proxy.rowCount(), 1)
        self.window.proxy.set_mode("全部")
        self.window.select_group("b")
        self.assertFalse(self.window.editor.review_buttons["hidden"].isChecked())
        self.assertFalse(self.window.editor.review_buttons["marked"].isChecked())

    def test_note_dialog_multiselect_and_help(self):
        self.group("a")
        self.group("b")
        def accepted(dialog):
            self.assertEqual(dialog.help_anchor, "hide-mark")
            dialog.setTextValue("复查")
            return QDialog.DialogCode.Accepted
        with patch.object(QInputDialog, "exec", accepted):
            self.window._mark_note_dialog(["a", "b"])
        self.assertEqual([g.extra["mark_note"] for g in self.doc.groups], ["复查", "复查"])
        self.session.editor.history.undo()
        self.assertFalse(any(g.extra.get("mark_note") for g in self.doc.groups))

    def test_right_click_multiselect_flags_and_help(self):
        self.group("a")
        self.group("b")
        self.select("a", "b")
        self.context_menu("隐藏（不显示、不导出）")
        self.assertTrue(all("hidden" in g.flags for g in self.doc.groups))
        self.session.editor.history.undo()
        self.assertFalse(any("hidden" in g.flags for g in self.doc.groups))
        self.select("a", "b")
        self.context_menu("标记待复查")
        self.assertTrue(all("marked" in g.flags for g in self.doc.groups))
        actions = self.context_menu()
        note = next(a for a in actions if a.text() == "编辑标记备注…")
        self.assertEqual(note.property("help_anchor"), "hide-mark")

    def test_review_export_current_save_and_browser_prompt(self):
        self.group("a", text="复查 <&>")
        self.project.root = self.path
        self.project.data["episodes"] = [{"id": "E01", "title": "标题", "path": "E01"}]
        directory = self.path / "E01"
        directory.mkdir()
        self.session.editor.mark_many(["a"])
        target = self.path / "list.html"
        def save_current():
            save_document(self.doc, directory / "document.json")
        def choose(dialog):
            self.assertEqual(dialog.help_anchor, "hide-mark")
            return QDialog.DialogCode.Accepted
        with patch.object(self.session, "save", side_effect=save_current) as save, \
                patch.object(QFileDialog, "exec", choose), \
                patch.object(QFileDialog, "selectedFiles", return_value=[str(target)]), \
                patch.object(self.window, "_help_message", return_value=QMessageBox.StandardButton.Yes) as prompt, \
                patch.object(QDesktopServices, "openUrl", return_value=True) as browser:
            self.window._export_review_list()
        save.assert_called_once()
        self.assertTrue(target.exists())
        self.assertTrue(target.with_suffix(".csv").exists())
        self.assertEqual(prompt.call_args.args[-1], "用浏览器打开吗？")
        self.assertEqual(Path(browser.call_args.args[0].toLocalFile()), target.resolve())

    def test_review_empty_saves_current_first(self):
        self.project.root = self.path
        self.project.data["episodes"] = []
        with patch.object(self.session, "save") as save, patch.object(self.window, "_help_message") as message:
            self.window._export_review_list()
        save.assert_called_once()
        self.assertEqual(message.call_args.args[-1], "工程里没有标记或隐藏的句子")
