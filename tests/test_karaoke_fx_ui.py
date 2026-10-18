"""FX4: Qt offscreen/fake player, project-local presets and transactional scopes."""
import copy
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from tests import test_karaoke_ui as karaoke_fixture
from tests.test_passages import member, recorded, match, plan
from subkit.core import karaoke as kara
from subkit.core.ops import Editor
from subkit.core.export import ExportOptions, build_ass
from subkit.model.document import Document, Group
from subkit.karaoke_fx import REGISTRY
from subkit.karaoke_fx.presets import PresetStore
from subkit_app.views.karaoke_fx_panel import KaraokeFxPanel
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QCheckBox, QComboBox, QInputDialog, QMessageBox


class FxPanelTests(karaoke_fixture.KaraokeUI):
    # Reuse the fixture, without repeating inherited KR tests.
    def setUp(self):
        super().setUp()
        self.project.root = self.path

    def test_registry_schemas_controls_and_float_slider(self):
        panel = KaraokeFxPanel(PresetStore(self.path), self.window)
        self.addCleanup(panel.deleteLater)
        for name, fx in REGISTRY.items():
            panel.set_effect(name, {})
            self.assertEqual(set(panel.controls), set(fx.schema) | {"ktv", "line_advance_ms"})
            resolved = fx.resolve({k: v for k, v in panel.values().items() if k in fx.schema})
            self.assertEqual(resolved, fx.resolve())
        panel.set_effect("glow", {})
        panel.controls["glow_size"].setValue(4.25)
        self.assertEqual(panel.values()["glow_size"], 4.25)
        self.assertEqual(panel.windowFlags() & Qt.WindowType.WindowType_Mask, Qt.WindowType.Tool)
        self.assertFalse(panel.isModal())

    def test_live_refresh_translation_undo_and_debounce(self):
        self.line("o", gid="g", timed=True)
        m = self.line("t", gid="g", lane="translation", timed=True)
        self.window.wave_panel.target.setCurrentIndex(self.window.wave_panel.target.findData("translation"))
        self.window._k_effect("glow")
        self.window._k_parameters()
        panel = self.window._karaoke_fx_panel
        before = copy.deepcopy(kara.tracks_of(self.doc, m.id)[0]["beats"])
        self.session.request_preview = type(self.session).request_preview.__get__(self.session)
        count = len(self.session.editor.history.done)
        panel.controls["glow_size"].setValue(7.25)
        self.assertEqual(len(self.session.editor.history.done), count + 1)
        self.assertEqual(panel.effect, "glow")
        self.assertEqual(kara.tracks_of(self.doc, m.id)[0]["effect"]["params"]["glow_size"], 7.25)
        self.assertEqual(kara.tracks_of(self.doc, m.id)[0]["beats"], before)
        self.assertTrue(self.session._debounce.isActive())
        self.assertEqual(self.session._debounce.interval(), 120)
        self.session._debounce.stop()
        self.session.editor.history.undo()
        self.assertNotIn("glow_size", kara.tracks_of(self.doc, m.id)[0]["effect"]["params"])

    def test_selection_stays_selected_across_parameter_changes(self):
        a = self.line("a", gid="ga", timed=True)
        b = self.line("b", gid="gb", timed=True)
        for gid in ("ga", "gb"):
            index = self.window.proxy.mapFromSource(self.window.model.index(self.window.model.row_of(gid), 0))
            self.window.table.selectionModel().select(index, QItemSelectionModel.SelectionFlag.Select |
                                                      QItemSelectionModel.SelectionFlag.Rows)
        self.window._k_apply_fx("pop", {"scale": 170}, "selection")
        self.assertEqual(set(self.window.table.selected_gids()), {"ga", "gb"})
        self.window._k_apply_fx("pop", {"scale": 180}, "selection")
        for m in (a, b):
            self.assertEqual(kara.tracks_of(self.doc, m.id)[0]["effect"]["params"]["scale"], 180)

    def test_preset_buttons_toolbar_roundtrip_and_help(self):
        self.line(timed=True)
        self.window._k_effect("pop")
        self.window._k_parameters()
        panel = self.window._karaoke_fx_panel
        with patch.object(QInputDialog, "getText", return_value=("弹出测试", True)):
            panel.save()
        self.assertGreaterEqual(self.window.wave_panel.k_fx.findData("preset:弹出测试"), 0)
        self.window._k_effect("preset:弹出测试")
        with patch.object(QInputDialog, "getText", return_value=("新名称", True)):
            panel.rename()
        self.assertGreaterEqual(self.window.wave_panel.k_fx.findData("preset:新名称"), 0)
        self.assertEqual(PresetStore(self.path).list()[0]["name"], "新名称")
        panel.activateWindow()
        self.app.processEvents()
        with patch("subkit_app.help.open_manual") as opener:
            QTest.keyClick(panel, Qt.Key.Key_F1)
            opener.assert_called_once_with("karaoke-fx")
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            panel.delete()
        self.assertEqual(PresetStore(self.path).list(), [])
        self.assertEqual(next(iter(self.doc.karaoke.values()))["effect"]["preset"], "pop")

    def test_toolbar_multiselection_effect_and_preset_one_undo(self):
        members = [self.line("あい", gid=gid, lane=lane) for gid in ("ga", "gb")
                   for lane in ("original", "translation")]
        dialogue = self.line("台词", gid="gd")
        self.doc.group("gd").kind = "dialogue"
        self.window.select_group("ga")
        store = self.window._k_fx_store()
        store.save("弹出批量", "pop", {"scale": 170})
        self.window._k_presets_changed()
        for key, effect, label in (("ko", "ko", "描边先出"), ("preset:弹出批量", "pop", "弹出批量")):
            before = copy.deepcopy(self.doc.karaoke)
            count = len(self.session.editor.history.done)
            menu = next(a.menu() for a in self.window.wave_panel.more_menu.actions() if a.text() == "特效")
            menu.aboutToShow.emit()
            with patch.object(self.window.table, "selected_gids", return_value=["ga", "gb", "gd"]):
                next(a for a in menu.actions() if a.text() == label).trigger()
            self.assertEqual(len(self.session.editor.history.done), count + 1)
            for m in members:
                track = kara.tracks_of(self.doc, m.id)[0]
                self.assertEqual(track["effect"]["preset"], effect)
                self.assertEqual(track["beats"], next(t for t in before.values() if t["member"] == m.id)["beats"])
            self.assertEqual(kara.tracks_of(self.doc, dialogue.id)[0], next(t for t in before.values() if t["member"] == dialogue.id))
            self.assertEqual(self.window.statusBar().currentMessage(), f"已把特效「{label}」套到选中的 2 句（Ctrl+Z 撤销）")
            self.session.editor.history.undo()
            self.assertEqual(before, self.doc.karaoke)

    def test_group_context_menu_effects_presets_mixed_selection_and_no_song(self):
        from PySide6.QtGui import QContextMenuEvent
        from PySide6.QtWidgets import QMenu
        originals = [self.line("あい", gid=gid) for gid in ("ga", "gb", "gd")]
        originals[2].role = "dialogue"
        self.doc.group("gd").kind = "dialogue"
        self.window._k_fx_store().save("列表预设", "glow", {"glow_size": 7})
        self.window._k_presets_changed()
        self.window.select_group("gd")
        seen = []
        class MenuDouble(QMenu):
            def exec(menu, _pos):
                fx = next((a.menu() for a in menu.actions() if a.text() == "K 轴特效"), None)
                if fx:
                    seen.append((fx.help_anchor, [(a.text(), a.property("help_anchor")) for a in fx.actions()]))
                    return fx.actions()[-1]
                seen.append(None)
                return None
        table = self.window.table
        table.resize(1000, 400)
        pos = table.visualRect(table.currentIndex()).center()
        event = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, pos, table.mapToGlobal(pos))
        before = copy.deepcopy(self.doc.karaoke)
        with patch("subkit_app.views.group_table.QMenu", MenuDouble), patch.object(table, "selected_gids", return_value=["ga", "gb", "gd"]):
            table.contextMenuEvent(event)
        expected = ["不用特效" if key == "none" else fx.display_name for key, fx in REGISTRY.items()] + ["列表预设"]
        self.assertEqual(seen[0], ("karaoke-fx", [(s, "karaoke-fx") for s in expected]))
        for m in originals[:2]: self.assertEqual(kara.tracks_of(self.doc, m.id)[0]["effect"]["preset"], "glow")
        self.assertEqual(kara.tracks_of(self.doc, originals[2].id)[0], next(t for t in before.values() if t["member"] == originals[2].id))
        self.session.editor.history.undo()
        self.assertEqual(self.doc.karaoke, before)
        with patch("subkit_app.views.group_table.QMenu", MenuDouble), patch.object(table, "selected_gids", return_value=["gd"]):
            table.contextMenuEvent(event)
        self.assertIsNone(seen[-1])

    def test_group_context_menu_confirms_karaoke_of_selected_songs(self):
        from PySide6.QtGui import QContextMenuEvent
        from PySide6.QtWidgets import QMenu
        originals = [self.line("あい", gid=gid) for gid in ("ga", "gb", "gd")]
        originals[2].role = "dialogue"
        self.doc.group("gd").kind = "dialogue"
        for track in self.doc.karaoke.values():
            track["state"] = "suggested"
        self.window.select_group("ga")
        seen = []
        class MenuDouble(QMenu):
            def exec(menu, _pos):
                action = next((a for a in menu.actions() if a.text().startswith("确认 K 轴")), None)
                seen.append(action and (action.text(), action.property("help_anchor")))
                return action
        table = self.window.table
        table.resize(1000, 400)
        pos = table.visualRect(table.currentIndex()).center()
        event = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, pos, table.mapToGlobal(pos))
        count = len(self.session.editor.history.done)
        with patch("subkit_app.views.group_table.QMenu", MenuDouble), patch.object(table, "selected_gids", return_value=["ga", "gb", "gd"]):
            table.contextMenuEvent(event)
        self.assertEqual(seen[0], ("确认 K 轴（2 句歌词）", "karaoke-auto"))
        self.assertEqual([kara.effective(self.doc, m).state for m in originals], ["confirmed", "confirmed", "suggested"])
        self.assertEqual(len(self.session.editor.history.done), count + 1)
        self.session.editor.history.undo()
        self.assertEqual([kara.effective(self.doc, m).state for m in originals[:2]], ["suggested", "suggested"])
        with patch("subkit_app.views.group_table.QMenu", MenuDouble), patch.object(table, "selected_gids", return_value=["gd"]):
            table.contextMenuEvent(event)
        self.assertIsNone(seen[-1])

    def test_export_summary_keeps_all_effect_report_lines(self):
        from subkit.core.export import ExportReport
        from subkit_app import mainwindow as mw
        from subkit_app.views.dialogs import ExportDialog

        warnings = ["其他提示"] * 9 + ["卡拉 OK 特效：2 行生成了 80 条字幕行",
                                       "0:00:01.00 这一行特效太复杂，已改用渐变扫过"]
        ep = self.session.episode
        ep.video = None
        ep.validate = lambda: SimpleNamespace(ok=True, summary=lambda: "通过")
        ep.build_export = lambda **_: (None, ExportReport(warnings=warnings))
        ep.write_export = lambda *_: self.path / "fx-export.ass"
        self.project.languages["original"] = "ja"
        with patch.object(ExportDialog, "exec", return_value=ExportDialog.DialogCode.Accepted), \
                patch.object(mw, "check_fonts", return_value=SimpleNamespace(ran=False)), \
                patch.object(mw, "report_lines", return_value=[]), \
                patch.object(self.window, "_help_message", return_value=QMessageBox.StandardButton.Yes) as message:
            self.window.export()
        text = message.call_args.args[-1]
        for line in warnings[-2:]:
            self.assertIn(line, text)


    def test_every_registered_context_effect_applies_once_and_undo_preserves_beats(self):
        members = [self.line("あい", gid=gid) for gid in ("ga", "gb", "gd")]
        self.doc.group("gd").kind = "dialogue"
        self.window.select_group("ga")
        members[2].role = "dialogue"
        choices = self.window._k_context_effect_choices()
        self.assertEqual([key for key, label in choices], list(REGISTRY))
        for name in REGISTRY:
            with self.subTest(effect=name):
                before = copy.deepcopy(self.doc.karaoke)
                count = len(self.session.editor.history.done)
                self.window._k_context_effect(["ga", "gb", "gd"], name)
                self.assertEqual(len(self.session.editor.history.done), count+1)
                for m in members[:2]:
                    track = kara.tracks_of(self.doc, m.id)[0]
                    self.assertEqual(track["effect"]["preset"], name)
                    self.assertEqual(track["beats"], next(t for t in before.values() if t["member"] == m.id)["beats"])
                self.assertEqual(kara.tracks_of(self.doc, members[2].id)[0],
                                 next(t for t in before.values() if t["member"] == members[2].id))
                self.session.editor.history.undo()
                self.assertEqual(before, self.doc.karaoke)


# unittest collects inherited test methods too; the fixture's original coverage runs in its own module.
for _name in list(vars(karaoke_fixture.KaraokeUI)):
    if _name.startswith("test_"):
        setattr(FxPanelTests, _name, None)


class FxCoreTests(unittest.TestCase):
    def setUp(self):
        self.doc = Document("fx", groups=[Group("g0", "song"), Group("g1", "song"),
            Group("stop", "dialogue"), Group("g2", "song"), Group("ed", "song")])
        self.ed = Editor(self.doc, {})
        self.members = []
        for i, gid in enumerate(("g0", "g1", "g2", "ed")):
            m = member(self.doc, gid, f"m{i}", "あい", start=1000+i*4000, end=4000+i*4000,
                       style="ED.JP" if gid == "ed" else "OP.JP")
            self.members.append(m)
            beats = kara.derive(self.doc, m)
            for j, b in enumerate(beats):
                b.t0, b.t1, b.by = m.start+j*500, m.start+(j+1)*500, "tap"
            self.ed.set_karaoke(m.id, beats)
        self.doc.structure_changed()

    def test_scopes_twins_boundaries_one_undo_preserves_tracks(self):
        twin = member(self.doc, "g0", "twin", "あい", layer=0)
        twin.start, twin.end = self.members[0].start, self.members[0].end
        self.ed.set_karaoke(twin.id, kara.effective(self.doc, self.members[0]).beats)
        for scope, gids, expected in (("line", [], {"m0", "twin"}),
                                     ("selection", ["g0", "g2"], {"m0", "m2", "twin"}),
                                     ("song", [], {"m0", "m1", "m2", "twin"})):
            before = copy.deepcopy(self.doc.karaoke)
            count = len(self.ed.history.done)
            self.ed.apply_karaoke_effect("m0", "pop", {"scale": 180, "ktv": "corners"}, scope=scope, gids=gids)
            self.assertEqual(len(self.ed.history.done), count+1)
            actual = {t["member"] for k, t in self.doc.karaoke.items() if t != before[k]}
            self.assertEqual(actual, expected)
            for k, t in self.doc.karaoke.items():
                self.assertEqual(t["beats"], before[k]["beats"])
            self.ed.history.undo()
            self.assertEqual(self.doc.karaoke, before)

    def test_invalid_params_atomic_and_placeholder_track(self):
        before = copy.deepcopy(self.doc.karaoke)
        with self.assertRaises(ValueError):
            self.ed.apply_karaoke_effect("m0", "pop", {"scale": 500})
        self.assertEqual(self.doc.karaoke, before)
        self.doc.karaoke.clear()
        self.ed.apply_karaoke_effect("m0", "pop", {}, scope="song")
        self.assertEqual({t["member"] for t in self.doc.karaoke.values()}, {"m0", "m1", "m2"})
        self.assertTrue(all(t["state"] == "placeholder" for t in self.doc.karaoke.values()))
        self.ed.history.undo()
        self.assertFalse(self.doc.karaoke)

    def test_whole_song_skips_credits_but_stops_at_15001_ms_gap(self):
        self.doc.groups[2].kind = "sign"
        credit = member(self.doc, "stop", "credit", "staff", start=0, end=999999, style="Default")
        credit.role = "sign"
        before = copy.deepcopy(self.doc.karaoke)
        self.members[2].start = self.members[1].end + 15000
        self.ed.apply_karaoke_effect("m1", "glow", {}, scope="song")
        changed = {t["member"] for k, t in self.doc.karaoke.items() if t != before[k]}
        self.assertEqual(changed, {"m0", "m1", "m2"})
        self.ed.history.undo()
        self.assertEqual(self.doc.karaoke, before)
        self.members[2].start += 1
        self.ed.apply_karaoke_effect("m0", "glow", {}, scope="song")
        changed = {t["member"] for k, t in self.doc.karaoke.items() if t != before[k]}
        self.assertEqual(changed, {"m0", "m1"})

    def test_soma_copy_export_preview_report_and_passage_roundtrip(self):
        from subkit.ass import AssDocument
        from subkit.styles_default import default_styles_ass
        with tempfile.TemporaryDirectory(prefix="fx4_soma_") as scratch:
            root = Path(scratch)
            # Copy only project metadata/template; the synthetic document never writes into the source project.
            source = Path(__file__).resolve().parents[1] / "test_projects" / "soma"
            shutil.copytree(source, root / "soma", ignore=shutil.ignore_patterns("sources", "cache", "episodes"))
            store = PresetStore(root / "soma")
            params = {"scale": 180, "ktv": "stacked", "line_advance_ms": 600}
            store.save("歌曲预设", "pop", params)
            entry = PresetStore(root / "soma").list()[0]
            self.ed.apply_karaoke_effect("m0", entry["effect"], entry["params"], scope="song")
            template = AssDocument.parse(default_styles_ass())
            preview, rep = build_ass(self.doc, {}, template, root, ExportOptions(mode="preview"))
            release, _ = build_ass(self.doc, {}, template, root, ExportOptions(mode="release"))
            self.assertEqual(preview.events, release.events)
            self.assertGreater(rep.karaoke_fx_lines, 0)
            self.assertIn(f"卡拉 OK 特效：{rep.karaoke_fx_lines} 行生成了 {rep.karaoke_fx_events} 条字幕行", rep.warnings)
            saved = recorded(self.doc, ["g0"])
            self.assertEqual(saved["groups"][0]["members"][0]["karaoke"]["effect"],
                             {"preset": "pop", "params": params})
            target = Document("target", groups=[Group("target", "song")])
            m = member(target, "target", "target-m", "あい", start=10000, end=13000)
            result = plan(target, saved, match(covered=(0, 3000)))
            chosen = [a.id for line in result.lines for a in line.actions if a.kind == "karaoke"]
            Editor(target, {}).apply_passage(result, chosen)
            self.assertEqual(kara.tracks_of(target, m.id)[0]["effect"], {"preset": "pop", "params": params})
