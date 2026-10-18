"""Offscreen watch wiring with Qt events, real Editor history and a fake player."""

from __future__ import annotations

import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from tests import test_convert_ui as conversion_ui
from PySide6.QtCore import QPointF, QTimer, Qt, qVersion
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLineEdit
from subkit.core.export import ExportOptions, apply_overrides, build_ass
from subkit.core.layout_index import LineGeom, Rect
from subkit_app.watch import OV_GHOST, OV_TRANSFORM, OV_SELECT


class TransformUI(unittest.TestCase):
    setUpClass = classmethod(conversion_ui.ConversionUI.setUpClass.__func__)
    _dispose = conversion_ui.ConversionUI._dispose
    group = conversion_ui.ConversionUI.group

    def setUp(self):
        conversion_ui.ConversionUI.setUp(self)
        self.watch, self.player = self.window.watch, self.window.player
        self.player.osd = {"w": 1920, "h": 1080, "ml": 0, "mt": 0}
        self.player.time = 1.5
        self.player._autohide = True
        self.player._hide_timer = QTimer(self.player)
        self.player.pause = Mock(side_effect=lambda: setattr(self.player, "paused", True))
        self.player.mpv = SimpleNamespace(set=Mock(), get=Mock(return_value="1080"))
        self.player.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cn = self.group("g", text=r"{\bord3}HH")
        self.jp = self.group("j", text=r"{\frz0}HH")
        self.doc.groups.pop()
        self.jp.group, self.jp.lane, self.jp.style = "g", "original", "JP"
        self.doc.groups[0].members.append(self.jp.id)
        self.cn.overrides = {"an": 7, "pos": [100, 100]}
        self.jp.overrides = {"an": 7, "pos": [100, 170]}
        self.doc.structure_changed()
        self.window.model.reset()
        self.ass, self.report = build_ass(self.doc, {}, self.template, self.path, ExportOptions(ruby=False))
        # Geometry doubles isolate the wiring from subprocesses/libass.
        self.geoms = {}
        self.anchors = {}
        for i, owner in enumerate(self.report.owners):
            mid = owner["member"]
            y = 100 if mid == self.cn.id else 170
            rect = Rect(100, y, 200, y+50)
            self.geoms[i] = LineGeom(i, mid, "g", "base", 0, rect,
                                    _frames=[(rect, rect.corners, [])], layout_origins=((100,y),))
            self.anchors[i] = {"an": 7, "x": 100, "y": y, "move": None}
        self.li = SimpleNamespace(precise_geometry=lambda indices: [copy.deepcopy(self.geoms[i]) for i in indices],
                                  anchor=lambda i: self.anchors[i], perspective_distance=312.5, close=Mock())
        p = patch.object(self.watch, "_layout_index", return_value=self.li)
        p.start()
        self.addCleanup(p.stop)
        self.watch.request_geometry = Mock()
        self.watch.set_preview(self.ass, self.report.owners, {self.cn.id:"translation", self.jp.id:"original"}, None)
        self.watch.geo, self.watch.anchors = copy.deepcopy(self.geoms), copy.deepcopy(self.anchors)
        self.window.select_group("g")
        self.session.engine = SimpleNamespace(analyze=Mock(return_value=[]))
        self.window.watch_mode = True
        self.window._update_transform_action()
        self.window.show()
        self.window.activateWindow()
        self.player.setFocus()
        self.app.processEvents()
        self.session.editor.set_overrides = Mock(wraps=self.session.editor.set_overrides)
        self.initial = {m.id: copy.deepcopy(m.overrides) for m in (self.cn, self.jp)}

    def key(self, key, mods=Qt.KeyboardModifier.NoModifier):
        QTest.keyClick(self.player, key, mods)
        self.app.processEvents()

    def enter(self, alt=False):
        self.key(Qt.Key.Key_T, Qt.KeyboardModifier.AltModifier if alt else Qt.KeyboardModifier.NoModifier)
        self.assertIsNotNone(self.watch.transform)

    def drag(self, handle="inside", dx=15, dy=12, mods=Qt.KeyboardModifier.NoModifier):
        tool = self.watch.transform
        if handle == "inside":
            q = tool.box()
            start = (q[0][0]+25, q[0][1]+20)
        else:
            start = next(h.point for h in tool.handles() if h.name == handle)
        a, b = QPointF(*start), QPointF(start[0]+dx,start[1]+dy)
        self.player.mousePressed.emit(a, Qt.MouseButton.LeftButton, mods)
        self.player.mouseMoved.emit(b, Qt.MouseButton.LeftButton, mods)
        self.player.mouseReleased.emit(b, Qt.MouseButton.LeftButton, mods)
        self.assertIsNotNone(self.watch.transform)

    def assert_restored(self):
        self.assertIsNone(self.watch.transform)
        self.player.mpv.set.assert_any_call("sub-visibility", "yes")
        self.player.clear_overlay.assert_any_call(OV_GHOST)
        self.player.clear_overlay.assert_any_call(OV_TRANSFORM)
        self.assertTrue(self.player.autohide)
        self.assertEqual(self.player.cursor().shape(), Qt.CursorShape.PointingHandCursor)

    def test_t_group_alt_last_clicked_and_original_fallback(self):
        self.enter()
        self.assertEqual(set(self.watch.transform.current_overrides), {self.cn.id,self.jp.id})
        self.player.pause.assert_called()
        self.watch.finish_transform()
        self.enter(alt=True)
        self.assertEqual(set(self.watch.transform.current_overrides), {self.jp.id})
        self.watch.finish_transform()
        self.player.mousePressed.emit(QPointF(125,120),Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier)
        self.player.mouseReleased.emit(QPointF(125,120),Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier)
        self.enter(alt=True)
        self.assertEqual(set(self.watch.transform.current_overrides), {self.cn.id})
        self.li.close.assert_called()

    def test_pending_preview_reenters_untouched_tool_with_same_scope(self):
        for only_line, clicked in ((False, None), (True, None), (True, self.cn.id)):
            with self.subTest(only_line=only_line, clicked=clicked):
                self.watch.last_clicked_member = clicked
                self.enter(alt=only_line)
                old_tool = self.watch.transform
                scope = set(old_tool.current_overrides)
                # A fresh preview supplies newly measured geometry after an edit.
                refreshed = self.ass.copy()
                self.geoms[0].rect = Rect(100, 100, 250, 150)
                self.geoms[0]._frames = [(self.geoms[0].rect, self.geoms[0].rect.corners, [])]
                self.watch.set_preview(refreshed, self.report.owners, self.watch.member_lane, None)
                self.assertIsNotNone(self.watch.transform)
                self.assertIsNot(self.watch.transform, old_tool)
                self.assertEqual(set(self.watch.transform.current_overrides), scope)
                self.assertIs(self.watch.ass, refreshed)
                if self.cn.id in scope:
                    self.assertEqual(self.watch.geo[0].rect.x1, 250)
                self.session.editor.set_overrides.assert_not_called()
                self.key(Qt.Key.Key_Escape)
                self.assert_restored()

    def test_pending_preview_applies_in_progress_drag(self):
        self.enter()
        self.player.mousePressed.emit(QPointF(125, 120), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.player.mouseMoved.emit(QPointF(140, 132), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.watch.set_preview(self.ass, self.report.owners, self.watch.member_lane, None)
        self.session.editor.set_overrides.assert_called_once()
        self.assertEqual(self.cn.overrides['pos'], [115, 112])
        self.assert_restored()

    def test_t_does_not_enter_while_typing_in_fields(self):
        # Inline/reading/format inputs all use QLineEdit; test the app filter
        # with a floating field too, beyond the main-window widget hierarchy.
        for parent in (self.window, None):
            edit = QLineEdit(parent)
            edit.show()
            edit.activateWindow()
            edit.setFocus()
            self.app.processEvents()
            QTest.keyClick(edit,Qt.Key.Key_T)
            self.assertEqual(edit.text(),"t")
            self.assertIsNone(self.watch.transform)
            edit.close()
            edit.deleteLater()

    def test_drag_then_enter_is_one_undo_and_source_unchanged(self):
        texts = (self.cn.text,self.jp.text)
        self.enter()
        self.drag()
        self.session.editor.set_overrides.assert_not_called()
        self.player.mpv.set.assert_any_call("sub-visibility","no")
        self.key(Qt.Key.Key_Return)
        self.session.editor.set_overrides.assert_called_once_with(
            {self.cn.id:{"pos":[115,112]},self.jp.id:{"pos":[115,182]}},"自由变换")
        self.assertEqual((self.cn.text,self.jp.text),texts)
        self.assertEqual(self.session.editor.history.undo().label,"自由变换")
        self.assertEqual({mid: self.doc.members[mid].overrides for mid in self.initial},self.initial)
        self.assertIsNone(self.session.editor.history.undo())
        self.assert_restored()

    def test_esc_changes_nothing(self):
        self.enter()
        self.drag("se",20,20)
        self.key(Qt.Key.Key_Escape)
        self.session.editor.set_overrides.assert_not_called()
        self.assertEqual(self.cn.overrides,self.initial[self.cn.id])
        self.assert_restored()

    def test_backspace_then_enter_clears_only_transform(self):
        self.cn.overrides["raw"] = r"\frz30\bord4"
        # The entry measurement double remains intentionally unrotated.
        self.enter()
        self.key(Qt.Key.Key_Backspace)
        self.session.editor.set_overrides.assert_not_called()
        self.key(Qt.Key.Key_Enter)
        self.assertEqual(self.cn.overrides["raw"],r"\bord4")
        self.assertEqual(self.cn.overrides["pos"],[100,100])
        self.session.editor.set_overrides.assert_called_once()
        self.assert_restored()

    def test_nudges_accumulate_and_3d_uses_screen_scale(self):
        self.player.osd["w"],self.player.osd["h"] = 960,540
        self.enter()
        self.key(Qt.Key.Key_Right)
        self.key(Qt.Key.Key_Down,Qt.KeyboardModifier.ShiftModifier)
        self.assertEqual(self.watch.transform.current_overrides[self.cn.id]["pos"],[101,110])
        self.key(Qt.Key.Key_3)
        self.assertTrue(self.watch.transform.mode_3d)
        point = next(h.point for h in self.watch.transform.handles() if h.name == "se")
        a = QPointF(point[0]*.5,point[1]*.5)
        b = QPointF(a.x()+10,a.y()+20)
        self.player.mousePressed.emit(a,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier)
        self.player.mouseMoved.emit(b,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier)
        self.assertEqual(self.watch._transform_drag.screen_scale,(.5,.5))
        self.assertIn(r"\frx10",self.watch._transform_preview.current_overrides[self.cn.id]["raw"])
        self.player.mouseReleased.emit(b,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier)
        self.key(Qt.Key.Key_Return)
        self.session.editor.set_overrides.assert_called_once()

    def test_double_inside_and_empty_click_apply(self):
        for double in (True,False):
            self.enter()
            self.drag()
            signal = self.player.mouseDoubleClicked if double else self.player.mousePressed
            signal.emit(QPointF(130,125) if double else QPointF(800,800),Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier)
            self.assert_restored()
        self.assertEqual(self.session.editor.set_overrides.call_count,2)

    def test_exit_group_episode_preview_mode_and_error_restore(self):
        for reason in ("group","episode","preview","mode","error"):
            with self.subTest(reason=reason):
                self.watch.set_selected("g")
                self.window.watch_mode = True
                self.player.setCursor(Qt.CursorShape.PointingHandCursor)
                self.player.autohide = True
                self.enter()
                self.drag()
                if reason == "group": self.watch.set_selected(None)
                elif reason == "episode": self.watch.stop_warm()
                elif reason == "preview": self.watch.set_preview(self.ass,self.report.owners,self.watch.member_lane,None)
                elif reason == "mode": self.window.set_watch_mode(False)
                else:
                    q = self.watch.transform.box()          # press inside the current box (outside would apply)
                    a = QPointF(q[0][0]+25, q[0][1]+20)
                    with patch.object(self.watch.transform,"preview",side_effect=ValueError("test error")):
                        self.player.mousePressed.emit(a,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier)
                        self.player.mouseMoved.emit(a+QPointF(10,10),Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier)
                if reason == "mode":
                    self.assertIsNone(self.watch.transform)
                    self.assertFalse(self.player.autohide)
                    self.assertEqual(self.player.cursor().shape(),Qt.CursorShape.ArrowCursor)
                    self.player.mpv.set.assert_any_call("sub-visibility","yes")
                else:
                    self.assert_restored()
                # leaving the tool applies what was done (Esc is the only discard); switching episodes and errors drop it
                if reason in ("group","preview","mode"):
                    self.session.editor.set_overrides.assert_called_once()
                    self.assertEqual(self.session.editor.set_overrides.call_args.args[1],"自由变换")
                else:
                    self.session.editor.set_overrides.assert_not_called()
                self.session.editor.set_overrides.reset_mock()

    def test_menu_state_and_trigger(self):
        self.assertTrue(self.window.act_transform.isEnabled())
        self.window.act_transform.trigger()
        self.assertIsNotNone(self.watch.transform)
        self.watch.finish_transform()
        self.window.watch_mode = False
        self.window._update_transform_action()
        self.assertFalse(self.window.act_transform.isEnabled())

    def test_tab_and_watch_shortcut_exit_active_tool(self):
        for key in (Qt.Key.Key_Tab,Qt.Key.Key_F11):
            self.window.watch_mode = True
            self.watch.set_selected("g")
            self.player.autohide = True
            self.player.setFocus()
            self.enter()
            self.drag()
            self.key(key)
            self.assertFalse(self.window.watch_mode)
            self.assertIsNone(self.watch.transform)
            self.player.clear_overlay.assert_any_call(OV_TRANSFORM)
            self.player.mpv.set.assert_any_call("sub-visibility","yes")

    def test_click_inside_without_drag_then_enter_is_noop(self):
        self.enter()
        self.player.mousePressed.emit(QPointF(125,120),Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier)
        self.player.mouseReleased.emit(QPointF(125,120),Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier)
        self.key(Qt.Key.Key_Return)
        self.session.editor.set_overrides.assert_not_called()
        self.window.watch_mode = True
        self.window._video_deselect()
        self.assertFalse(self.window.act_transform.isEnabled())

    def test_ghost_retags_source_and_draws_all_active_lines(self):
        self.enter(alt=True)
        self.drag()
        ghost = self.watch._transform_ghost_text()
        self.assertEqual(ghost.count("HH"),2)
        self.assertIn(r"\pos(115.00,182.00)",ghost)
        self.assertIn(r"\bord3",ghost)
        drawn = [call.args[1] for call in self.player.overlay.call_args_list if call.args[0] == OV_TRANSFORM]
        self.assertIn("旋转",drawn[-1])

    def test_selection_outline_uses_quads(self):
        q = ((20,30),(50,10),(80,50),(40,70))
        self.watch.geo[0].quads = (q,)
        self.watch.redraw()
        drawn = [call.args[1] for call in self.player.overlay.call_args_list if call.args[0] == OV_SELECT][-1]
        self.assertIn("m 20.0 30.0 l 50.0 10.0 l 80.0 50.0 l 40.0 70.0",drawn)

    def test_ruby_ghost_follows_group_move_scale_and_clear(self):
        from subkit.ass.document import Event
        from subkit.ass.text import line_tag

        self.ass.events.append(Event(start=100,end=300,style="JP",text=r"{\an2\pos(140,165)\fs14}ruby"))
        self.watch.owners.append({"member":self.jp.id,"group":"g","kind":"ruby"})
        self.watch._starts.append((100,300,2))
        self.watch._starts.sort()
        self.enter()
        self.drag()
        self.assertEqual(line_tag(self.watch._transform_ruby[2].text,"pos").float_args(),[155,177])
        self.assertIn("ruby",self.watch._transform_ghost_text())
        self.drag("se",20,20)
        self.assertIsNotNone(line_tag(self.watch._transform_ruby[2].text,"fscx"))
        self.key(Qt.Key.Key_Backspace)
        self.assertIsNotNone(self.watch.transform)
        self.assertIn("ruby",self.watch._transform_ghost_text())

    def test_real_precise_entry_has_origins_and_uses_video_storage_height(self):
        from subkit.core.layout_index import LayoutIndex

        with patch.object(self.watch,"_layout_index",lambda: LayoutIndex(self.ass,self.report.owners,
                                                                        measure_cache=self.path/"probe.sqlite3",
                                                                        video_height=self.watch._video_height())):
            self.player.mpv.get.return_value = "720"
            self.enter()
            self.assertEqual(self.watch.transform.perspective_distance,468.75)
            for geom in self.watch.geo.values():
                self.assertTrue(geom.layout_origins)
                self.assertEqual(len(geom.layout_origins),len(geom._frames))

    def test_source_text_before_overrides_survives_export_cache(self):
        from subkit.core.export import ExportCache

        self.cn.overrides["raw"] = r"\frz25\bord4"
        cache = ExportCache()
        for unused in range(2):
            ass, report = build_ass(self.doc,{},self.template,self.path,ExportOptions(ruby=False),cache=cache)
            i = next(i for i,o in enumerate(report.owners) if o["member"] == self.cn.id)
            self.assertIn(r"\bord3",report.owners[i]["source_text"])
            self.assertNotIn(r"\frz25",report.owners[i]["source_text"])
            self.assertEqual(apply_overrides(report.owners[i]["source_text"],self.cn.overrides)[0],ass.events[i].text)


if __name__ == "__main__":
    print(f"Qt {qVersion()}, platform=offscreen, fake mpv/geometry, real Editor history")
    unittest.main()
