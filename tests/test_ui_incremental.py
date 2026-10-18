"""X4: deferred selection and incremental metadata/preview updates."""
import copy
import unittest
from unittest.mock import Mock, patch

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from tests import test_convert_ui as fixtures
from subkit.ass import AssDocument, Event
from subkit.core.people import NameForm


class IncrementalUI(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    _dispose = fixtures.ConversionUI._dispose
    group = fixtures.ConversionUI.group

    def setUp(self):
        fixtures.ConversionUI.setUp(self)
        self.window.player._hide_timer = Mock()
        self.window.player._autohide = False
        self.window.player.pause = Mock()
        self.project.pinyin_mode = "off"
        for i in range(12):
            self.group(str(i), text=f"line {i}", start=3000 * i)
        self.window.select_group("0")
        self.window.show()
        self.window.table.setFocus()
        self.app.processEvents()

    def test_ten_ctrl_clicks_move_current_and_defer_editor_until_burst_stops(self):
        w = self.window
        w.player.seek.reset_mock()
        with patch.object(w.editor, "set_group", wraps=w.editor.set_group) as rebuild:
            for i in range(1, 11):
                idx = w.proxy.index(i, 4)
                w.table.scrollTo(idx)
                QTest.mouseClick(w.table.viewport(), Qt.MouseButton.LeftButton,
                                 Qt.KeyboardModifier.ControlModifier, w.table.visualRect(idx).center())
                self.assertEqual((w.current_gid, w.table.current_gid(), w.wave.current_gid, w.watch.selected),
                                 (str(i),) * 4)
            rebuild.assert_not_called()
            QTest.qWait(200)
            self.assertEqual(rebuild.call_count, 1)
        self.assertEqual(len(w.table.selected_gids()), 11)
        self.assertEqual(w.editor.group.id, "10")
        self.assertIsNone(w._pending_group)
        self.assertEqual(w.player.seek.call_count, 11)

    def test_translation_row_reused_and_callbacks_target_current_member(self):
        w = self.window
        edit = w.editor.cn_edits[0][1]
        w.select_group("1")
        self.assertIs(w.editor.cn_edits[0][1], edit)
        edit.setPlainText("updated current row")
        edit.commit()
        self.assertEqual(self.doc.members["m_1"].text, "updated current row")
        self.assertEqual(self.doc.members["m_0"].text, "line 0")
        self.session.editor.history.undo()
        self.assertEqual(w.editor.cn_edits[0][1].toPlainText(), "line 1")
        combo = w.editor.speaker_combos["m_1"]
        with patch.object(w, "_edit", return_value=False), patch.object(w.editor, "speakerRequested") as requested:
            combo.chosen.emit("nobody")
            requested.emit.assert_called_with("m_1", "nobody", True)

    def test_speaker_metadata_does_not_rebuild_editor_or_move_waveform(self):
        w = self.window
        ch = self.session.editor.character_book.add_character({"en": NameForm(given="Alice")})
        w.player.pause = Mock()
        before = w.wave.t0, w.wave.current_gid
        with patch.object(w.editor, "set_group", wraps=w.editor.set_group) as rebuild:
            w._set_speaker("m_0", ch.id, True)
            rebuild.assert_not_called()
        self.assertEqual((w.wave.t0, w.wave.current_gid), before)
        self.assertFalse(self.session.preview_change)
        self.assertEqual(self.doc.members["m_0"].speaker["character"], ch.id)

    def test_metadata_refresh_retains_unaffected_row_cache(self):
        model = self.window.model
        old = model._row(self.doc.group("1"))
        model._row(self.doc.group("0"))
        self.doc.revision += 1
        model.refresh_groups({"0"})
        self.assertIs(model._row(self.doc.group("1")), old)
        self.assertNotIn("0", model._cache)

    def test_preview_keeps_exactly_unchanged_geometry_invalidates_style_and_owner(self):
        watch = self.window.watch
        ass = AssDocument.parse(fixtures.default_styles_ass())
        ass.events = [Event(text="a"), Event(text="b")]
        owners = [{"group": "0", "member": "m_0", "kind": "base"},
                  {"group": "1", "member": "m_1", "kind": "base"}]
        with patch.object(watch, "request_geometry"), patch.object(watch, "redraw"):
            watch.set_preview(ass, owners, {}, None)
            first, second = Mock(), Mock()
            watch.geo = {0: first, 1: second}
            watch.anchors = {0: {"x": 1}, 1: {"x": 2}}
            changed = copy.deepcopy(ass)
            changed.events[1].text = "changed"
            watch.set_preview(changed, owners, {}, None)
            self.assertEqual(watch.geo, {0: first})
            self.assertEqual(watch.anchors, {0: {"x": 1}})
            changed_owner = copy.deepcopy(owners)
            changed_owner[0]["member"] = "m_2"
            watch.set_preview(copy.deepcopy(changed), changed_owner, {}, None)
            self.assertEqual(watch.geo, {})
            watch.geo = {0: first}
            style_changed = copy.deepcopy(changed)
            style_changed.styles[0].values["Fontsize"] = "999"
            watch.set_preview(style_changed, changed_owner, {}, None)
            self.assertEqual(watch.geo, {})

    def test_measurement_copy_skips_unrelated_events_and_isolates_styles(self):
        from subkit.core.transform import measurement_copy

        ass = AssDocument.parse(fixtures.default_styles_ass())
        ass.styles[0].set("Angle", 15)
        event = Event(text=r"{\frz20\fscx150}line")
        unrelated = Mock()
        unrelated.__deepcopy__ = Mock(side_effect=AssertionError("copied unrelated event"))
        ass.events = [event, unrelated]
        measured, neutral = measurement_copy(ass, event)
        self.assertEqual(measured.events, [neutral])
        self.assertEqual(neutral.text, r"{\fscx150}line")
        self.assertEqual(measured.styles[0].num("Angle"), 0)
        self.assertEqual(ass.styles[0].num("Angle"), 15)
        measured.set_info("PlayResX", 123)
        self.assertNotEqual(ass.playres[0], 123)
        self.assertIs(ass.events[1], unrelated)

    def test_filter_does_not_activate_a_replacement_row_or_defer_selection(self):
        w = self.window
        self.doc.group("0").kind = "song"
        w.model.reset()
        w.select_group("0")
        w.filter_box.setCurrentText("只看对白")
        self.assertEqual(w.current_gid, "0")
        self.assertEqual(w.wave.current_gid, "0")
        self.assertIsNone(w._pending_group)

    def test_time_prefetches_upcoming_even_when_current_geometry_is_ready(self):
        watch = self.window.watch
        watch.ass = AssDocument()
        watch.geo = {0: Mock()}
        with patch.object(watch, "active", return_value=[0]), patch.object(watch, "upcoming", return_value=[1]), \
                patch.object(watch, "request_geometry") as request, patch.object(watch, "redraw"):
            watch.on_time(0)
            request.assert_called_once()

    def test_watch_follow_refreshes_hidden_panels_on_visibility_and_exit(self):
        w = self.window
        w.set_watch_mode(True, pinned=True)
        self.app.processEvents()
        with patch.object(w.editor, "set_group", wraps=w.editor.set_group) as rebuild, \
                patch.object(w.source, "show_group", wraps=w.source.show_group) as source:
            w.select_group("1")
            rebuild.assert_not_called()
            source.assert_not_called()
            self.assertEqual((w.wave.current_gid, w.watch.selected), ("1", "1"))
            w.right_host.show()
            w.editor.show()
            w.dock_source.show()
            self.app.processEvents()
            self.assertEqual(w.editor.cn_edits[0][1].toPlainText(), "line 1")
            self.assertEqual(source.call_args.args[0].id, "1")
            w.right_host.hide()
            w.dock_source.hide()
            w.select_group("2")
            w.set_watch_mode(False)
            self.app.processEvents()
            self.assertEqual(w.editor.cn_edits[0][1].toPlainText(), "line 2")
            self.assertEqual(source.call_args.args[0].id, "2")

    def test_row_cache_refresh_updates_changed_text_time_badges_and_undo(self):
        model = self.window.model
        untouched = model._row(self.doc.group("1"))
        model._row(self.doc.group("0"))
        self.session.editor.replace_text("m_0", "new text")
        self.assertEqual(model.data(model.index(0, 4)), "new text")
        self.session.editor.history.undo()
        self.assertEqual(model.data(model.index(0, 4)), "line 0")
        self.doc.members["m_0"].start = 900
        self.doc.group("0").kind = "sign"
        self.doc.revision += 1
        model.refresh_groups({"0"})
        row = model._row(self.doc.group("0"))
        self.assertEqual(row[0], 900)
        self.assertIn(("屏幕字", "info"), row[4])
        # A cache entry older than the preceding revision cannot be declared current.
        self.doc.members["m_1"].runs[0].text = "uncached change"
        self.doc.revision += 2
        model.refresh_groups({"0"})
        self.assertEqual(model.data(model.index(1, 4)), "uncached change")
