"""Per-episode editing position with real projects/settings and offscreen player doubles."""
from __future__ import annotations

import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from tests import test_help as fixtures
from PySide6.QtWidgets import QAbstractItemView
from subkit.core.workspace import Project
from subkit.model.document import Document, Group, Member, Run
from subkit.model.serialize import document_to_json, write_json_atomic
from subkit_app.mainwindow import MainWindow
from subkit_app.player.mpv import MpvWidget
from subkit_app.session import EpisodeSession


class RestorePosition(unittest.TestCase):
    setUpClass = classmethod(fixtures.HelpTests.setUpClass.__func__)

    def setUp(self):
        fixtures.HelpTests.setUp(self)
        preview = patch.object(EpisodeSession, "request_preview", Mock())
        preview.start()
        self.addCleanup(preview.stop)
        w = self.window
        w.player.pause = Mock(side_effect=lambda: setattr(w.player, "paused", True))
        w.player.open = Mock()
        w.wave.load = Mock()
        self.project = Project.create(Path(self.tmp.name) / "project", "Restore")
        for eid in ("A", "B"):
            ep = self.project.add_episode(eid)
            doc = Document(episode=eid)
            for i in range(40):
                gid, mid = f"{eid}-{i}", f"m-{eid}-{i}"
                doc.groups.append(Group(gid, "dialogue", [mid]))
                doc.members[mid] = Member(mid, gid, "zh-Hans", "translation", "dialogue",
                                          i * 3000, i * 3000 + 2000,
                                          runs=[Run(f"line {i}", "text", ins=mid)])
            ep.document = doc
            ep.save()
        w.open_project(self.project.root)

    def dispose(self):
        self.window.close_session()
        fixtures.HelpTests.dispose(self)

    def key(self, eid="A", project=None):
        return f"last_position/{(project or self.project).data['id']}/{eid}"

    def position(self, gid, seconds):
        w = self.window
        w.player.fileLoaded.emit()
        w.select_group(gid)
        w.player.time = seconds

    def assert_position(self, gid, seconds):
        w = self.window
        self.assertEqual(w.current_gid, gid)
        self.assertEqual(w.table.current_gid(), gid)
        self.assertEqual(w.wave.current_gid, gid)
        self.assertEqual(w.editor.group.id, gid)
        self.assertEqual(w.watch.selected, gid)
        self.assertFalse(w.session.dirty)
        self.assertEqual(w.session.editor.history.done, [])
        self.assertEqual(w.session.editor.history.undone, [])
        w.player.fileLoaded.emit()
        w.player.seek.assert_called_with(seconds)
        self.assertTrue(w.player.paused)

    def test_save_reopen_restores_episode_group_time_and_centres_table(self):
        w = self.window
        w.ep_box.setCurrentText("B")
        self.position("B-25", 76.123)
        w.save()
        self.assertEqual(json.loads(self.settings.value(self.key("B"))),
                         {"group_id": "B-25", "time_ms": 76123})
        w.close_session()
        with patch.object(w.table, "scrollTo", wraps=w.table.scrollTo) as scroll:
            w.open_project(self.project.root)
            self.assert_position("B-25", 76.123)
            self.assertEqual(scroll.call_args.args[1], QAbstractItemView.ScrollHint.PositionAtCenter)
        self.assertEqual(w.session.episode.id, "B")

    def test_deleted_group_uses_nearest_start_time(self):
        w = self.window
        self.position("A-25", 80.9)
        w.save()
        w.close_session()
        ep = self.project.episode("A")
        ep.document.groups = [g for g in ep.document.groups if g.id != "A-25"]
        del ep.document.members["m-A-25"]
        ep.save()
        w.open_project(self.project.root)
        self.assert_position("A-27", 80.9)

    def test_combo_switch_keeps_each_episode_position(self):
        w = self.window
        self.position("A-12", 36.5)
        w.ep_box.setCurrentText("B")
        self.position("B-30", 92.1)
        w.ep_box.setCurrentText("A")
        self.assert_position("A-12", 36.5)
        w.ep_box.setCurrentText("B")
        self.assert_position("B-30", 92.1)

    def test_dirty_switch_save_keeps_time(self):
        w = self.window
        self.position("A-12", 36.5)
        w.session.dirty = True
        w.ep_box.setCurrentText("B")
        self.assertEqual(json.loads(self.settings.value(self.key()))["time_ms"], 36500)
        w.ep_box.setCurrentText("A")
        self.assert_position("A-12", 36.5)

    def test_clean_close_event_remembers_without_document_changes(self):
        w = self.window
        self.position("A-20", 61.125)
        before = document_to_json(w.session.doc)
        w.close()
        self.assertEqual(json.loads(self.settings.value(self.key())),
                         {"group_id": "A-20", "time_ms": 61125})
        self.assertEqual(document_to_json(self.project.episode("A").document), before)

    def test_deferred_seek_applies_once_and_is_paused(self):
        w = self.window
        self.position("A-20", 61.125)
        w.close_session()
        w.player.seek.reset_mock()
        w.player.pause.reset_mock()
        w.player.paused = False
        w.open_project(self.project.root)
        w.player.seek.assert_not_called()
        self.assertEqual(w.current_gid, "A-20")
        w.player.fileLoaded.emit()
        w.player.fileLoaded.emit()
        w.player.seek.assert_called_once_with(61.125)
        w.player.pause.assert_called_once()

    def test_pending_seek_preserved_by_save_before_media_ready(self):
        w = self.window
        self.position("A-20", 61.125)
        w.close_session()
        w.open_project(self.project.root)
        w.player.time = 0
        w.save()
        self.assertEqual(json.loads(self.settings.value(self.key()))["time_ms"], 61125)

    def test_save_before_seek_result_does_not_replace_time_with_old_position(self):
        w = self.window
        self.position("A-20", 61.125)
        w.close_session()
        w.open_project(self.project.root)
        w.player.time = 0
        w.player.fileLoaded.emit()
        w.save()
        self.assertEqual(json.loads(self.settings.value(self.key()))["time_ms"], 61125)
        w.player.time = 70
        w.player.timeChanged.emit(70)
        w.save()
        self.assertEqual(json.loads(self.settings.value(self.key()))["time_ms"], 70000)

    def test_switch_discards_previous_pending_seek(self):
        w = self.window
        self.position("A-20", 61.125)
        w.close_session()
        w.open_project(self.project.root)
        w.ep_box.setCurrentText("B")
        w.player.seek.reset_mock()
        w.player.fileLoaded.emit()
        w.player.seek.assert_not_called()

    def test_autosave_records_position_without_undo_entry(self):
        w = self.window
        self.position("A-9", 27.7776)
        w.session.dirty = True
        w.session.autosave()
        self.assertEqual(json.loads(self.settings.value(self.key()))["time_ms"], 27778)
        self.assertTrue(w.session.dirty)
        self.assertEqual(w.session.editor.history.done, [])

    def test_recovery_prompt_precedes_selection_restore(self):
        w = self.window
        self.position("A-20", 61.125)
        w.close_session()
        ep = self.project.episode("A")
        auto = ep.dir / "journal" / "autosave.json"
        recovered = document_to_json(ep.document)
        recovered["groups"] = [g for g in recovered["groups"] if g["id"] != "A-20"]
        del recovered["members"]["m-A-20"]
        write_json_atomic(auto, recovered)
        stamp = (ep.dir / "document.json").stat().st_mtime + 5
        os.utime(auto, (stamp, stamp))
        calls = []
        def answer(episode):
            calls.append("prompt")
            self.assertIsNone(w.session)
            episode.document, _ = episode.autosave_recovery()
            episode.recovered = True
        restore = w._restore_episode_position
        def restored():
            calls.append("restore")
            restore()
        with patch.object(w, "_offer_autosave_recovery", side_effect=answer), \
                patch.object(w, "_restore_episode_position", side_effect=restored):
            w.open_project(self.project.root)
        self.assertEqual(calls, ["prompt", "restore"])
        self.assertEqual(w.current_gid, "A-21")
        self.assertTrue(w.session.dirty)  # Recovery itself remains unsaved.
        self.assertEqual(w.session.editor.history.done, [])

    def test_missing_episode_and_no_record_keep_default(self):
        w = self.window
        w.close_session()
        self.settings.remove(self.key())
        self.settings.setValue(f"last_episode/{self.project.data['id']}", "missing")
        w.open_project(self.project.root)
        self.assertEqual(w.session.episode.id, "A")
        self.assertEqual(w.current_gid, "A-0")
        w.player.seek.reset_mock()
        w.player.fileLoaded.emit()
        w.player.seek.assert_not_called()

    def test_filtered_group_becomes_visible_without_layout_change(self):
        w = self.window
        self.position("A-20", 61.125)
        w.close_session()
        w.filter_box.setCurrentText("只看歌曲")
        layout = w.edit_layout
        w.open_project(self.project.root)
        self.assertEqual(w.edit_layout, layout)
        self.assert_position("A-20", 61.125)

    def test_corrupt_setting_keeps_default(self):
        w = self.window
        w.close_session()
        for value in ('broken', '[]', '{"time_ms": -1}', '{"time_ms": true}'):
            with self.subTest(value=value):
                self.settings.setValue(self.key(), value)
                w.open_project(self.project.root)
                self.assertEqual(w.current_gid, "A-0")
                self.assertIsNone(w._pending_position_seek)
                w.close_session()

    def test_player_open_pauses_before_loading(self):
        player = Mock()
        player.ensure.return_value = engine = Mock()
        MpvWidget.open(player, "test.mkv")
        self.assertEqual(engine.method_calls[0].args, ("pause", True))
        engine.command.assert_called_once_with("loadfile", "test.mkv", "replace")

    def test_project_settings_are_isolated(self):
        w = self.window
        self.position("A-20", 61.125)
        other = Project.create(Path(self.tmp.name) / "other", "Other")
        ep = other.add_episode("A")
        ep.document = Document(episode="A")
        ep.save()
        w.open_project(other.root)
        self.assertIsNone(w.current_gid)
        w.open_project(self.project.root)
        self.assert_position("A-20", 61.125)


if __name__ == "__main__":
    unittest.main()
