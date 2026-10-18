"""Real context menu / MainWindow transactions without model inference."""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import unittest
from unittest.mock import Mock

from PySide6.QtWidgets import QDialog, QPlainTextEdit

from tests import test_convert_ui as conversion
from subkit.listening import Candidate, ListeningResult
from subkit.model.document import Member, Run


class ProposalDialog(QDialog):
    def __init__(self, request, cn, parent):
        super().__init__(parent)
        self.request, self.cn = request, cn
        self.text = QPlainTextEdit(self)
        self.job = None
        self.before_adopt = None
        self.invalid = ""
        self.candidate = Candidate("風船はいかがですか", request.start_ms, request.end_ms, ["fake acoustic fixture"], "")
        self.result = ListeningResult([self.candidate], {}, max(0, request.start_ms - 1200),
                                      request.end_ms + 1200, "test", [], {"model": "fixture"})

    def invalidate(self, text):
        self.invalid = text

    def choose(self, text=None):
        c = self.candidate
        okay = self.before_adopt((c, text or c.text, c.start_ms, c.end_ms))
        if okay:
            self.accept()
        return okay


class ListeningMainWindow(unittest.TestCase):
    setUpClass = classmethod(conversion.ConversionUI.setUpClass.__func__)
    setUp = conversion.ConversionUI.setUp
    _dispose = conversion.ConversionUI._dispose
    group = conversion.ConversionUI.group
    select = conversion.ConversionUI.select
    snapshot = conversion.ConversionUI.snapshot
    context_menu = conversion.ConversionUI.context_menu

    def prepare(self, kind="sign"):
        cn = self.group("target", kind, r"{\an7\pos(965,128)\fad(0,720)}要氣球嗎", 806200)
        cn.role, cn.style = kind, "Screen" if kind == "sign" else "CN"
        self.video = self.path / "video.mkv"
        self.video.write_bytes(b"identity-only fixture, no decoding")
        self.episode.video = self.video
        self.episode.data = {"media": {"audio_track": 2}}
        self.project.languages = {"original": "ja", "translation": "zh-Hant"}
        self.window.model.languages = self.project.languages
        self.window.player.pause = Mock()
        self.window.listening_dialog_factory = ProposalDialog
        self.select("target")
        return cn

    def dialog(self):
        self.window._start_listening("target")
        context = self.window._listening_context
        self.assertIsNotNone(context)
        return context, context["dialog"]

    def test_context_menu_has_single_chinese_sign_action_and_dispatch(self):
        self.prepare()
        calls = []
        self.window.table.listeningRequested.connect(calls.append)
        actions = self.context_menu("AI听译")
        action = next(a for a in actions if "AI听译" in a.text())
        self.assertTrue(action.isEnabled())
        self.assertEqual(action.property("help_anchor"), "ai-transcribe")
        self.assertEqual(calls, ["target"])
        self.assertEqual(self.window._listening_context["dialog"].request.audio_track, 2)

    def test_context_menu_disables_existing_original_non_japanese_and_multiselect(self):
        self.prepare()
        def enabled():
            return next(a for a in self.context_menu() if "AI听译" in a.text()).isEnabled()
        self.project.languages["original"] = "en"
        self.assertFalse(enabled())
        self.project.languages["original"] = "ja"
        mid = self.session.editor.add_missing_language({"target": "既存"}, "original")[0]
        self.assertFalse(enabled())
        self.session.editor.delete_member(mid)
        self.group("second")
        self.select("target", "second")
        self.assertFalse(enabled())

    def test_proposal_is_readonly_then_adoption_keeps_sign_and_undo(self):
        cn = self.prepare()
        before, original = self.snapshot(), copy.deepcopy(asdict(cn))
        context, dialog = self.dialog()
        self.assertEqual(before, self.snapshot())
        self.assertTrue(dialog.choose("風船はいかがですか"))
        self.assertIsNone(self.window._listening_context)
        self.assertEqual(original, asdict(self.doc.members[cn.id]))
        jp = next(m for m in self.doc.members_of(self.doc.group("target")) if m.lane == "original")
        self.assertEqual((jp.role, jp.style), ("dialogue", "JP"))
        self.assertIn("asr", jp.extra)
        self.assertTrue(self.doc.annotations)
        self.assertTrue(all(r.ins and r.src is None for r in jp.runs))
        self.session.editor.history.undo()
        self.assertEqual(before, self.snapshot())

    def test_target_text_or_time_change_prevents_old_adoption(self):
        cn = self.prepare()
        context, dialog = self.dialog()
        self.session.editor.replace_text(cn.id, "變更")
        self.assertFalse(dialog.choose())
        self.assertIn("已经改变", dialog.invalid)
        self.assertFalse(any(m.lane == "original" for m in self.doc.members.values()))

    def test_unrelated_group_edit_does_not_invalidate_target(self):
        self.prepare()
        other = self.group("second")
        context, dialog = self.dialog()
        self.session.editor.replace_text(other.id, "別の訳")
        self.assertTrue(self.window._validate_listening(context))
        self.assertTrue(dialog.choose())

    def test_video_identity_and_track_change_rejected(self):
        self.prepare()
        context, dialog = self.dialog()
        self.episode.data["media"]["audio_track"] = 1
        self.assertFalse(dialog.choose())
        self.episode.data["media"]["audio_track"] = 2
        self.video.write_bytes(b"different video bytes")
        self.assertFalse(self.window._validate_listening(context))

    def test_existing_original_added_while_listening_is_never_overwritten(self):
        self.prepare()
        context, dialog = self.dialog()
        mid = self.session.editor.add_missing_language({"target": "人の修正"}, "original")[0]
        self.assertFalse(dialog.choose())
        self.assertEqual(self.doc.members[mid].text, "人の修正")

    def test_new_session_identity_and_close_reject_results(self):
        self.prepare()
        context, dialog = self.dialog()
        old = self.window.session
        self.window.session = object()
        self.assertFalse(self.window._validate_listening(context))
        self.window.session = old
        self.assertTrue(self.window._close_listening())
        self.assertIsNone(self.window._listening_context)

    def test_close_retains_owner_when_cancelled_job_not_reaped(self):
        self.prepare()
        context, dialog = self.dialog()
        dialog.job = Mock()
        dialog.job.finished.is_set.return_value = False
        dialog.job.finished.wait.return_value = False
        # Real ListeningDialog defers rejected; this fixture keeps it open too.
        dialog.reject = Mock()
        self.assertFalse(self.window._close_listening())
        self.assertIs(self.window._listening_context, context)
        dialog.job.cancel.assert_called_once()
        dialog.job.finished.wait.assert_called_once_with(2)
        dialog.job = None
        dialog.reject = lambda: dialog.done(QDialog.DialogCode.Rejected)
        self.assertTrue(self.window._close_listening())


if __name__ == "__main__":
    unittest.main()
