"""F3: offscreen Qt, player double; speaker-only changes keep preview and focus."""
import time
import unittest
from unittest.mock import Mock, patch
from tests import test_convert_ui as fixtures
from subkit.core.people import NameForm
from subkit.core.export import ExportOptions, build_ass
from subkit.core.layout_index import LineGeom, Rect
from subkit.model.document import Member, Run


class ReviewF3(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__['setUpClass']
    _dispose = fixtures.ConversionUI._dispose
    group = fixtures.ConversionUI.group
    select = fixtures.ConversionUI.select

    def setUp(self):
        fixtures.ConversionUI.setUp(self)
        self.project.pinyin_mode = 'off'
        self.a = self.group('a', text='body')
        self.group('b', start=5000)
        self.a.speaker = {'label': 'pending'}
        self.book = self.session.editor.character_book
        self.ch = self.book.add_character({'en': NameForm(given='Alice')})
        self.book.link_document(self.doc)
        self.window.model.character_book = self.book
        self.window.people.set_context(self.project, self.session)
        self.window.watch._measuring_stopped = True
        self.select('a', 'b')
        self.window.show()
        self.window.activateWindow()
        self.window.table.setFocus()
        self.app.processEvents()

    def test_assign_undo_redo_preserve_follow_current_multiselect_focus_and_geometry(self):
        w = self.window
        w._follow = True
        w.player.paused = False
        w.watch.geo = {7: object()}
        before = (w.current_gid, w.watch.selected, w.wave.current_gid, w.table.selected_gids(), self.app.focusWidget())
        with patch.object(w.model, 'reset', wraps=w.model.reset) as reset:
            self.session.editor.edit_characters(lambda b: b.resolve_pending('pending', self.ch.id))
            for action in (None, self.session.editor.history.undo, self.session.editor.history.redo):
                if action: action()
                self.assertEqual((w.current_gid,w.watch.selected,w.wave.current_gid,w.table.selected_gids(),self.app.focusWidget()),before)
                self.assertTrue(w._follow)
                self.assertIn(7,w.watch.geo)
            reset.assert_not_called()
        self.session.request_preview.assert_not_called()
        self.assertEqual(len(self.session.editor.history.done),1)
        w._on_time(5.2)
        self.assertEqual(w.current_gid,'b')
        self.assertEqual(w.table.selected_gids(),['a','b'])

    def test_name_scan_cached_for_people_changes_and_invalidated_by_text(self):
        import subkit_app.views.people as people
        with patch.object(people,'find_names',wraps=people.find_names) as scan:
            self.session.editor.edit_characters(lambda b: b.resolve_pending('pending',self.ch.id))
            self.window.people.refresh()
            scan.assert_not_called()
            self.a.runs=[Run('changed',ins=self.a.id,a=0,b=7)]
            self.window.people.refresh()
            self.assertEqual(scan.call_count,1)

    def test_cached_overlay_bounded_and_assignment_invalidates_name(self):
        w = self.window
        ass,rep=build_ass(self.doc,{},self.template,self.path,ExportOptions(ruby=False))
        watch=w.watch
        watch.ass,watch.owners=ass,rep.owners
        watch._starts=sorted((e.start,e.end,i) for i,e in enumerate(ass.events))
        watch._max_dur=200
        index=next(i for i,o in enumerate(rep.owners) if o['member']==self.a.id)
        watch.geo={index:LineGeom(index,self.a.id,'a','base',0,Rect(600,400,800,460))}
        w.watch_mode=True
        self.addCleanup(setattr,w,'watch_mode',False)
        w.player.osd={'w':1920,'h':1080};w.player.time=1.2
        w.act_watch_speaker.setChecked(True)
        watch.invalidate_speakers()
        with patch.object(watch,'_speaker_text',wraps=watch._speaker_text) as text:
            start=time.perf_counter()
            for _ in range(100):watch.on_time(1.2)
            self.assertLess((time.perf_counter()-start)*1000/100,16)
            self.assertEqual(text.call_count,1)
            self.session.editor.edit_characters(lambda b:b.resolve_pending('pending',self.ch.id))
            self.assertEqual(text.call_count,2)
        from subkit_app.watch import OV_SPEAKER
        overlay=next(call.args[1] for call in reversed(w.player.overlay.call_args_list) if call.args[0]==OV_SPEAKER)
        self.assertIn('Alice',overlay)

    def test_inflight_preview_does_not_repeat_for_speaker_only_revision(self):
        self.session._preview_running=True
        revision=self.doc.revision
        self.session.editor.edit_characters(lambda b:b.resolve_pending('pending',self.ch.id))
        with patch.object(self.session,'_start_preview') as start:
            self.session._preview_done(revision,'preview.ass',None,None)
            start.assert_not_called()

    def test_changed_character_reading_still_requests_preview(self):
        jp=Member('jp','a','ja','original','dialogue',1000,3000,style='JP',
                  runs=[Run('太郎',ins='jp',a=0,b=2)])
        self.doc.members[jp.id]=jp;self.doc.groups[0].members.append(jp.id)
        from subkit.reading.dictionary import ReadingDict
        from subkit.reading.engine import ReadingEngine
        def engine():
            rd=ReadingDict();rd.merge_character_book(self.book);return ReadingEngine(rd)
        self.session.editor.reading_engine_factory=engine
        self.session.editor.edit_characters(lambda b:b.add_character({'ja':NameForm(given='太郎',given_reading='たろう')}))
        self.session.request_preview.assert_called_once()
        self.assertTrue(any(a.member=='jp' and a.text=='たろう' for a in self.doc.annotations.values()))

    def test_queued_preview_uses_last_visual_revision_after_assignment(self):
        requested = self.doc.revision
        self.session.editor.edit_characters(lambda b:b.resolve_pending('pending',self.ch.id))
        with patch.object(self.session.pool,'start') as start:
            self.session._start_preview()
            job=start.call_args.args[0].job
            self.assertTrue(self.session.layout_current(job.revision))
            job.finished.set()
        with patch.object(self.session,'_start_preview') as start:
            self.session._preview_done(requested,'preview.ass',None,None)
            start.assert_not_called()
