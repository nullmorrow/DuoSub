"""Release cleanup and annotation order preserve layout, tags, timing and sources."""
from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from subkit.ass.document import AssDocument, Event
from subkit.ass.text import VisibleText, tokenize
from subkit.core.export import ExportCache, ExportOptions, ExportReport, build_ass
from subkit.core.export_presentation import clean_text, empty_dialogue, finish_release, preserve_positioned_order
from subkit.core.pinyin_notes import annotate_pinyin
from subkit.core.readings import anchor_of, visible_map
from subkit.core.workspace import Project
from subkit.model.document import Annotation, Document, Group, Member, Run
from subkit.model.serialize import document_to_json
from subkit.proc import run
from tests.test_halfwidth_kana import fixture
from tests.test_karaoke_fx import fixture as fx_fixture, ktv_fixture
from subkit.karaoke_fx import REGISTRY


def mixed():
    doc, template = fixture()
    doc.members.clear()
    doc.groups.clear()
    template.styles.append(template.style('JP').copy('CN'))
    template.style('CN').set('MarginV', 140)
    for n, kind in enumerate(('dialogue', 'song')):
        group = Group(f'g_{n}', kind)
        doc.groups.append(group)
        for lane, lang, text, style in (('original', 'ja', ' 国 ', 'JP'),
                                        ('translation', 'zh-Hans', ' 银行 ', 'CN')):
            mid = f'{lane}_{n}'
            member = Member(mid, group.id, lang, lane, kind, 1000+n*3000, 3000+n*3000,
                            style=style, runs=[Run(text, ins=mid, b=len(text))])
            doc.members[mid] = member
            group.members.append(mid)
            if lane == 'original':
                ids = visible_map(member)[1]
                aid = f'a_{n}'
                doc.annotations[aid] = Annotation(aid, 'ruby', mid, anchor_of(ids[1:2]), '国', 'くに',
                                                  status='confirmed', parts=[[0, 1, 'くに']])
            else:
                annotate_pinyin(doc, member)
    return doc, template


class CleanupText(unittest.TestCase):
    def test_literal_spaces_across_tags_each_hard_line_and_boundary_map(self):
        raw = r' {\b1} 国 {\b0} \N  {\fad(0,720)}名   '
        cleaned = clean_text(raw)
        self.assertEqual(cleaned.text, r'{\b1}国{\b0}\N{\fad(0,720)}名')
        self.assertEqual(cleaned.spaces, 9)
        self.assertEqual(cleaned.bounds[-1], 2)
        self.assertEqual([t.raw for t in tokenize(cleaned.text) if t.kind == 'tags'],
                         [t.raw for t in tokenize(raw) if t.kind == 'tags'])
        self.assertEqual(clean_text(cleaned.text).text, cleaned.text)
        self.assertEqual(clean_text(cleaned.text).spaces, 0)

    def test_internal_gaps_fullwidth_hardspace_tabs_and_drawings_remain(self):
        for raw in ('国  名', '　国　', r'\h 国 \h', '\t国\t', r'国\N\N名',
                    r'{\p1}m 0 0 l 50 0 50 50{\p0}'):
            with self.subTest(raw=raw):
                self.assertEqual(clean_text(raw).text, raw)
                self.assertFalse(empty_dialogue(raw))
        self.assertEqual(clean_text(r' 国 {\p1}m 0 0 l 50 50{\p0} 名 ').text,
                         r'国 {\p1}m 0 0 l 50 50{\p0} 名')

    def test_empty_only_text_not_comments_or_drawing_commands(self):
        for raw in ('', '   ', '\t', r'\N', r' \N \N ', r'{\fad(0,720)}', r'{a note}'):
            with self.subTest(raw=raw):
                self.assertTrue(empty_dialogue(clean_text(raw).text))
        for raw in (r'\h', '　', r'{\p1}m 0 0 l 50 50{\p0}', '0'):
            self.assertFalse(empty_dialogue(raw))

    def test_trim_consumes_karaoke_wait_on_deleted_glyph_without_retiming(self):
        for kind in ('k','ko','kf','K'):
            raw = '{\\' + kind + '100} 国 '
            expected = '{\\' + kind + '100}{\\' + kind + '0}国'
            self.assertEqual(clean_text(raw).text, expected)
            self.assertEqual(clean_text(expected).text, expected)
        self.assertEqual(clean_text(r'{\k50} {\kf100}国 ').text, r'{\k50}{\k0}{\kf100}国')
        self.assertEqual(clean_text(r'{\k100}国 ').text, r'{\k100}国')

    def test_soft_break_edges_only_trim_when_the_escape_is_a_line_break(self):
        self.assertEqual(clean_text(r'{\q2} 国 \n 名 ').text, r'{\q2}国\n名')
        self.assertEqual(clean_text(r'{\q0} 国 \n 名 ').text, r'{\q0}国 \n 名')


class FileOrder(unittest.TestCase):
    def data(self):
        ass = AssDocument()
        ass.events = [Event(text=' A '), Event(text='ruby1'), Event(text='B', style='User.Ruby'),
                      Event(text='ruby2'), Event(text='pinyin1'), Event(text='gloss'),
                      Event(kind='Comment', text='  {original author note}  ')]
        report = ExportReport(events=4, ruby_lines=2, pinyin_lines=1)
        report.owners = [dict(member=mid, kind=kind) for mid,kind in
                         [('a','base'), ('a','ruby'), ('b','base'), ('b','ruby'),
                          ('a','pinyin'), ('a','gloss'), ('c','base')]]
        return ass, report

    def test_end_extracts_annotations_by_owner_preserves_other_order(self):
        ass, report = self.data()
        finish_release(ass, report, 'end')
        self.assertEqual([e.text for e in ass.events],
                         ['A','B','gloss','  {original author note}  ','ruby1','ruby2','pinyin1'])
        self.assertEqual(report.edge_spaces_removed, 2)
        self.assertEqual(report.empty_events_removed, 0)
        self.assertEqual([o['kind'] for o in report.owners], ['base','base','gloss','base','ruby','ruby','pinyin'])

    def test_after_parent_keeps_annotation_order_and_comment_bytes(self):
        ass, report = self.data()
        finish_release(ass, report, 'after_parent')
        self.assertEqual([e.text for e in ass.events],
                         ['A','ruby1','pinyin1','B','ruby2','gloss','  {original author note}  '])
        self.assertEqual(report.annotation_order, 'after_parent')
        self.assertEqual(len(ass.events), len(report.owners))

    def test_effect_block_multiple_parents_orphan_and_no_style_guess(self):
        ass, report = self.data()
        ass.events.insert(3, Event(text='B effect'))
        report.owners.insert(3, dict(member='b',kind='base'))
        ass.events.append(Event(text='orphan'))
        report.owners.append(dict(member='missing',kind='ruby'))
        finish_release(ass, report, 'after_parent')
        texts = [e.text for e in ass.events]
        self.assertEqual(texts[texts.index('B effect')+1], 'ruby2')
        self.assertEqual(texts[-1], 'orphan')
        self.assertLess(texts.index('B'), texts.index('B effect'))

    def test_empty_events_counts_owner_maps_and_idempotence(self):
        ass, report = self.data()
        ass.events[0].text = '   '
        ass.events[1].text = r'{\fad(0,720)}'
        finish_release(ass, report, 'after_parent')
        self.assertEqual(report.empty_events_removed, 2)
        self.assertEqual(report.events, 3)
        self.assertEqual(report.ruby_lines, 1)
        self.assertEqual(report.pinyin_lines, 1)
        before = ass.serialize(), copy.deepcopy(report.owners)
        finish_release(ass, report, 'after_parent')
        self.assertEqual((ass.serialize(), report.owners), before)
        self.assertEqual(report.empty_events_removed, 2)

    def test_validation_rejects_bad_order_or_unpaired_owners(self):
        with self.assertRaisesRegex(ValueError, '注音行顺序'):
            ExportOptions(annotation_order='typo')
        ass, report = self.data()
        report.owners.pop()
        with self.assertRaisesRegex(ValueError, '数量'):
            finish_release(ass, report, 'end')

    def test_positioned_overlap_keeps_original_compositing_order_in_both_files(self):
        ass,report=self.data()
        ass.events=ass.events[:5]
        report.owners=report.owners[:5]
        for event in ass.events:
            event.start,event.end,event.layer=100,300,5
            event.text=r'{\pos(200,200)}'+event.text.strip(' ')
        owners=copy.deepcopy(report.owners)
        original_text=[e.text for e in ass.events]
        self.assertGreater(preserve_positioned_order(ass,report.owners),0)
        self.assertEqual(sorted(e.layer for e in ass.events),[e.layer for e in ass.events])
        self.assertEqual([e.text for e in ass.events],original_text)
        self.assertEqual(report.owners,owners)
        before={e.text:e.layer for e in ass.events}
        finish_release(ass,report,'after_parent')
        self.assertEqual([e.layer for e in ass.events], [before[e.text] for e in ass.events])

    def test_layer_expansion_keeps_unpositioned_collision_groups_and_original_z_groups(self):
        ass=AssDocument()
        ass.events=[Event(start=0,end=100,layer=4,text=r'{\pos(1,2)}A'),
                    Event(start=0,end=100,layer=4,text=r'{\pos(1,2)}reading'),
                    Event(start=0,end=100,layer=4,text=r'{\pos(1,2)}B'),
                    Event(start=300,end=500,layer=4,text='normal A'),
                    Event(start=350,end=550,layer=4,text='normal B'),
                    Event(start=0,end=100,layer=8,text='higher layer')]
        owners=[dict(member=m,kind=k) for m,k in [('a','base'),('a','ruby'),('b','base'),
                                                 ('c','base'),('d','base'),('e','base')]]
        self.assertGreater(preserve_positioned_order(ass,owners),0)
        self.assertEqual(ass.events[3].layer,ass.events[4].layer)
        self.assertLess(max(e.layer for e in ass.events[:5]),ass.events[5].layer)
        self.assertEqual([e.text for e in ass.events[3:]],['normal A','normal B','higher layer'])

    def test_mixed_or_invalid_position_components_do_not_split_collision_layers(self):
        for text in ('B',r'{\pos(1)}B',r'{\move(bad)}B'):
            ass=AssDocument()
            ass.events=[Event(start=0,end=100,layer=5,text=r'{\pos(1,2)}A'),
                        Event(start=0,end=100,layer=5,text=r'{\pos(1,2)}reading'),
                        Event(start=0,end=100,layer=5,text=text)]
            owners=[dict(member=m,kind=k) for m,k in [('a','base'),('a','ruby'),('b','base')]]
            self.assertEqual(preserve_positioned_order(ass,owners),0)
            self.assertEqual([e.layer for e in ass.events],[5,5,5])


class ExportIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='subkit-presentation-')
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def export(self, doc, template, **kw):
        cache = kw.pop('cache', None)
        return build_ass(doc, {}, template, self.path, ExportOptions(**kw), cache=cache,
                         measure_cache=self.path/'measure.sqlite3')

    def assert_after_parent(self, report):
        for i, owner in enumerate(report.owners):
            if owner['kind'] not in ('ruby','pinyin'):
                continue
            previous = i-1
            while previous >= 0 and report.owners[previous]['kind'] in ('ruby','pinyin'):
                self.assertEqual(report.owners[previous]['member'], owner['member'])
                previous -= 1
            self.assertGreaterEqual(previous, 0)
            self.assertEqual(report.owners[previous]['member'], owner['member'])
            self.assertEqual(report.owners[previous]['kind'], 'base')

    def test_dialogue_song_ruby_and_pinyin_only_order_differs(self):
        doc, template = mixed()
        saved = copy.deepcopy(document_to_json(doc))
        end, er = self.export(doc, template, pinyin_mode='PY')
        following, fr = self.export(doc, template, pinyin_mode='PY', annotation_order='after_parent')
        self.assertTrue(er.ruby_lines and er.pinyin_lines)
        self.assertEqual(Counter(e.to_line(end.event_format) for e in end.events),
                         Counter(e.to_line(following.event_format) for e in following.events))
        self.assertEqual((er.ruby_lines,er.pinyin_lines), (fr.ruby_lines,fr.pinyin_lines))
        kinds=[o['kind'] for o in er.owners]
        last_base=max(i for i,k in enumerate(kinds) if k=='base')
        self.assertTrue(all(i>last_base for i,k in enumerate(kinds) if k in ('ruby','pinyin')))
        self.assert_after_parent(fr)
        self.assertEqual(document_to_json(doc), saved)
        self.assertEqual(er.edge_spaces_removed, 8)

    def test_only_release_cleans_and_remaps_final_metadata_cache_is_immutable(self):
        doc, template = mixed()
        cache=ExportCache()
        release, rr = self.export(doc, template, cache=cache, pinyin_mode='PY')
        self.assertTrue(all(not VisibleText(e.text).text.startswith(' ') for e in release.events))
        self.assertTrue(cache.members['original_0'][1].event.text.endswith(' '))
        base = next((e,o) for e,o in zip(release.events,rr.owners) if o['member']=='original_0' and o['kind']=='base')
        self.assertEqual(base[1]['ids'], visible_map(doc.members['original_0'])[1][1:2])
        for mode in ('preview','work'):
            ass, report=self.export(doc,template,mode=mode,pinyin_mode='PY',annotation_order='after_parent')
            text=next(e.text for e,o in zip(ass.events,report.owners) if o['member']=='original_0' and o['kind']=='base')
            self.assertTrue(text.startswith(' ') and text.endswith(' '))
            self.assertEqual(report.edge_spaces_removed,0)
            self.assertEqual(report.empty_events_removed,0)

    def test_all_deleted_row_only_removed_from_release_not_document(self):
        from subkit.core.display import DisplayRules
        doc, template = fixture('余')
        rules=DisplayRules(replacements=({'source':'余','target':'','lane':'original'},))
        for mode in ('release','preview'):
            ass, report=self.export(doc,template,mode=mode,display=rules)
            self.assertEqual(len(ass.events),0 if mode=='release' else 1)
            self.assertEqual(report.empty_events_removed,1 if mode=='release' else 0)
        self.assertEqual(doc.members['m_kana'].text,'余')

    def test_all_builtin_song_effects_ruby_and_source_are_preserved(self):
        for preset in REGISTRY:
            with self.subTest(preset=preset):
                doc, template = fx_fixture(preset,ruby=True)
                before=copy.deepcopy(document_to_json(doc))
                end, er=self.export(doc,template)
                following, fr=self.export(doc,template,annotation_order='after_parent')
                self.assertEqual(Counter(e.to_line(end.event_format) for e in end.events),
                                 Counter(e.to_line(following.event_format) for e in following.events))
                self.assert_after_parent(fr)
                self.assertEqual(document_to_json(doc),before)
                self.assertEqual(len(following.events),len(fr.owners))

    def test_episode_api_default_and_explicit_order_do_not_save_settings(self):
        project=Project.create(self.path/'project','导出选项')
        project.set_pinyin_mode('PY')
        episode=project.add_episode('E1')
        episode.document, template=mixed()
        project.template=lambda: template
        before=copy.deepcopy(project.data)
        ass, report=episode.build_export(annotation_order='after_parent')
        self.assert_after_parent(report)
        self.assertEqual(project.data,before)
        self.assertEqual(episode.build_export()[1].annotation_order,'end')

    @unittest.skipUnless(shutil.which('ffmpeg'), 'requires real ffmpeg/libass')
    def test_legacy_k_and_ko_space_syllables_keep_actual_rendered_highlight(self):
        samples=[r'{\k100} 国 ',r'{\ko100} 国 ',r'{\kf100} 国 ',r'{\k50} {\ko100} 国 ']
        for i,text in enumerate(samples):
            doc,template=fixture(text)
            digests=[]
            for mode in ('preview','release'):
                ass,report=self.export(doc,template,mode=mode)
                stem=f'karaoke_{i}_{mode}'
                ass.save(self.path/(stem+'.ass'))
                proc=run(['ffmpeg','-hide_banner','-loglevel','error','-f','lavfi','-i',
                          'color=c=black:s=640x360:r=25:d=3.5','-vf',f'ass={stem}.ass',
                          '-f','framemd5','-'],cwd=self.path,capture_output=True,timeout=30)
                self.assertEqual(proc.returncode,0,proc.stderr.decode('utf-8','replace'))
                frames=[row.rsplit(',',1)[-1] for row in proc.stdout.decode().splitlines() if not row.startswith('#')]
                self.assertGreater(len(set(frames)),1)
                digests.append(frames)
            self.assertEqual(digests[0],digests[1],text)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'requires real ffmpeg/libass')
    def test_multiline_particle_compositing_matches_legacy_renderer_after_reordering(self):
        doc,template=ktv_fixture('particles',mode='stacked',ruby=True)
        with patch('subkit.core.export_presentation.preserve_positioned_order',return_value=0):
            legacy,_=self.export(doc,template,mode='preview')
        variants=[('legacy',legacy)]
        for mode,order in [('preview','end'),('release','end'),('release','after_parent')]:
            ass,report=self.export(doc,template,mode=mode,annotation_order=order)
            if mode=='release':self.assertGreater(report.order_layer_events,0)
            else:self.assertEqual(report.order_layer_events,0)
            variants.append((mode+'_'+order,ass))
        reference=None
        for stem,ass in variants:
            ass.save(self.path/(stem+'.ass'))
            proc=run(['ffmpeg','-hide_banner','-loglevel','error','-f','lavfi','-i',
                      'color=c=black:s=1920x1080:r=25:d=5','-vf',f'ass={stem}.ass',
                      '-f','framemd5','-'],cwd=self.path,capture_output=True,timeout=30)
            self.assertEqual(proc.returncode,0,proc.stderr.decode('utf-8','replace'))
            frames=[row.rsplit(',',1)[-1] for row in proc.stdout.decode().splitlines() if not row.startswith('#')]
            self.assertGreater(len(set(frames)),1)
            if reference is None:reference=frames
            else:self.assertEqual(frames,reference,stem)


if __name__ == '__main__':
    unittest.main()
