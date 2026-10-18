"""Interrupted karaoke and collision fixes, including real host libass renders."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subkit.ass.document import AssDocument, Event, Style
from subkit.ass.tags import KARAOKE_TAGS
from subkit.ass.text import VisibleText, remove_tags_named, split_karaoke_overrides, tokenize
from subkit.core.export import ExportCache, ExportOptions, _MemberOut, _finish_song_events, build_ass
from subkit.model.document import Document, Group, Member, Run
from subkit.proc import run
from tests.test_karaoke_fx import ktv_fixture


class Syllables(unittest.TestCase):
    def test_kf_font_switch_regression(self):
        # 啦 and the held … share the beat (the block between them splits it); the closing 」 takes no time.
        self.assertEqual(split_karaoke_overrides(r"{\kf81}啦{\fnArial}…」{\fnTimes New Roman}"),
                         r"{\kf41}啦{\fnArial\kf40}…{\kf0}」{\fnTimes New Roman}")

    def test_kf_leading_punctuation_takes_no_time_with_or_without_font_switch(self):
        # 「我 (font switch before 我) and 「分 (same font) behave alike: the bracket fills at the beat start.
        self.assertEqual(split_karaoke_overrides(r"{\kf22}{\kf119}{\fnA}「{\fnB}我{\kf44}知"),
                         r"{\kf22}{\kf0}{\fnA}「{\fnB\kf119}我{\kf44}知")
        self.assertEqual(split_karaoke_overrides(r"{\kf36}「分{\kf11}{\kf72}かっ"),
                         r"{\kf0}「{\kf36}分{\kf11}{\kf72}かっ")
        self.assertEqual(split_karaoke_overrides(r"{\kf50}「A」B"), r"{\kf0}「{\kf50}A」B")
        self.assertEqual(split_karaoke_overrides(r"{\kf50}\h『A"), r"{\kf0}\h『{\kf50}A")
        self.assertEqual(split_karaoke_overrides(r"{\kf118}る…」"), r"{\kf118}る…{\kf0}」")
        # trailing spaces stay in the syllable and … ー 〜 are sung (う…, く␣ as in the user's finished files);
        # a beat of punctuation only and \k (one change at the start) are left as they are
        for text in (r"{\kf51}う…", r"{\kf33}く {\kf21}カッ", r"{\kf60}ら〜", r"{\kf60}ロー", r"{\kf30}…",
                     r"{\k36}「分", r"{\kf40}A B"):
            self.assertEqual(split_karaoke_overrides(text), text)
        out = split_karaoke_overrides(r"{\kf119}{\fnA}「{\fnB}我")
        self.assertEqual(split_karaoke_overrides(out), out)

    def test_k_and_ko_keep_whole_duration_on_last(self):
        for name in ("k", "ko"):
            self.assertEqual(split_karaoke_overrides("{\\"+name+r"81}A{\b1}B{\i1}C"),
                             "{\\"+name+"0}A{\\b1\\"+name+"0}B{\\i1\\"+name+"81}C")

    def test_largest_remainder_tie_source_order_and_alias(self):
        for name in ("kf", "K"):
            text = "{\\"+name+r"5}A{\b1}B{\i1}C"
            out = split_karaoke_overrides(text)
            self.assertEqual([int(t.arg) for tok in tokenize(out) for t in tok.tags if t.name in KARAOKE_TAGS], [2, 2, 1])
            self.assertEqual(split_karaoke_overrides(out), out)

    def test_consecutive_blocks_and_empty_waits(self):
        self.assertEqual(split_karaoke_overrides(r"{\kf50}{\kf80}A{\fnArial}{\b1}BC{\kf20}"),
                         r"{\kf50}{\kf27}A{\fnArial}{\b1\kf53}BC{\kf20}")

    def test_visible_escapes_comments_and_drawing(self):
        text = r"{\kf7}A\h{comment}{\i1}\nB\N{\p1}m 0 0 l 1 1{\p0}"
        out = split_karaoke_overrides(text)
        self.assertIn(r"{\kf4}A\h{comment}{\i1\kf3}\nB", out)
        self.assertEqual(VisibleText(text).text, VisibleText(out).text)

    def test_edges_existing_karaoke_and_plain_text_unchanged(self):
        for text in (r"{\fnArial}{\kf81}{\b1}ABC{\i1}", r"{\kf0}A{\kf5\b1}B",
                     r"A{\fnArial}B", r"{\kf81}{\p1}m 0 0{\p0}"):
            self.assertEqual(split_karaoke_overrides(text), text)


class Layout(unittest.TestCase):
    def fixture(self, starts=(1000, 2980), ends=(3000, 5000)):
        doc, tpl = ktv_fixture(starts=starts, ruby=True)
        for track in doc.karaoke.values():
            track["effect"]["params"] = {}
        for g, end in zip(doc.groups, ends):
            for mid in g.members:
                doc.members[mid].end = end
        return doc, tpl

    def export(self, doc, tpl, cache=None):
        return build_ass(doc, {}, tpl, Path(os.environ['TEMP']), ExportOptions(playres=(960, 540)), cache=cache)

    def test_dialogue_translation_layer_after_sign_conversion(self):
        # A sign converted to dialogue keeps layer 0 for both lines; at the top libass then pushed the
        # translation below the original.  Export puts the translation one layer up; dialogue is never trimmed.
        doc, tpl = self.fixture(ends=(3500, 5000))
        for g in doc.groups:
            g.kind = 'dialogue'
            for mid in g.members:
                m = doc.members[mid]
                m.role, m.layer = 'top', 0
        doc.karaoke.clear()
        ass, rep = self.export(doc, tpl)
        bases = [(e, doc.members[o['member']]) for e, o in zip(ass.events, rep.owners) if o['kind'] == 'base']
        self.assertTrue(bases)
        for e, member in bases:
            self.assertEqual(e.layer, 1 if member.lane == 'translation' else 0)
        self.assertEqual(max(e.end for e, m in bases if m.group == 'g_0'), 350)

    def test_layers_trim_ruby_document_and_cache(self):
        doc, tpl = self.fixture()
        before = copy.deepcopy(doc)
        cache = ExportCache()
        ass, rep = self.export(doc, tpl, cache)
        for e, owner in zip(ass.events, rep.owners):
            member = doc.members[owner['member']]
            self.assertEqual(e.layer, 1 if member.lane == 'translation' else 0)
            if member.group == 'g_0':
                self.assertEqual(e.end, 298)
        self.assertEqual(doc, before)
        self.assertEqual(self.export(doc, tpl, cache)[0].serialize(), ass.serialize())
        # Changing only the neighbour must undo the previous export's trim.
        for mid in doc.groups[1].members:
            doc.members[mid].start = 3100
        updated, _ = self.export(doc, tpl, cache)
        self.assertEqual(updated.events[0].end, 300)
        self.assertEqual(cache.members['JP_0'][1].event.end, 300)

    def test_existing_layers_source_style_and_explicit_position(self):
        for case in ('layer', 'source', 'pos', 'move'):
            doc, tpl = self.fixture(starts=(1000,), ends=(3000,))
            cn = doc.members['CN_0']
            if case == 'layer': cn.layer = 3
            if case == 'source': cn.extra['source_style'] = True
            if case == 'pos': cn.overrides['pos'] = [200, 200]
            if case == 'move': cn.overrides['move'] = [200, 200, 300, 200]
            ass, rep = self.export(doc, tpl)
            self.assertEqual(next(e.layer for e, o in zip(ass.events, rep.owners) if o['member'] == cn.id),
                             3 if case == 'layer' else 0)

    def test_overlap_limits_and_half_duration(self):
        for start, end, trimmed in ((2100, 3100, True), (2090, 3100, False),
                                    (2000, 3000, False), (2985, 3000, True), (3000, 3000, False)):
            doc, tpl = self.fixture(starts=(1000, start), ends=(end, 6000))
            ass, _ = self.export(doc, tpl)
            from subkit.core.export import ms_to_cs
            self.assertEqual(ass.events[0].end, ms_to_cs(start if trimmed else end))

    def test_position_style_and_intervening_line_prevent_trimming(self):
        for case in ('pos', 'move', 'style'):
            doc, tpl = self.fixture()
            later = doc.members['JP_1']
            if case == 'style':
                later.style = 'Other'; tpl.styles.append(tpl.styles[0].copy('Other'))
            else: later.overrides[case] = [200, 200] if case == 'pos' else [200, 200, 300, 200]
            ass, _ = self.export(doc, tpl)
            self.assertEqual(ass.events[0].end, 300)

    def test_final_copy_text_and_annotations_but_not_effect_events(self):
        doc, _ = self.fixture(starts=(1000,), ends=(3000,))
        events = [Event(text=r"{\kf81}A{\fnArial}BC", layer=0) for _ in range(3)]
        owners = [{'member':'JP_0', 'kind':kind} for kind in ('base', 'ruby', 'gloss')]
        results = {'JP_0':_MemberOut(events[0])}
        _finish_song_events(doc, events, owners, results)
        self.assertIn(r'\kf27', events[0].text)
        self.assertIn(r'\kf27', events[1].text)
        self.assertNotIn(r'\kf27', events[2].text)
        original = r"{\kf81}A{\fnArial}BC"
        results['JP_0'].fx_events = [Event()]
        for event in events: event.text = original
        _finish_song_events(doc, events, owners, results)
        self.assertTrue(all(e.text == original for e in events))


class Libass(unittest.TestCase):
    def setUp(self):
        # Evidence remains in run TEMP for the bridge; all subprocesses are waited.
        self.path = Path(tempfile.mkdtemp(prefix='S4-libass-'))

    def render(self, ass, time=2.5):
        ass.save(self.path / 'render.ass')
        proc = run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
                    'color=c=black:s=960x540:r=100:d=1', '-vf',
                    f'setpts=PTS+{time}/TB,subtitles=render.ass,format=rgb24', '-frames:v', '1',
                    '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], cwd=self.path, capture_output=True, timeout=20)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode('utf-8', 'replace'))
        return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(540, 960, 3)

    def template(self):
        ass = AssDocument()
        ass.set_info('PlayResX', '960'); ass.set_info('PlayResY', '540')
        ass.styles = [Style.new('JP', Fontname='Arial', Fontsize=60, Outline=0, Shadow=0,
                     Alignment=2, MarginV=15, PrimaryColour='&H0000FF00', SecondaryColour='&H000000FF'),
                      Style.new('CN', Fontname='Arial', Fontsize=60, Outline=0, Shadow=0,
                     Alignment=2, MarginV=90, PrimaryColour='&H00FF0000')]
        return ass

    def test_font_switched_syllables_complete_for_k_kf_ko(self):
        for name in ('k', 'kf', 'ko'):
            ass = self.template()
            text = '{\\'+name+r'81}H{\fnTimes New Roman}Hg'
            ass.events = [Event(start=100, end=400, style='JP', text=split_karaoke_overrides(text))]
            sung = self.render(ass)
            ass.events[0].text = remove_tags_named(text, KARAOKE_TAGS)
            expected = self.render(ass)
            self.assertTrue(np.any(expected[:, :, 1] > 128))
            np.testing.assert_array_equal(sung, expected)

    def test_later_line_uses_its_own_position_at_overlap(self):
        ass = self.template()
        doc = Document('render', groups=[Group('a','song',['a']), Group('b','song',['b'])])
        for mid, start, end in (('a',1000,3000), ('b',2980,5000)):
            doc.members[mid] = Member(mid,mid,'ja','original','song',start,end,style='JP')
        ass.events = [Event(start=100,end=300,style='JP',text='Earlier'), Event(start=298,end=500,style='JP',text='Later')]
        owners = [{'member':mid,'kind':'base'} for mid in ('a','b')]
        _finish_song_events(doc, ass.events, owners, {mid:_MemberOut(e) for mid,e in zip(('a','b'),ass.events)})
        together = self.render(ass,2.99)
        ass.events = [ass.events[1]]
        np.testing.assert_array_equal(together,self.render(ass,2.99))

    def test_tall_font_does_not_push_original_above_translation(self):
        ass = self.template()
        doc = Document('render', groups=[Group('g','song',['jp','cn'])])
        doc.members = {mid:Member(mid,'g','ja',lane,'song',1000,4000,style=style)
                       for mid,lane,style in (('jp','original','JP'),('cn','translation','CN'))}
        ass.events = [Event(start=100,end=400,style='JP',text=r'H{\fnTimes New Roman\fs95}Hg'),
                      Event(start=100,end=400,style='CN',text='Translation')]
        owners = [{'member':mid,'kind':'base'} for mid in ('jp','cn')]
        _finish_song_events(doc,ass.events,owners,{mid:_MemberOut(e) for mid,e in zip(('jp','cn'),ass.events)})
        together = self.render(ass)
        both = list(ass.events)
        ass.events = [both[0]]
        original = self.render(ass)
        np.testing.assert_array_equal(together[:,:,1] > 128,original[:,:,1] > 128)
        ass.events = [both[1]]
        translation = self.render(ass)
        np.testing.assert_array_equal(together[:,:,2] > 128,translation[:,:,2] > 128)
        self.assertLess(np.where(translation[:,:,2]>128)[0].min(),np.where(original[:,:,1]>128)[0].min())
