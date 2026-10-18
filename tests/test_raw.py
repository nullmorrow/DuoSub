"""Raw regions: visible identities, display-rule exclusion and Windows libass output."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import Mock

from tests import test_punctuation as punctuation

document, rules = punctuation.document, punctuation.rules
from subkit.ass.state import state_at
from subkit.ass.text import VisibleText, private_blocks, remove_private, tokenize
from subkit.core.display import DisplayRules, PunctuationRules, punctuation_text, single_line_text, traditional_text
from subkit.core.editor_text import editing_runs, editing_text, ordinary_offset
from subkit.core.layout_index import LayoutIndex
from subkit.core.merge import runs_of_event
from subkit.core.ops import Editor
from subkit.core.readings import anchor_of, plain_range_to_visible, reading_spans, visible_map
from subkit.model.document import Annotation
from subkit.model.serialize import document_from_json, document_to_json
from subkit.model.source import Source, SourceEvent


class RawCore(unittest.TestCase):
    def test_tokens_markers_breaks_offsets_and_private_removal(self):
        text = r'A{@Raw=漢\N字\h!}B{@注:漢=word}'
        tokens = tokenize(text)
        self.assertEqual(''.join(t.raw for t in tokens), text)
        for t in tokens:
            self.assertEqual(text[t.start:t.end], t.raw)
        self.assertEqual(VisibleText(text).lines(), ['A漢', '字 !B'])
        self.assertEqual(remove_private(text), r'A漢\N字\h!B')
        self.assertEqual(remove_private(text, keep_raw=True), r'A{@Raw=漢\N字\h!}B')
        self.assertEqual([(p, body) for p, body, _ in private_blocks(text)], [(6, '注:漢=word')])
        self.assertEqual(remove_private('A{@Raw=}B{@Raw=字}'), 'AB字')

    def test_all_rules_skip_raw_including_spaces_and_font_positions(self):
        raw = r'{@Raw=  “测试文字”，...!?·。\N　字　 }， “其他”'
        options = PunctuationRules(comma="spaces", dunhao=True, period=True, spaces=True, ellipsis='single',
                                   exclaim='full', question='full', separator=True, quotes='nested', font='Punct')
        text, bounds, fonts, hidden = punctuation_text(raw, options, hide_final_period=True)
        self.assertEqual(text, r'{@Raw=  “测试文字”，...!?·。\N　字　 }  「其他」')
        raw_count = len(VisibleText(raw[:raw.index('}') + 1]))
        self.assertTrue(all(i >= raw_count for i in fonts))
        self.assertEqual(bounds[-1], len(VisibleText(text)))
        self.assertEqual(hidden, 0)
        self.assertEqual(punctuation_text(raw, PunctuationRules())[0], raw)

    def test_traditional_and_merge_keep_raw_break_and_map(self):
        text = r'互联网{@Raw=互联网。\N绿。}\N绿。'
        options = DisplayRules(zh_traditional_enabled=True, zh_traditional_phrases=True)
        converted, bounds = traditional_text(text, options)
        self.assertEqual(converted, r'網際網路{@Raw=互联网。\N绿。}\N綠。')
        self.assertEqual(bounds[-1], len(VisibleText(converted)))
        merged, starts, ends = single_line_text(converted)
        self.assertEqual(merged, r'網際網路{@Raw=互联网。\N绿。}  綠。')
        self.assertEqual(starts[-1], len(VisibleText(merged)))
        self.assertEqual(ends[-1], starts[-1])

    def test_source_and_inserted_markers_are_tag_runs(self):
        text = '前{@Raw=漢字}後'
        src = Source('s', 'ja', 'original', 'ass', '', '', '', 0, 'utf-8', 'crlf',
                     events=[SourceEvent(0, 'Dialogue', 0, 1000, text)])
        for runs in (runs_of_event(src, 0), editing_runs(text, 'i')):
            self.assertEqual([(r.kind, r.text) for r in runs],
                             [('text', '前'), ('tag', '{@Raw='), ('text', '漢字'), ('tag', '}'), ('text', '後')])
            m = document('字', lang='ja', lane='original').members['m_1']
            m.runs = runs
            self.assertEqual(visible_map(m)[0].text, '前漢字後')
            self.assertTrue(all(visible_map(m)[1]))
            self.assertEqual(editing_text(m), text)

    def test_editor_wrap_remove_save_undo_preserves_readings_and_ids(self):
        doc = document('漢字へ', lang='ja', lane='original', style='JP')
        m = doc.members['m_1']
        ids = visible_map(m)[1]
        doc.annotations = {'r': Annotation('r', 'ruby', m.id, anchor_of(ids[:2]), '漢字', 'かんじ',
                                         status='confirmed', origin='manual')}
        saved = copy.deepcopy(doc)
        ed = Editor(doc, {})
        ed.replace_editor_text(m.id, '{@Raw=漢字}へ')
        self.assertEqual(visible_map(m)[1], ids)
        self.assertEqual(doc.annotations, saved.annotations)
        self.assertTrue(ed.history.done[-1].keeps_text)
        self.assertEqual(ordinary_offset(editing_text(m), 8), 2)
        self.assertEqual(plain_range_to_visible(m, 6, 8), (0, 2))
        self.assertIsNone(plain_range_to_visible(m, 0, 6))
        engine = Mock()
        engine.analyze.return_value = []
        reading_spans(doc, m, engine)
        self.assertEqual(engine.analyze.call_args.args[0], '漢字へ')
        restored = document_from_json(document_to_json(doc))
        self.assertEqual(editing_text(restored.members[m.id]), '{@Raw=漢字}へ')
        ed.replace_editor_text(m.id, '漢字へ')
        self.assertEqual(visible_map(m)[1], ids)
        self.assertEqual(doc.annotations, saved.annotations)
        ed.history.undo()
        self.assertEqual(editing_text(doc.members[m.id]), '{@Raw=漢字}へ')
        ed.history.undo()
        self.assertEqual(doc.members, saved.members)
        ed.history.redo()
        self.assertEqual(editing_text(doc.members[m.id]), '{@Raw=漢字}へ')


class RawExport(unittest.TestCase):
    setUp = punctuation.PunctuationExport.setUp
    build = punctuation.PunctuationExport.build

    def test_quotes_comma_font_preview_export_and_no_brace_warning(self):
        doc = document('「{@Raw=“测试文字”}，“其他”」', lang='ja', lane='original', style='JP')
        options = rules(quotes='nested', comma=True, font='Arial')
        ass, rep = self.build(doc, options, ruby=False)
        self.assertEqual(VisibleText(ass.events[0].text).text, '「“测试文字”  『其他』」')
        self.assertNotIn('{@Raw', ass.events[0].text)
        self.assertFalse(rep.warnings)
        for i in range(1, 7):
            self.assertNotEqual(state_at(ass.events[0].text, i, ['fn'])[0].get('fn').arg, 'Arial')
        preview, prep = self.build(doc, options, ruby=False, mode='preview')
        self.assertEqual(preview.events, ass.events)
        self.assertEqual(prep.owners, rep.owners)

    def test_traditional_hide_and_merge_with_and_without_punctuation(self):
        for active in (False, True):
            options = rules(comma=active)
            options.zh_script = "traditional"
            options.hide_final_period = True
            options.merge_lines = (('original', 'dialogue'),)
            doc = document(r'{@Raw=测试。\N绿。}\N测试。', lang='zh-Hans', lane='original', style='JP')
            ass, rep = self.build(doc, options, ruby=False)
            self.assertEqual(ass.events[0].text, r'测试。\N绿。  測試')
            self.assertEqual(rep.hidden_by_rules, 1)
            self.assertFalse(rep.warnings)

    def test_raw_furigana_identity_and_actual_libass_hit_testing(self):
        doc = document('漢字へ', lang='ja', lane='original', style='JP')
        m = doc.members['m_1']
        ed = Editor(doc, {})
        ids = visible_map(m)[1]
        doc.annotations = {'r': Annotation('r', 'ruby', m.id, anchor_of(ids[:2]), '漢字', 'かんじ',
                                         parts=[[0, 2, 'かんじ']])}
        ed.replace_editor_text(m.id, '{@Raw=漢字}へ')
        saved = copy.deepcopy(doc)
        ass, rep = self.build(doc, rules(period=True, separator=True))
        self.assertEqual(rep.ruby_lines, 1)
        self.assertEqual(rep.owners[0]['ids'], ids)
        self.assertFalse(rep.warnings)
        index = LayoutIndex(ass, rep.owners, measure_cache=self.path / 'render.sqlite3')
        try:
            geoms = index.geometry(index.events_of_group('g_1'))
            base = next(g for g in geoms if g.kind == 'base')
            for i in range(2):
                box = base.chars[i]
                self.assertIsNotNone(box)
                self.assertEqual(index.hit([base], box.cx, box.cy, pad=0).identity, ids[i])
        finally:
            index.close()
        self.assertEqual(doc, saved)
        ass.save(self.path / 'raw-ruby.ass')

    def test_malformed_raw_literal_and_warning(self):
        for raw in ('{@Raw=漢字', '{@Raw:漢字}', '{@Raw}'):
            doc = document(raw, lang='ja', lane='original', style='JP')
            self.assertEqual(VisibleText(raw).text, raw)
            ass, rep = self.build(doc, ruby=False)
            self.assertEqual(ass.events[0].text, raw)
            self.assertTrue(any('Raw 写法有误' in w for w in rep.warnings), rep.warnings)

    def test_rules_off_raw_preview_preserved_release_only_cleans_edges(self):
        for raw in (r'甲，乙。...!?“丙”{\fnArial}丁\N戊', r'{@Raw= “测试”　。\N字 }，乙'):
            doc = document(raw, lang='ja', lane='original', style='JP')
            off, _ = self.build(doc, ruby=False)
            explicit, _ = self.build(doc, rules(separator=False, quotes='off'), ruby=False)
            self.assertEqual(off.serialize(), explicit.serialize())
            self.assertEqual(off.events[0].text, remove_private(raw).strip(" "))
            preview, _ = self.build(doc, ruby=False, mode="preview")
            self.assertEqual(preview.events[0].text, remove_private(raw))
            self.assertEqual(doc.members["m_1"].text, raw)
