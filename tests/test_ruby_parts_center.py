"""Colliding parts share a word centre without changing per-part output."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.test_ruby_word_boundaries import fake_job
from subkit.ass.document import AssDocument, Event, Style
from subkit.ass.text import line_tag
from subkit.ruby.generate import RubyGenerator, RubyJob, RubyPiece
from subkit.ruby.measure import Box


def measured(*, widths=(94, 38), scales=(None, None), percent=150, legacy=False,
             prefer_center=True, vertical=False, single=False):
    doc = AssDocument()
    doc.styles = [Style.new('JP', Fontsize=48, Outline=1, Shadow=0,
                            Fontname='@MS Gothic' if vertical else 'MS Gothic', Angle=270 if vertical else 0),
                  Style.new('JP.Ruby', Fontsize=24, Outline=1.7, Shadow=0)]
    gen = RubyGenerator(doc, SimpleNamespace(playres_x=1920, playres_y=1080, anchor_x=lambda *a: 960))
    gen._metrics = {'JP': (60, 5, 50), 'JP.Ruby': (30, 2, 25)}
    pieces = ([RubyPiece(0, 3, 'じゅうなんせい', tag='word')] if single else
              [RubyPiece(0, 2, 'じゅうなん', scale=scales[0], tag='word', k_text=r'{\k20}じゅうなん'),
               RubyPiece(2, 3, 'せい', scale=scales[1], tag='word', k_text=r'{\k10}せい')])
    job = RubyJob(Event(style='JP', text='柔軟性'), 'JP.Ruby', pieces,
                  max_width_percent=percent, prefer_center=prefer_center)
    job.base_advances = {(0, 2): 80, (2, 3): 40, (0, 3): 120}
    boxes = [Box(955, 992, 5, 50), Box(995, 1032, 5, 50), Box(1035, 1072, 5, 50)]
    rubies = [Box(960-w/2, 960+w/2, 2, 26) for w in widths]
    if legacy:
        with patch.object(gen, '_merge_word_parts', side_effect=lambda pairs, *args: (pairs, {})):
            gen._layout_job(job, boxes, rubies)
    else:
        gen._layout_job(job, boxes, rubies)
    return job


def extents(job, widths):
    return [(x-(w+3.4)*scale/200, x+(w+3.4)*scale/200)
            for (_p, x, scale, _overlap), w in zip(job.placements, widths)]


class MultipartWordGeometry(unittest.TestCase):
    def assert_merged(self, job, widths=(94, 38)):
        edges = extents(job, widths)
        self.assertAlmostEqual((edges[0][0]+edges[-1][1])/2, 1013.5, delta=0.5)
        self.assertAlmostEqual(edges[1][0]-edges[0][1], 2, places=6)
        self.assertEqual(job.placements[0][2], job.placements[1][2])
        self.assertEqual(len(job.out), 2)
        return edges

    def test_measured_flexibility_centres_and_matches_single_reading(self):
        job = measured()
        edges = self.assert_merged(job)
        single = measured(single=True, widths=(136,))
        single_edges = extents(single, (136,))[0]
        for actual, expected in zip((edges[0][0], edges[-1][1]), single_edges):
            self.assertAlmostEqual(actual, expected, delta=2)
        self.assertAlmostEqual(80*150/94, 127.7, delta=0.1)
        self.assertTrue(all(r'\k' in event.text for event in job.out))
        self.assertEqual([g['base_start'] for g in job.geoms], [0, 2])
        self.assertEqual([g['ann'] for g in job.geoms], ['word', 'word'])

    def test_noncolliding_parts_also_share_one_word_centre(self):
        current = measured(widths=(35, 20))
        self.assert_merged(current, widths=(35, 20))
        self.assertEqual([p[2] for p in current.placements], [100, 100])

    def test_same_manual_scale_is_shared(self):
        job = measured(scales=(125, 125))
        self.assert_merged(job)
        self.assertEqual([p[2] for p in job.placements], [125, 125])

    def test_one_manual_scale_is_shared_with_automatic_part(self):
        job = measured(scales=(125, None))
        self.assert_merged(job)
        self.assertEqual([p[2] for p in job.placements], [125, 125])
        self.assertTrue(all(line_tag(e.text, 'fscx').num() == 125 for e in job.out))

    def test_conflicting_manual_scales_keep_legacy_bytes(self):
        current = measured(scales=(90, 125))
        legacy = measured(scales=(90, 125), legacy=True)
        self.assertEqual(current.out, legacy.out)
        self.assertEqual(current.placements, legacy.placements)

    def test_combined_advance_cap_and_fixed_gap_when_compressed(self):
        job = measured(percent=100)
        self.assert_merged(job)
        self.assertAlmostEqual(job.placements[0][2], 120*100/(94+38+2))
        self.assertTrue(all(line_tag(e.text, 'fscx').num() <= p[2]
                            for e, p in zip(job.out, job.placements)))

    def test_combined_cap_respects_minimum_scale(self):
        job = measured(widths=(294, 138), percent=80)
        self.assert_merged(job, (294, 138))
        self.assertEqual([p[2] for p in job.placements], [60, 60])
        self.assertTrue(any('下限' in warning for warning in job.warnings))

    def test_disabled_cap_still_merges(self):
        self.assert_merged(measured(percent=None))

    def test_legacy_bands_and_vertical_keep_bytes(self):
        for options in ({'prefer_center': False}, {'vertical': True}):
            with self.subTest(options=options):
                current, legacy = measured(**options), measured(legacy=True, **options)
                self.assertEqual(current.out, legacy.out)
                self.assertEqual(current.geoms, legacy.geoms)

    def test_all_parts_of_word_merge_when_only_one_pair_collides(self):
        pieces = [RubyPiece(0, 2, 'だいひょう', tag='w'), RubyPiece(2, 5, 'とりしまりやく', tag='w'),
                  RubyPiece(5, 7, 'しゃちょう', tag='w')]
        job, _boxes, _rubies, lines = fake_job(pieces=pieces, widths=(30, 200, 40), percent=None)
        self.assertEqual(len(lines[0]), 1)
        self.assertEqual(len(job.placements), 3)
        self.assertEqual(len({p[2] for p in job.placements}), 1)
        self.assertAlmostEqual((job.geoms[0]['x']-15 + job.geoms[-1]['x']+20)/2, 1040)

    def test_neighbouring_word_keeps_gap_under_compression_and_floor(self):
        for widths in ((200, 140, 100), (500, 500, 500)):
            with self.subTest(widths=widths):
                pieces = [RubyPiece(0, 2, 'だいひょう', tag='a'), RubyPiece(2, 5, 'とりしまりやく', tag='a'),
                          RubyPiece(5, 7, 'しゃちょう', tag='b')]
                job, _boxes, _rubies, lines = fake_job(pieces=pieces, widths=widths, percent=None)
                self.assertTrue(all(p[2] >= 60 for p in job.placements))
                items = lines[0]
                self.assertGreaterEqual(items[1].x-items[1].w()/2-items[0].x-items[0].w()/2, 7.2-1e-5)
                self.assertEqual(job.placements[0][2], job.placements[1][2])

    def test_interword_caps_apply_to_the_whole_word_before_its_internal_parts(self):
        pieces = [RubyPiece(0, 2, 'first', tag='a'), RubyPiece(2, 3, 'second', tag='a'),
                  RubyPiece(3, 5, 'next', tag='b')]
        current, _boxes, _ruby, lines = fake_job(text='abcde', pieces=pieces, widths=(76, 54, 94))
        self.assertEqual(len(lines[0]), 2)
        first, second, neighbour = current.placements
        self.assertAlmostEqual(first[2], second[2])
        self.assertGreater(first[2], 60)
        self.assertLess(first[2], 100)
        edges = [(x-(w+2)*scale/200, x+(w+2)*scale/200)
                 for (_piece, x, scale, _overlap), w in zip(current.placements, (76, 54, 94))]
        self.assertAlmostEqual((edges[0][0]+edges[1][1])/2, 960, places=6)
        self.assertAlmostEqual(edges[1][0]-edges[0][1], 2, places=6)
        self.assertGreaterEqual(edges[2][0]-edges[1][1], 7.2-1e-6)
        self.assertAlmostEqual(neighbour[1], 1060, places=6)

    def test_unannotated_okurigana_between_parts_keeps_its_own_position(self):
        pieces = [RubyPiece(0, 1, 'お', tag='one-word'), RubyPiece(2, 3, 'つ', tag='one-word')]
        job, _boxes, _ruby, lines = fake_job(text='落ち着く', pieces=pieces, widths=(20, 20))
        self.assertEqual(len(lines[0]), 2)
        self.assertEqual([p[1] for p in job.placements], [920, 1000])
        self.assertEqual([g['base_start'] for g in job.geoms], [0, 2])

    def test_word_parts_on_separate_visual_lines_are_not_combined(self):
        pieces = [RubyPiece(0, 2, 'かいはつ', tag='one-word'), RubyPiece(2, 3, 'りょく', tag='one-word')]
        job, _boxes, _ruby, lines = fake_job(text=r'開発\N力', pieces=pieces, widths=(50, 30))
        self.assertEqual([len(items) for items in lines], [1, 1])
        self.assertEqual(len(job.placements), 2)
        self.assertNotEqual(job.geoms[0]['y'], job.geoms[1]['y'])
