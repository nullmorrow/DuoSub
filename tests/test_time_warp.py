"""The time relation between the original captions and the translation: constant offset, jumps, speed ratio.

Synthetic timelines (a seeded random dialogue rhythm): the translation's lines start where the original's start,
mapped by a known function plus jitter; the tracker must recover the mapping.
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit.align.dp2 import Row, time_warp  # noqa: E402
from subkit.core.units import Unit  # noqa: E402


def timeline(seed: int = 7, minutes: int = 24) -> list[tuple[int, int]]:
    rnd = random.Random(seed)
    t, out = 5000, []
    while t < minutes * 60000:
        d = rnd.randint(900, 4200)
        out.append((t, t + d))
        t += d + int(rnd.expovariate(1 / 1400.0)) + 150
        if rnd.random() < 0.03:
            t += rnd.randint(4000, 15000)             # scene without dialogue
    return out


def make(jp_lines, mapping, seed: int = 3):
    rnd = random.Random(seed)
    units = [Unit(i, "o1", i, 0, 0, 4, "テスト", a, b) for i, (a, b) in enumerate(jp_lines)]
    rows = []
    for i, (a, b) in enumerate(jp_lines):
        if rnd.random() < 0.12:
            continue                                   # lines the translation merged / left out
        j = rnd.randint(-120, 120)
        rows.append(Row(len(rows), int(mapping(a)) + j, int(mapping(b)) + j, "测试", i))
    return rows, units


class TimeWarp(unittest.TestCase):
    def check(self, mapping, *, tol: int = 450, label: str = "") -> None:
        jp = timeline()
        rows, units = make(jp, mapping)
        warp, _off, rep = time_warp(rows, units)
        bad = [(a, warp(a), mapping(a)) for a, _b in jp if abs(warp(a) - mapping(a)) > tol]
        self.assertLessEqual(len(bad), len(jp) * 0.02, (label, bad[:5], rep.lines() if rep else None))

    def test_constant_offset(self) -> None:
        self.check(lambda t: t - 2900, label="−2.9 s")

    def test_jump_bd_added_a_scene(self) -> None:
        self.check(lambda t: t + 1000 + (6000 if t > 600000 else 0), label="+6 s at 10:00")

    def test_two_jumps(self) -> None:
        self.check(lambda t: t + (30000 if t > 400000 else 0) - (8000 if t > 1000000 else 0), label="+30 s, −8 s")

    def test_speed_ratio(self) -> None:
        self.check(lambda t: t * 23976 / 25000 + 500, label="25 ↔ 23.976 fps")

    def test_report_names_the_jump(self) -> None:
        jp = timeline()
        rows, units = make(jp, lambda t: t + (6000 if t > 600000 else 0))
        _w, _o, rep = time_warp(rows, units)
        self.assertEqual(len(rep.jumps), 1)
        at, before, after = rep.jumps[0]
        self.assertLess(abs(at - 600000), 30000)
        self.assertLess(abs((after - before) - 6000), 400)


if __name__ == "__main__":
    unittest.main()
