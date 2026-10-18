"""Release font attribution, complete libass probes and build/write export invariants."""

from __future__ import annotations

import copy
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit.ass import AssDocument, Event, Style  # noqa: E402
from subkit.core.fontcheck import FontCheck, check_fonts, font_runs, probe_ass, prompt_text, report_lines  # noqa: E402
from subkit.core.workspace import Project  # noqa: E402
from subkit.model.document import Document, Group, Member, Run  # noqa: E402
from subkit.model.source import Source  # noqa: E402
from tests.test_templates import UNDRAWABLE_LOG  # noqa: E402


def document(*texts: str, font: str = "Arial") -> AssDocument:
    doc = AssDocument()
    doc.script_info = [("ScriptType", "v4.00+"), ("PlayResX", "1920"), ("PlayResY", "1080"),
                       ("WrapStyle", "2"), ("ScaledBorderAndShadow", "yes")]
    doc.styles = [Style.new("Default", Fontname=font), Style.new("Other", Fontname="Times New Roman")]
    doc.events = [Event(start=i * 100 + 125, end=i * 100 + 225, text=text) for i, text in enumerate(texts)]
    return doc


class Attribution(unittest.TestCase):
    def test_style_and_font_resets(self) -> None:
        doc = document(r"A{\fn Another Font }B{\fn}C{\fnX}D{\fn0}E", font="Base Font")
        self.assertEqual(list(font_runs(doc.events[0], doc)),
                         [("Base Font", "A"), ("Another Font", "B"), ("Base Font", "C"),
                          ("X", "D"), ("Base Font", "E")])

    def test_style_resets_and_unknown_style(self) -> None:
        doc = document(r"A{\fnX}B{\r}C{\rOther}D{\fnY}E{\rNoSuchStyle}F{\rOther\fn0}G", font="Base")
        self.assertEqual(list(font_runs(doc.events[0], doc)),
                         [("Base", "A"), ("X", "B"), ("Base", "C"), ("Times New Roman", "D"),
                          ("Y", "E"), ("Base", "F"), ("Base", "G")])

    def test_missing_event_style(self) -> None:
        doc = document("A{\\fnX}B{\\r}C")
        doc.events[0].style = "Missing"
        self.assertEqual(list(font_runs(doc.events[0], doc)), [("Arial", "A"), ("X", "B"), ("Arial", "C")])
        doc.styles = []
        self.assertEqual(list(font_runs(doc.events[0], doc)), [("", "A"), ("X", "B"), ("", "C")])

    def test_drawing_comments_breaks_and_hardspace(self) -> None:
        doc = document(r"A{comment 脱}{\p1\fnDrawing}m 0 0 l 10 10\h{\p0\fnText}B\hC\ND\nE")
        self.assertEqual(list(font_runs(doc.events[0], doc)),
                         [("Arial", "A"), ("Text", "B"), ("Text", "\u00a0"), ("Text", "C"),
                          ("Text", "D"), ("Text", "E")])

    def test_canned_log_attribution_and_family_comparison(self) -> None:
        doc = document(r"ABC{\fn@missing family}晚", r"{\rOther}脱{\r}A", "脱", "\U0010fffd",
                       "脱{comment}" + "x" * 30)
        doc.events[0].kind = "Comment"
        doc.events.append(Event(start=500, end=600, text=r"{\fn Missing Family}late"))
        doc.events.append(doc.events[-1].copy())   # identical reference is reported only once
        doc.styles[0].set("Fontname", "@aRiAl")
        log = ("fontselect: (ARIAL, 400, 0) -> ArialMT, 0, ArialMT\n"
               "fontselect: (@MISSING FAMILY, 400, 0) -> ArialMT, 0, ArialMT\n"
               "Glyph 0x8131 not found, selecting one more font for (ARIAL, 400, 0)\n"
               "Glyph 0x665A not found, selecting one more font for (@MISSING FAMILY, 400, 0)\n"
               "fontselect: (Times New Roman, 400, 0) -> TimesNewRomanPSMT, 0, TimesNewRomanPSMT\n"
               + UNDRAWABLE_LOG)
        with patch("subkit.core.fontcheck.run", return_value=subprocess.CompletedProcess([], 0, b"", log.encode())) as run:
            fc = check_fonts(doc)
        self.assertTrue(fc.ran)
        self.assertFalse(fc.ok)
        self.assertEqual(fc.families, ["aRiAl", "Missing Family", "Times New Roman"])
        self.assertEqual(fc.missing, {"Missing Family": ["0:00:05.00 [Default] late"]})
        self.assertEqual(fc.gaps, {"aRiAl": ("脱", ["0:00:03.25 [Default] 脱", "0:00:05.25 [Default] 脱" + "x" * 23])})
        self.assertEqual(fc.undrawable, {"aRiAl": ("\U0010fffd", ["0:00:04.25 [Default] \U0010fffd"])})
        self.assertGreaterEqual(fc.elapsed, 0)
        run.assert_called_once()
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[0], "ffmpeg")
        self.assertIn("verbose", cmd)
        self.assertIn("ass=probe.ass", cmd)
        self.assertFalse(Path(run.call_args.kwargs["cwd"]).exists())

    def test_gap_is_attributed_only_to_the_current_family(self) -> None:
        doc = document(r"{\fnOther}脱", r"脱{\fnOther}胎", r"{\fn Arial }胎")
        log = ("fontselect: (Arial, 400, 0) -> ArialMT, 0, ArialMT\n"
               "Glyph 0x8131 not found, selecting one more font for (Arial, 400, 0)\n"
               "Glyph 0x80CE not found, selecting one more font for (Arial, 400, 0)\n"
               "Glyph 0x80ce not found, selecting one more font for (@ARIAL, 700, 0)\n")
        with patch("subkit.core.fontcheck.run", return_value=subprocess.CompletedProcess([], 0, b"", log.encode())):
            fc = check_fonts(doc)
        self.assertEqual(fc.gaps, {"Arial": ("脱胎", ["0:00:02.25 [Default] 脱胎", "0:00:03.25 [Default] 胎"])})


class Probe(unittest.TestCase):
    def test_all_dialogues_hit_frames_without_mutating_document(self) -> None:
        doc = document(*[r"{\fnArial\k20}原文\N\h{comment}{\p1}m 0 0 l 10 10{\p0}" + str(i) for i in range(377)])
        doc.events.insert(100, Event(kind="Comment", text="comment must not render"))
        doc.events.append(Event(kind="Picture", text="not a Dialogue"))
        before = copy.deepcopy(doc.__dict__)
        probe = AssDocument.parse(probe_ass(doc))
        self.assertEqual(doc.__dict__, before)
        self.assertEqual(probe.script_info, doc.script_info)
        self.assertEqual(probe.styles, doc.styles)
        self.assertEqual([e.text.encode("utf-8") for e in probe.events],
                         [e.text.encode("utf-8") for e in doc.dialogues()])
        self.assertTrue(all(e.is_dialogue for e in probe.events))
        frame_times = range(0, max(e.end for e in probe.events), 4)
        self.assertTrue(all(any(e.start <= t < e.end for t in frame_times) for e in probe.events))

    def test_slots_survive_ffmpeg_time_truncation(self) -> None:
        from subkit.core.fontcheck import _EVENTS_PER_FRAME, _slot

        for frame in range(5000):                       # vf_subtitles: (long long)(pts * av_q2d(1/25) * 1000)
            now = int(frame * (1 / 25) * 1000)
            starts, ends = zip(*(_slot(frame * _EVENTS_PER_FRAME + k) for k in (0, _EVENTS_PER_FRAME - 1)))
            self.assertTrue(all(s * 10 <= now < e * 10 for s, e in zip(starts, ends)), frame)
            if frame:
                self.assertLessEqual(_slot(frame * _EVENTS_PER_FRAME - 1)[1] * 10, now, frame)

    def test_empty_probe(self) -> None:
        doc = document()
        self.assertEqual(AssDocument.parse(probe_ass(doc)).events, [])


class Failures(unittest.TestCase):
    def test_nonzero_exit_is_unverified(self) -> None:
        with patch("subkit.core.fontcheck.run", return_value=subprocess.CompletedProcess([], 7, b"", b"Error opening filter\n")):
            fc = check_fonts(document("A"))
        self.assertFalse(fc.ran)
        self.assertFalse(fc.ok)
        self.assertIn("退出码 7", fc.error)
        self.assertIn("Error opening filter", fc.error)
        self.assertIn("缺字体、缺字没有验证", report_lines(fc)[0])
        self.assertNotIn("没有缺字", "".join(report_lines(fc)))

    def test_oserror_is_unverified(self) -> None:
        with patch("subkit.core.fontcheck.run", side_effect=OSError("cannot start tool")):
            fc = check_fonts(document("A"))
        self.assertFalse(fc.ran)
        self.assertIn("cannot start tool", fc.error)

    def test_missing_executable_is_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fc = check_fonts(document("A"), ffmpeg=str(Path(tmp, "not-installed-ffmpeg.exe")))
        self.assertFalse(fc.ran)
        self.assertFalse(fc.ok)
        self.assertIn("找不到 ffmpeg", fc.error)
        self.assertTrue(report_lines(fc)[0].startswith("字体检查没有运行："))


class Messages(unittest.TestCase):
    def test_ok_and_not_run(self) -> None:
        fc = FontCheck(ran=True, families=["Arial", "Times New Roman"])
        self.assertTrue(fc.ok)
        self.assertEqual(report_lines(fc), ["字体：本机都已安装，没有缺字（libass 实测）", "用到的字体：Arial、Times New Roman"])
        fc = FontCheck(error="找不到 ffmpeg")
        self.assertFalse(fc.ok)
        self.assertEqual(report_lines(fc), ["字体检查没有运行：找不到 ffmpeg（缺字体、缺字没有验证）"])
        self.assertNotIn("仍然导出", prompt_text(fc))

    def test_problems_and_invisible_characters(self) -> None:
        fc = FontCheck(ran=True, families=["A", "Arial", "B"], missing={"A": ["ref 1"], "B": ["ref 1", "ref 2"]},
                       gaps={"Arial": ("脱脱 \u00a0\t\u200b\ufe0f", ["ref 3"])},
                       undrawable={"Arial": ("\U0010fffd", ["ref 4"])})
        lines = report_lines(fc)
        self.assertIn("没有安装：A、B——用到的 2 行会换成别的字体显示，振假名位置也是按替代字体测量的。", lines)
        self.assertIn("  例：ref 1；ref 2", lines)
        self.assertIn("Arial 缺字：「脱U+0020U+00A0U+0009U+200BU+FE0F」", "\n".join(lines))
        self.assertIn("没有任何字体能显示：「U+10FFFD」（1 行，例：ref 4）", lines)
        self.assertEqual(lines[-1], "用到的字体：A、Arial、B")
        text = prompt_text(fc)
        self.assertIn("本机 libass 实测", text)
        self.assertTrue(text.endswith("\n\n仍然导出吗？（会写进导出报告；双语字幕工坊不会替换字体）"))
        self.assertNotIn("仍然导出", "".join(lines))

    def test_prompt_has_one_shared_example_budget(self) -> None:
        refs = [f"REF{i:02d}" for i in range(30)]
        fc = FontCheck(ran=True, missing={"A": refs[:9]}, gaps={"B": ("脱", refs[9:20])},
                       undrawable={"C": ("\U0010fffd", refs[20:])})
        text = prompt_text(fc)
        self.assertEqual(sum(ref in text for ref in refs), 12)
        self.assertIn("REF11…", text)
        self.assertEqual(sum(ref in prompt_text(fc, limit=2) for ref in refs), 2)
        self.assertEqual(sum(ref in prompt_text(fc, limit=0) for ref in refs), 0)
        self.assertEqual(sum(ref in "\n".join(report_lines(fc)) for ref in refs), 30)


def episode(root: Path):
    template = document().serialize()
    project = Project.create(root / "project", "test", template_text=template)
    ep = project.add_episode("E1")
    member = Member("m", "g", "en", "original", "dialogue", 1230, 3450, style="Default",
                    runs=[Run(r"A{\fnArial}B\NC", "text", ins="i", a=0, b=16)])
    ep.document = Document(episode="E1", groups=[Group("g", members=["m"])], members={"m": member})
    return ep


class ExportSplit(unittest.TestCase):
    def test_export_matches_build_then_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ep = episode(root)
            path, report = ep.export(root / "legacy.ass", mode="work", include_actor=False, blur=0.6, video_height=720)
            ass, built_report = ep.build_export(mode="work", include_actor=False, blur=0.6, video_height=720)
            built_path = ep.write_export(ass, root / "built.ass")
            self.assertEqual(path.read_bytes(), built_path.read_bytes())
            self.assertEqual(report, built_report)
            self.assertEqual(ass.events[0].text, r"{\blur0.6}A{\fnArial}B\NC")

    def test_write_export_refuses_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ep = episode(root)
            source_path = root / "source.ass"
            source_path.write_bytes(b"original source bytes")
            ep.sources["s"] = Source(id="s", lang="en", lane="original", format="ass", path=str(source_path),
                                     copy="sources/s.ass", sha256="", bytes=21, encoding="utf-8", newline="\n")
            with self.assertRaisesRegex(PermissionError, "不能覆盖原料字幕"):
                ep.write_export(document("modified"), root / "project" / ".." / "source.ass")
            self.assertEqual(source_path.read_bytes(), b"original source bytes")


@unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg (libass)")
class Libass(unittest.TestCase):
    def test_late_event_font_is_checked(self) -> None:
        doc = document(*[f"Event {i}" for i in range(349)], r"{\fnSubKit 不存在的字体}late font")
        fc = check_fonts(doc)
        self.assertTrue(fc.ran, fc.error)
        self.assertEqual(fc.missing, {"SubKit 不存在的字体": ["0:05:50.25 [Default] late font"]})
        self.assertNotIn("SubKit 不存在的字体", fc.gaps)

    def test_arial_gap_has_only_its_line(self) -> None:
        doc = document("Latin", "脱", r"{\fnMicrosoft YaHei}脱")
        fc = check_fonts(doc)
        self.assertTrue(fc.ran, fc.error)
        self.assertEqual(fc.missing, {})
        self.assertEqual(fc.gaps, {"Arial": ("脱", ["0:00:02.25 [Default] 脱"])})

    def test_installed_fonts(self) -> None:
        doc = document("Arial only", r"{\rOther}Times New Roman")
        fc = check_fonts(doc)
        self.assertTrue(fc.ok, report_lines(fc))
        self.assertEqual(fc.families, ["Arial", "Times New Roman"])

    def test_undrawable_character(self) -> None:
        doc = document("Latin", "\U0010fffd", r"{comment}" + "\U0010fffd")
        fc = check_fonts(doc)
        self.assertTrue(fc.ran, fc.error)
        self.assertEqual(fc.undrawable, {"Arial": ("\U0010fffd", ["0:00:02.25 [Default] \U0010fffd",
                                                                "0:00:03.25 [Default] \U0010fffd"])})
        self.assertEqual(fc.gaps, {})

    def test_check_does_not_change_written_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ep = episode(root)
            ass, _ = ep.build_export()
            before = copy.deepcopy(ass.__dict__)
            first = ep.write_export(ass, root / "before.ass").read_bytes()
            fc = check_fonts(ass)
            self.assertTrue(fc.ran, fc.error)
            self.assertEqual(ass.__dict__, before)
            self.assertEqual(first, ep.write_export(ass, root / "after.ass").read_bytes())

    def test_1500_events_runtime(self) -> None:
        doc = document(*[f"Dialogue {i}: font probe text" for i in range(1500)])
        fc = check_fonts(doc)
        self.assertTrue(fc.ok, report_lines(fc))
        self.assertLessEqual(fc.elapsed, 30)
        print(f"\nFont probe: 1500 Dialogue events, {fc.elapsed:.3f} s")


if __name__ == "__main__":
    unittest.main()
