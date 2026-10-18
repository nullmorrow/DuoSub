"""Persistent layout probes invalidate after font/tool changes."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_fontcheck import document
from subkit.ruby import measure
from subkit.ruby.measure import Box, Measurer, forget_measurement_environment, measurement_environment
from subkit.core.export import ExportCache, build_ass
from tests.test_gloss import fixture


class FontKeys(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.font = self.path / "font.ttf"
        self.binary = self.path / "ffmpeg.exe"
        self.binary.write_bytes(b"tool")
        forget_measurement_environment()
        self.addCleanup(forget_measurement_environment)

    def key(self, files=None, binary=None):
        forget_measurement_environment()      # as if the recheck interval had passed
        with patch("subkit.ruby.measure.installed_font_files", return_value=files or {}), \
                patch("subkit.ruby.measure.tool", return_value=str(binary or self.binary)):
            with Measurer(document("A")) as measure:
                return measure._key("Default", "A", 1, 320)

    def test_missing_install_replace_remove_and_mtime(self):
        files = {"arial": [self.font]}
        missing = self.key(files)
        self.font.write_bytes(b"font")
        installed = self.key(files)
        self.assertNotEqual(missing, installed)
        self.font.write_bytes(b"replacement")
        replaced = self.key(files)
        self.assertNotEqual(installed, replaced)
        stat = self.font.stat()
        os.utime(self.font, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        self.assertNotEqual(replaced, self.key(files))
        self.font.unlink()
        self.assertEqual(missing, self.key(files))

    def test_binary_path_size_and_mtime(self):
        original = self.key()
        other = self.binary.with_name("other.exe")
        other.write_bytes(b"tool")
        self.assertNotEqual(original, self.key(binary=other))
        self.binary.write_bytes(b"different tool")
        resized = self.key()
        self.assertNotEqual(original, resized)
        stat = self.binary.stat()
        os.utime(self.binary, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        self.assertNotEqual(resized, self.key())

    def test_fallback_and_inline_fonts_invalidate_persistent_boxes(self):
        cache = self.path / "render.sqlite3"
        files = {"fallback": [self.font]}
        with patch("subkit.ruby.measure.installed_font_files", return_value=files):
            with Measurer(document("A"), cache_path=cache) as first:
                key = first._key("Default", r"{\fnMissing}A", 1, 320)
                first._db.execute("INSERT INTO probe VALUES (?,?)", (key, "[[1,2,3,4]]"))
                first._db.commit()
            with Measurer(document("A"), cache_path=cache) as second:
                self.assertEqual(second._get(key), [Box(1,2,3,4)])
            self.font.write_bytes(b"new fallback font")
            forget_measurement_environment()
            with Measurer(document("A"), cache_path=cache) as third:
                changed = third._key("Default", r"{\fnMissing}A", 1, 320)
                self.assertNotEqual(key, changed)
                self.assertIsNone(third._get(changed))

    def test_export_memory_cache_also_invalidates(self):
        doc, tpl = fixture("A")
        cache = ExportCache()
        with patch("subkit.core.export.measurement_environment", return_value=("old",)):
            self.assertEqual(build_ass(doc, {}, tpl, self.path, cache=cache)[1].laid_out, 1)
            self.assertEqual(build_ass(doc, {}, tpl, self.path, cache=cache)[1].laid_out, 0)
        with patch("subkit.core.export.measurement_environment", return_value=("new",)):
            self.assertEqual(build_ass(doc, {}, tpl, self.path, cache=cache)[1].laid_out, 1)

    def test_unchanged_environment_is_not_restatted_every_preview(self):
        fonts = [self.path / f"f{i}.ttf" for i in range(6)]
        for font in fonts:
            font.write_bytes(b"font")
        with patch("subkit.ruby.measure.installed_font_files", return_value={"f": fonts}),                 patch("subkit.ruby.measure.tool", return_value=str(self.binary)),                 patch("subkit.ruby.measure.time.monotonic", side_effect=[0.0, 1.0, 20.0, 400.0]),                 patch("subkit.ruby.measure.file_signature", wraps=measure.file_signature) as signature:
            first = measurement_environment()
            calls = signature.call_count
            self.assertEqual(measurement_environment(), first)      # within the recheck interval: no stat
            self.assertEqual(signature.call_count, calls)
            self.assertEqual(measurement_environment(), first)      # unchanged marker: tool and folders only
            self.assertLessEqual(signature.call_count - calls, 3)
            calls = signature.call_count
            self.assertEqual(measurement_environment(), first)      # periodic full re-stat
            self.assertGreaterEqual(signature.call_count - calls, len(fonts))
