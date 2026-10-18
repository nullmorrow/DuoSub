"""Test-only old ruby layout for historical, immutable ASS fingerprints."""
from contextlib import contextmanager
from unittest.mock import patch

from subkit.ruby.generate import RubyGenerator


@contextmanager
def legacy_ruby_layout():
    original = RubyGenerator._layout_job

    def layout(generator, job, boxes, ruby_boxes):
        preferred = job.prefer_center
        advances = job.base_advances
        job.prefer_center = False
        job.base_advances = {}
        try:
            return original(generator, job, boxes, ruby_boxes)
        finally:
            job.prefer_center = preferred
            job.base_advances = advances

    with patch.object(RubyGenerator, "_layout_job", layout):
        yield
