"""Text-box representation: expose editable gloss marks, keep ASS tags hidden."""

import re

from ..ass.text import tokenize

GLOSS_MARK = re.compile(r"\{@(?:注|g|gloss)(?:\s*[:：][^{}]*|)\}", re.IGNORECASE)


def is_gloss_run(run):
    return run.kind == "tag" and GLOSS_MARK.fullmatch(run.text) is not None


def is_private_run(run):
    return is_gloss_run(run) or run.kind == "tag" and run.text in ("{@Raw=", "}")


def editing_text(member):
    return "".join(r.text for r in member.runs if r.kind != "tag" or is_private_run(r))


def ordinary_offset(text, position):
    """Text-box cursor offset with complete gloss marks excluded, including their interior."""
    return position - sum(max(0, min(position, b) - a) for a, b in _private_ranges(text))


def _private_ranges(text):
    tokens = tokenize(text)
    gloss = [(m.start(), m.end()) for m in GLOSS_MARK.finditer(text)
             if not any(t.protected and t.start <= m.start() < t.end for t in tokens)]
    raw = [(t.start, t.end) for t in tokens if t.protected and t.kind == "comment"]
    return sorted(gloss + raw)


def editing_runs(text, insertion_id):
    from ..model.document import Run

    runs, end = [], 0
    for a, b in _private_ranges(text):
        if a > end:
            runs.append(Run(text[end:a], ins=insertion_id, a=end, b=a))
        runs.append(Run(text[a:b], "tag", ins=insertion_id, a=a, b=b))
        end = b
    if end < len(text):
        runs.append(Run(text[end:], ins=insertion_id, a=end, b=len(text)))
    return runs


def split_private_marks(text):
    """Return ordinary text and (ordinary-text offset, raw marker) pairs."""
    parts, marks, end, offset = [], [], 0, 0
    for a, b in _private_ranges(text):
        part = text[end:a]
        parts.append(part)
        offset += len(part)
        marks.append((offset, text[a:b]))
        end = b
    parts.append(text[end:])
    return "".join(parts), marks
