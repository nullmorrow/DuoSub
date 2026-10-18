"""ASS rendering context retained through copies/edits, independent of the character ledger.

Copied characters are insertions. Their old source styles, coordinate system and
file-level Name decisions still matter to rendering, but must never count as a
second occurrence of source text. Optional Run.render_src/render_e travel with
the run through moves and splits; character identity and the ledger use Run.src.
An empty insertion tag can retain the line's original source context after all
its source characters are replaced, without rescaling newly typed ASS tags.
"""

from __future__ import annotations

from ..model.document import Comment, Member, Run


def run_source(member: Member | Comment, run: Run) -> str | None:
    return run.src or run.render_src


def source_id(member: Member) -> str | None:
    return next((sid for r in member.runs if (sid := run_source(member, r))), None)


def source_events(member: Member | Comment) -> set[tuple[str, int]]:
    return {(sid, event) for r in member.runs
            if (sid := run_source(member, r)) is not None
            and (event := r.e if r.src is not None else r.render_e) is not None}
