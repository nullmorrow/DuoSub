"""Parse review answers and prepare explicit, undoable acceptances (no Qt)."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

from ..ass.text import VisibleText
from ..languages import get_language, normalize_language_code
from ..model.document import Document, Member
from ..model.serialize import read_json, write_json_atomic
from .display import DisplayRules, member_display_text
from .replacements import replace_ass_text
from .ops import Editor, Transaction
from .readings import anchor_positions, pieces_of, visible_map
from .review import RESPONSE_FORMAT

KINDS = frozenset(("reading", "text", "pairing", "speaker", "persona", "other"))
DECISIONS = frozenset(("accepted", "rejected", "handled"))
FORMAT_ANSWERS = "subkit-review-answers"


@dataclass(frozen=True)
class Answer:
    group: str
    lang: str
    kind: str
    target: str
    old: str
    new: str
    reason: str

    @property
    def fingerprint(self) -> str:
        values = [getattr(self, key) for key in ("group", "lang", "kind", "target", "old", "new")]
        return hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _answer(value) -> Answer:
    if isinstance(value, Answer):
        return value
    if not isinstance(value, dict) or not RESPONSE_FORMAT.keys() <= value.keys():
        raise ValueError("Missing review answer fields")
    if any(isinstance(value[k], (dict, list, bool)) for k in RESPONSE_FORMAT):
        raise ValueError("Review answer fields must be strings or scalar values")
    fields = {k: "" if value[k] is None else str(value[k]) for k in RESPONSE_FORMAT}
    fields["group"] = fields["group"].strip()
    lang = fields["lang"].strip().lower().replace("_", "-")
    fields["lang"] = "zh" if lang == "zh" else normalize_language_code(lang)
    kind = fields["kind"].strip().lower()
    fields["kind"] = kind if kind in KINDS else "other"
    return Answer(**fields)


def parse_answers(text: str) -> tuple[list[Answer], list[str]]:
    """Read JSONL, arrays and multiline/fenced JSON amid prose; retain unreadable raw lines."""
    decoder = json.JSONDecoder()
    items, spans, bad_spans = [], [], []
    end = 0
    for marker in re.finditer(r"[\[{]", text):
        start = marker.start()
        if start < end:
            continue
        try:
            value, end = decoder.raw_decode(text, start)
        except ValueError:
            continue
        spans.append((start, end))
        values = value if isinstance(value, list) else [value]
        for v in values:
            try:
                items.append(_answer(v))
            except ValueError:
                bad_spans.append((start, end))
    unreadable = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stop = offset + len(line)
        residual, cursor = [], offset
        for a, b in spans:
            if b <= offset or a >= stop:
                continue
            residual.append(text[cursor:max(cursor, a)])
            cursor = min(stop, b)
        residual.append(text[cursor:stop])
        raw = line.rstrip("\r\n")
        rest = "".join(residual).strip().lstrip("\ufeff")
        if (any(a < stop and b > offset for a, b in bad_spans) or
                (rest and not rest.startswith("```"))):
            unreadable.append(raw)
        offset = stop
    return items, unreadable


@dataclass(frozen=True)
class Location:
    group: str
    member: str
    a: int                          # original visible offsets, never display offsets
    b: int
    displayed_a: int
    displayed_b: int
    text: str
    identities: tuple
    exact: bool = True


@dataclass
class Suggestion:
    item: Answer
    status: str
    location: Optional[Location] = None
    candidates: list[Location] = field(default_factory=list)
    current: Optional[str] = None
    decision: Optional[str] = None
    display_rules: DisplayRules = field(default_factory=DisplayRules)

    @property
    def fingerprint(self) -> str:
        return self.item.fingerprint


def _shown(m: Member, rules: DisplayRules) -> tuple[str, list[int]]:
    vt, _ids = visible_map(m)
    if rules.converted_for(m) or rules.replaced_for(m):
        raw, bounds = member_display_text(m, rules)
        return VisibleText(raw).text, bounds
    return vt.text, list(range(len(vt.text) + 1))


def _same_language(wanted: str, actual: str) -> bool:
    if wanted == "zh":
        return get_language(actual).code in ("zh-Hans", "zh-Hant")
    return normalize_language_code(wanted) == normalize_language_code(actual)


def _locations(doc: Document, groups, rules: DisplayRules, item: Answer, needle: str) -> list[Location]:
    found = []
    if not needle:
        return found
    for g in groups:
        for m in doc.members_of(g):
            if m.role == "comment" or not _same_language(item.lang, m.lang):
                continue
            text, bounds = _shown(m, rules)
            vt, ids = visible_map(m)
            start = 0
            while (start := text.find(needle, start)) >= 0:
                stop = start + len(needle)
                ranges = [(a, b) for a, v in enumerate(bounds) if v == start
                          for b, w in enumerate(bounds) if w == stop and b > a]
                exact = bool(ranges)
                if not ranges:
                    # A displayed character inside an expanded phrase has no exact source boundary.
                    a = max(i for i, v in enumerate(bounds) if v <= start)
                    b = min(i for i, v in enumerate(bounds) if v >= stop)
                    ranges = [(a, b)]
                for a, b in ranges:
                    touched = replace_ass_text(m.text, rules.replacements, lane=m.lane).ranges
                    safe = exact and not any(x < b and a < z for x, z in touched)
                    found.append(Location(g.id, m.id, a, b, start, stop, vt.text[a:b], tuple(ids[a:b]), safe))
                start += 1
    return found


def _reading(doc: Document, loc: Location) -> Optional[str]:
    m = doc.members[loc.member]
    kind = get_language(m.lang).annotation_system
    if kind == "none":
        return None
    _vt, ids = visible_map(m)
    spans = []
    for ann in doc.annotations.values():
        if ann.member != m.id or ann.kind != kind:
            continue
        pos = anchor_positions(ann, ids)
        if pos is None or pos[1] <= loc.a or pos[0] >= loc.b:
            continue
        if pos == (loc.a, loc.b):
            spans.append((pos[0], pos[1], "" if ann.status == "hidden" else ann.text))
        else:
            for a, b, reading in pieces_of(ann):
                a, b = pos[0] + a, pos[0] + b
                if a < loc.b and b > loc.a:
                    spans.append((a, b, "" if ann.status == "hidden" else reading))
    if not spans:
        return ""
    cursor, readings = loc.a, []
    for a, b, reading in sorted(spans):
        if a != cursor or b > loc.b:
            return None
        readings.append(reading)
        cursor = b
    return (" " if kind == "pinyin" else "").join(readings) if cursor == loc.b else None


def _current(doc: Document, loc: Location, item: Answer, rules: DisplayRules) -> Optional[str]:
    if item.kind == "reading":
        return _reading(doc, loc)
    return _shown(doc.members[loc.member], rules)[0][loc.displayed_a:loc.displayed_b]


def _equals(doc: Document, loc: Location, item: Answer, current: Optional[str], value: str) -> bool:
    if current is None:
        return False
    if item.kind == "reading" and get_language(doc.members[loc.member].lang).annotation_system == "pinyin":
        from .pinyin_notes import normalize_rule

        try:
            return normalize_rule(loc.text, current)[1] == normalize_rule(loc.text, value)[1]
        except ValueError:
            return current == value
    return current == value


def match(doc: Document, sources: dict, display_rules: Optional[DisplayRules], item, *,
          decisions: Optional[Mapping[str, dict]] = None) -> Suggestion:
    """Locate without mutating. Old readings can disambiguate repeated targets, never IDs alone."""
    item = _answer(item)
    rules = copy.copy(display_rules) if display_rules is not None else DisplayRules()
    result = Suggestion(item, "not_found", display_rules=rules)
    record = (decisions or {}).get(item.fingerprint)
    if record is not None:
        result.decision = record["decision"]
        result.status = "already_done"
    groups = [g for g in doc.groups if g.id == item.group]
    candidates = _locations(doc, groups, rules, item, item.target)
    fallback = False
    if not candidates:
        candidates = _locations(doc, doc.groups, rules, item, item.target)
        fallback = True
    # A text answer imported again after replacement can still be recognised.
    if not candidates and item.kind == "text" and item.new:
        candidates = _locations(doc, groups or doc.groups, rules, item, item.new)
    if result.decision is not None:
        result.candidates = candidates
        result.location = candidates[0] if len(candidates) == 1 else None
        return result
    if not candidates:
        return result
    result.candidates = candidates
    if len(candidates) > 1:
        old = [loc for loc in candidates if
               _equals(doc, loc, item, _current(doc, loc, item, rules), item.old)]
        new = [loc for loc in candidates if
               _equals(doc, loc, item, _current(doc, loc, item, rules), item.new)]
        if all(loc.exact for loc in candidates):
            candidates = old if len(old) == 1 else new if not old and len(new) == 1 else candidates
    if len(candidates) != 1 or not candidates[0].exact:
        result.status = "ambiguous"
        return result
    loc = result.location = candidates[0]
    result.current = _current(doc, loc, item, rules)
    if item.kind not in ("reading", "text"):
        result.status = "found_by_text" if fallback else "ok"
    elif _equals(doc, loc, item, result.current, item.new):
        result.status = "already_done"
    elif _equals(doc, loc, item, result.current, item.old):
        result.status = "found_by_text" if fallback else "ok"
    else:
        result.status = "mismatch"
    return result


def _valid(editor: Editor, suggestion: Suggestion) -> bool:
    loc, item = suggestion.location, suggestion.item
    if loc is None or not loc.exact or suggestion.decision is not None or suggestion.status not in ("ok", "found_by_text"):
        return False
    m = editor.doc.members.get(loc.member)
    if m is None or m.group != loc.group or not _same_language(item.lang, m.lang):
        return False
    vt, ids = visible_map(m)
    if vt.text[loc.a:loc.b] != loc.text or tuple(ids[loc.a:loc.b]) != loc.identities:
        return False
    if item.kind == "reading" and (get_language(m.lang).annotation_system == "none" or
                                    any(ident is None for ident in loc.identities)):
        return False
    return _equals(editor.doc, loc, item, _current(editor.doc, loc, item, suggestion.display_rules), item.old)


class _InTransaction:
    """Reuse Editor operations inside one History transaction, without nested notifications."""
    def __init__(self, tx: Transaction):
        self.tx = tx

    def run(self, label, fn):
        return fn(self.tx)


def _transaction_editor(editor: Editor, tx: Transaction) -> Editor:
    child = copy.copy(editor)
    child.history = _InTransaction(tx)
    return child


def _apply_reading(editor: Editor, suggestion: Suggestion, new: str) -> bool:
    loc = suggestion.location
    m = editor.doc.members[loc.member]
    if get_language(m.lang).annotation_system == "pinyin":
        return editor.apply_pinyin_rule(m.id, loc.a, loc.b, new, scope="here") is not None
    # Preserve other kanji pieces of a manual annotation cut by this acceptance.
    _vt, ids = visible_map(m)
    keep = []
    for ann in editor.doc.annotations.values():
        if ann.member != m.id or ann.kind != "ruby":
            continue
        pos = anchor_positions(ann, ids)
        if pos is not None and pos[0] < loc.b and pos[1] > loc.a:
            for a, b, reading in pieces_of(ann):
                a, b = pos[0] + a, pos[0] + b
                if b <= loc.a or a >= loc.b:
                    keep.append((a, b, reading, ann))
                elif a < loc.a or b > loc.b:
                    raise ValueError("Reading span cuts an existing kanji block")
    aid = editor.add_annotation(m.id, loc.a, loc.b, new)
    if aid is None:
        return False
    for a, b, reading, old in keep:
        kept = editor.add_annotation(m.id, a, b, reading)
        if kept is not None:
            ann = editor.doc.annotations[kept]
            ann.status, ann.origin = old.status, old.origin
            ann.candidates, ann.rule = list(old.candidates), old.rule
            ann.mono, ann.layout, ann.extra = old.mono, copy.deepcopy(old.layout), copy.deepcopy(old.extra)
    return True


def apply(editor: Editor, suggestion: Suggestion, *, new_override: Optional[str] = None) -> bool:
    """Explicit acceptance only; stale, ambiguous and informational suggestions never edit."""
    if suggestion.item.kind not in ("reading", "text") or not _valid(editor, suggestion):
        return False
    loc, item = suggestion.location, suggestion.item
    new = item.new if new_override is None else new_override
    if item.kind == "reading":
        def accept(tx):
            return _apply_reading(_transaction_editor(editor, tx), suggestion, new)
        return editor.history.run("采用读音意见", accept)
    m = editor.doc.members[loc.member]
    rules = suggestion.display_rules
    if rules.converted_for(m):
        # Accepted text is in the display script; retain the member's stored script.
        language = get_language(m.lang).code
        if rules.direction == "traditional" and language == "zh-Hans":
            from opencc import OpenCC

            new = OpenCC("t2s").convert(new)
        elif rules.direction == "simplified" and language == "zh-Hant":
            new = DisplayRules(zh_script="traditional", zh_traditional_variant=rules.zh_traditional_variant,
                               zh_traditional_phrases=rules.zh_traditional_phrases).convert(new)
    # replace_text uses editing offsets: tags excluded, ASS escapes retained.
    plain = editor.visible_text(m.id)
    raw_base = 0
    plain_chars = {}
    plain_base = 0
    for run in m.runs:
        if run.kind != "tag":
            plain_chars.update((raw_base + k, plain_base + k) for k in range(len(run.text)))
            plain_base += len(run.text)
        raw_base += len(run.text)
    vt, _ids = visible_map(m)
    a = plain_chars[vt.chars[loc.a].raw_start]
    last = vt.chars[loc.b - 1]
    b = plain_chars[last.raw_end - 1] + 1
    # Keep line separators inside a multi-line target; callers must choose each line separately.
    if any(tok.kind in ("break", "drawing") and tok.start >= vt.chars[loc.a].raw_start and
           tok.end <= last.raw_end for tok in vt.tokens):
        return False
    return editor.replace_text(m.id, plain[:a] + new + plain[b:], span=(a, b))


def accept_all_readings(editor: Editor, suggestions) -> int:
    """Accept eligible 'ok' readings as ONE undo step; caller owns confirmation and decisions."""
    eligible = [s for s in suggestions if s.item.kind == "reading" and s.status == "ok" and _valid(editor, s)]
    if not eligible:
        return 0

    def accept(tx):
        child = _transaction_editor(editor, tx)
        count = 0
        for suggestion in eligible:
            if _valid(child, suggestion):
                count += _apply_reading(child, suggestion, suggestion.item.new)
        return count
    return editor.history.run("采用所有读音意见", accept)


def read_answers(episode_dir: Path) -> dict[str, dict]:
    path = Path(episode_dir) / "review" / "answers.json"
    if not path.exists():
        return {}
    data = read_json(path)
    if data.get("format") != FORMAT_ANSWERS or data.get("format_version") != 1:
        raise ValueError("Unsupported review answer log")
    out = {}
    for item in data["items"]:
        if item["decision"] not in DECISIONS or not isinstance(item["fingerprint"], str) or not isinstance(item["at"], str):
            raise ValueError("Invalid review decision")
        out[item["fingerprint"]] = dict(item)
    return out


def write_answers(episode_dir: Path, decisions: Mapping[str, dict]) -> None:
    items = []
    for fingerprint, value in decisions.items():
        if value["decision"] not in DECISIONS or not isinstance(value["at"], str):
            raise ValueError("Invalid review decision")
        items.append({"fingerprint": fingerprint, "decision": value["decision"], "at": value["at"]})
    write_json_atomic(Path(episode_dir) / "review" / "answers.json",
                      {"format": FORMAT_ANSWERS, "format_version": 1, "items": items})


def record_decision(episode_dir: Path, item, decision: str, *, at: Optional[str] = None) -> None:
    if decision not in DECISIONS:
        raise ValueError("Invalid review decision")
    fingerprint = _answer(item).fingerprint
    decisions = read_answers(episode_dir)
    decisions[fingerprint] = {"fingerprint": fingerprint, "decision": decision,
                              "at": at if at is not None else datetime.now(timezone.utc).isoformat()}
    write_answers(episode_dir, decisions)
