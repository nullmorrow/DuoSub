"""Read-only discovery and conservative alignment of native Chinese references.

The imported Source is the clock and identity boundary.  No edited Member times,
OpenCC output, directory neighbours, or filename order become reference text.
Unproven pairs stay absent so the exporter can retain its normal fallback.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path
import hashlib
import re
import unicodedata

from ..ass.document import AssDocument
from ..ass.text import plain_text
from ..encoding import decode_bytes
from ..model.source import Source
from .srt import parse_srt


@dataclass
class ReferenceEvent:
    i: int
    start: int
    end: int
    text: str
    kind: str = "Dialogue"


@dataclass
class ReferenceFile:
    path: str
    sha256: str
    encoding: str
    script_hint: str
    events: list[ReferenceEvent]


@dataclass
class ReferenceSelection:
    target_script: str
    target: ReferenceFile | None = None
    baseline: ReferenceFile | None = None
    target_matches: dict[int, int] = field(default_factory=dict)
    baseline_matches: dict[int, int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    protected_input_paths: list[str] = field(default_factory=list)


_SCRIPT_TAG = re.compile(r"(?i)(?<![a-z])(?:zh[-_]hans|chs|sc|zh[-_]hant|cht|tc)(?![a-z])")
_EPISODE = re.compile(r"(?i)(?<![a-z0-9])s(\d{1,2})e(\d{1,3})(?!\d)")
_SRT_TAGS = re.compile(r"</?(?:b|i|u|s|font|span)\b[^>]*>", re.I)
_BR_TAG = re.compile(r"<br\s*/?>", re.I)
_EXTENSIONS = frozenset((".ass", ".ssa", ".srt"))


def _cancel(runner) -> None:
    if runner is not None:
        runner.check_cancelled()


def _script(value: str) -> str:
    key = (value or "").strip().lower().replace("_", "-")
    if key in ("zh-hans", "hans", "simplified", "chs", "sc", "zh-cn", "zh-sg"):
        return "zh-Hans"
    if key in ("zh-hant", "hant", "traditional", "cht", "tc", "zh-tw", "zh-hk"):
        return "zh-Hant"
    return ""


def _filename_script(path: Path) -> str:
    hints = {_script(m.group()) for m in _SCRIPT_TAG.finditer(path.stem)}
    return next(iter(hints)) if len(hints) == 1 else ""


def _episode(name: str) -> tuple[int | None, int] | None:
    full = _EPISODE.search(name)
    if full:
        return int(full[1]), int(full[2])
    for pattern in (r"(?i)^e(?:p)?(\d{1,3})$", r"\[(\d{1,3})\]",
                    r"(?:^|\s)-\s*(\d{1,3})(?=\s|\[|\.|$)"):
        match = re.search(pattern, name)
        if match:
            return None, int(match[1])
    return None


def _same_episode(a, b) -> bool:
    return a[1] == b[1] and (a[0] is None or b[0] is None or a[0] == b[0])


def _pair_stem(path: Path) -> str:
    # S01E001 and S01E01 are the same episode, while release/version suffixes
    # remain significant: they must not silently select between translations.
    name = _EPISODE.sub(lambda m: f"s{int(m[1])}e{int(m[2])}", path.stem)
    return re.sub(r"[\s._-]+", " ", _SCRIPT_TAG.sub("", name)).strip().casefold()


def _visible(text: str, fmt: str, *, wrap_style: int = 0) -> str:
    if fmt.lower() == "srt":
        text = unescape(_SRT_TAGS.sub("", _BR_TAG.sub("\n", text)))
    return plain_text(text, breaks="\n", wrap_style=wrap_style)


class _Normalizer:
    def __init__(self, issues: list[str]):
        self._keys: dict[str, str] = {}
        try:
            from opencc import OpenCC
            self.to_s = OpenCC("t2s").convert
            self.to_t = OpenCC("s2t").convert
        except ImportError:
            self.to_s = self.to_t = lambda s: s
            issues.append("OpenCC 不可用；参考配对仅使用完全相同的文字。")

    def key(self, text: str) -> str:
        if text not in self._keys:
            if len(self._keys) >= 32768:
                self._keys.clear()
            self._keys[text] = " ".join(self.to_s(unicodedata.normalize("NFC", text)).split())
        return self._keys[text]

    def infer_script(self, events: list[ReferenceEvent]) -> str:
        chars = Counter(ch for event in events if event.kind == "Dialogue" for ch in event.text)
        # Script-neutral Han characters do not prove Chinese rather than Japanese.
        kana = sum(n for ch, n in chars.items() if "\u3040" <= ch <= "\u30ff" or "\uff66" <= ch <= "\uff9f")
        hans = hant = 0
        for ch, n in chars.items():
            if "\u3400" <= ch <= "\u9fff":
                hans += n * (self.to_t(ch) != ch and self.to_s(ch) == ch)
                hant += n * (self.to_s(ch) != ch and self.to_t(ch) == ch)
        if kana > max(4, hans + hant):
            return "ja"
        if hans and not hant or hans >= max(6, hant * 4):
            return "zh-Hans"
        if hant and not hans or hant >= max(6, hans * 4):
            return "zh-Hant"
        return ""


@dataclass
class _Alignment:
    matches: dict[int, int]
    credible: bool
    anchors: int
    total: int


def _monotonic(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Keep only pairs not involved in an inversion, without choosing an LIS tie."""
    ordered = sorted(pairs)
    suffix = [10**18] * (len(ordered) + 1)
    for n in range(len(ordered) - 1, -1, -1):
        suffix[n] = min(suffix[n + 1], ordered[n][1])
    high = -1
    result = []
    for n, pair in enumerate(ordered):
        if pair[1] > high and pair[1] < suffix[n + 1]:
            result.append(pair)
        high = max(high, pair[1])
    return result


def _overlapping(events: list[ReferenceEvent], lo: int, hi: int) -> set[int]:
    ordered = sorted((events[n].start, events[n].end, n) for n in range(lo, hi)
                     if events[n].end > events[n].start)
    overlap = set()
    previous_end = -1
    for k, (start, end, n) in enumerate(ordered):
        if previous_end > start or k + 1 < len(ordered) and ordered[k + 1][0] < end:
            overlap.add(n)
        previous_end = max(previous_end, end)
    return overlap


def _align(left: list[ReferenceEvent], right: list[ReferenceEvent], normal: _Normalizer, runner) -> _Alignment:
    _cancel(runner)
    a = [e for e in left if e.kind == "Dialogue" and normal.key(e.text)]
    b = [e for e in right if e.kind == "Dialogue" and normal.key(e.text)]
    ak, bk = [normal.key(e.text) for e in a], [normal.key(e.text) for e in b]
    ai, bi = defaultdict(list), defaultdict(list)
    for n, key in enumerate(ak):
        ai[key].append(n)
    for n, key in enumerate(bk):
        bi[key].append(n)
    anchors = _monotonic([(pos[0], bi[key][0]) for key, pos in ai.items()
                          if len(pos) == 1 and len(bi.get(key, ())) == 1])
    count = len(anchors)
    # A few common song/stock lines cannot validate an otherwise unrelated file.
    credible = (count >= 3 and count >= min(len(a), len(b)) * .25
                and count >= max(len(a), len(b)) * .15
                and anchors[-1][0] - anchors[0][0] >= (len(a) - 1) * .5
                and anchors[-1][1] - anchors[0][1] >= (len(b) - 1) * .5)
    tiny_exact = bool(a) and len(a) == len(b) <= 2 and len(set(ak)) == len(a) and ak == bk and all(
        (x.start, x.end) == (y.start, y.end) for x, y in zip(a, b))
    credible = bool(credible or tiny_exact)
    if not credible:
        return _Alignment({}, False, count, len(a))

    boundaries = [(-1, -1), *anchors, (len(a), len(b))]
    pairs = list(anchors)
    stamps_a = Counter((e.start, e.end) for e in a)
    stamps_b = Counter((e.start, e.end) for e in b)
    # Duplicates require unique exact original times inside the same anchored
    # gap.  Edited/exported times are never consulted.  Changed wording requires
    # the stronger whole-file anchor test, not the tiny-file exception.
    for (la, lb), (ra, rb) in zip(boundaries, boundaries[1:]):
        _cancel(runner)
        time_a, time_b = defaultdict(list), defaultdict(list)
        for n in range(la + 1, ra):
            time_a[(a[n].start, a[n].end)].append(n)
        for n in range(lb + 1, rb):
            time_b[(b[n].start, b[n].end)].append(n)
        overlaps_a = _overlapping(a, la + 1, ra)
        overlaps_b = _overlapping(b, lb + 1, rb)
        gap_pairs = []
        for step, (stamp, positions) in enumerate(time_a.items()):
            if step % 256 == 0:
                _cancel(runner)
            others = time_b.get(stamp, ())
            if len(positions) != 1 or len(others) != 1 or stamp[1] <= stamp[0]:
                continue
            x, y = positions[0], others[0]
            if ak[x] != bk[y]:
                if count < 3 or stamps_a[stamp] != 1 or stamps_b[stamp] != 1:
                    continue
                # A split/merge or parallel sign must not become a guessed
                # one-to-one pair simply because one interval stayed intact.
                if x in overlaps_a or y in overlaps_b:
                    continue
            gap_pairs.append((x, y))
        pairs.extend(_monotonic(gap_pairs))
    if tiny_exact:
        pairs = list(enumerate(range(len(a))))
    matches = {a[x].i: b[y].i for x, y in pairs}
    # Comments are preserved in the file but never validate an episode.  Their
    # own identity needs exact visible text and unique exact original times.
    comments_a, comments_b = defaultdict(list), defaultdict(list)
    for events, table in ((left, comments_a), (right, comments_b)):
        for event in events:
            if event.kind != "Dialogue" and event.text:
                table[(event.kind, event.start, event.end, normal.key(event.text))].append(event.i)
    for key, positions in comments_a.items():
        if len(positions) == 1 and len(comments_b.get(key, ())) == 1:
            matches[positions[0]] = comments_b[key][0]
    return _Alignment(matches, True, count, len(a))


def resolve_reference(source: Source, target_script: str, reference_path: Path | None = None,
                      *, episode_id: str = "", runner=None) -> ReferenceSelection:
    """Find one proven native reference, without changing any input bytes.

    Blank searches only the original subtitle directory.  A directory argument
    searches only that directory; a file argument explicitly selects the target.
    A same-stem, oppositely labelled native companion may supply the unedited
    baseline.  Otherwise baseline remains None and the caller uses Source text.
    """
    target_script = _script(target_script)
    result = ReferenceSelection(target_script)
    seen_paths = set()

    def protect(path: Path) -> Path:
        absolute = path.absolute()
        try:
            resolved = absolute.resolve()
        except (OSError, RuntimeError):
            resolved = absolute
        for item in (absolute, resolved):
            key = str(item).casefold()
            if key not in seen_paths:
                seen_paths.add(key)
                result.protected_input_paths.append(str(item))
        return absolute

    source_path = protect(Path(source.path)) if source.path else None
    requested = protect(Path(reference_path)) if reference_path is not None else None
    if not target_script:
        result.issues.append("不支持的参考目标字形；仅支持 zh-Hans / zh-Hant。")
        return result
    _cancel(runner)
    normal = _Normalizer(result.issues)
    original = [ReferenceEvent(e.i, e.start, e.end, _visible(e.text, source.format), e.kind)
                for e in source.events]
    source_script = _script(source.lang) or normal.infer_script(original)
    expected_episode = _episode(episode_id) or (_episode(source_path.name) if source_path else None)
    explicit_file = requested is not None and requested.is_file()
    if requested is not None and not explicit_file and not requested.is_dir():
        result.issues.append(f"参考路径不存在或不可读：{requested}")
        return result
    directory = requested if requested is not None and not explicit_file else (
        requested.parent if explicit_file else source_path.parent if source_path else None)
    directories = [directory] if directory is not None else []
    # An explicit file/folder may contain its native companion.  Source's own
    # directory is also a permitted baseline location, never its neighbours.
    if requested is not None and source_path is not None and source_path.parent not in directories:
        directories.append(source_path.parent)
    paths = []
    for folder in directories:
        _cancel(runner)
        protect(folder)
        try:
            children = sorted(folder.iterdir(), key=lambda p: p.name.casefold())
        except OSError as exc:
            result.issues.append(f"无法读取参考目录：{folder}（{exc}）")
            continue
        for path in children:
            if path.suffix.lower() in _EXTENSIONS and path.is_file():
                path = protect(path)
                if path not in paths:
                    paths.append(path)
    if explicit_file and requested not in paths:
        paths.insert(0, requested)
    parsed: dict[Path, ReferenceFile | None] = {}
    generated: set[Path] = set()

    def read(path: Path) -> ReferenceFile | None:
        if path in parsed:
            return parsed[path]
        _cancel(runner)
        protect(path)
        try:
            raw = path.read_bytes()
            decoded = decode_bytes(raw)
            if "\x00" in decoded.text or "\ufffd" in decoded.text:
                raise ValueError("包含无效编码或二进制字符")
            if path.suffix.lower() == ".srt":
                events = [ReferenceEvent(b.order, b.start, b.end, _visible(b.text, "srt"))
                          for b in parse_srt(decoded.text)]
            elif path.suffix.lower() in (".ass", ".ssa"):
                doc = AssDocument.parse(decoded.text)
                wrap = int(doc.info("WrapStyle", "0") or "0")
                events = [ReferenceEvent(i, e.start * 10, e.end * 10,
                                         _visible(e.text, "ass", wrap_style=wrap), e.kind)
                          for i, e in enumerate(doc.events)]
                header = "\n".join(v for k, v in doc.script_info if k is None)
                if "DuoSub" in header and "由工程导出" in header:
                    generated.add(path)
            else:
                raise ValueError("仅支持 ASS / SSA / SRT")
            if not any(e.kind == "Dialogue" and e.text.strip() for e in events):
                raise ValueError("没有可见对白")
            named_hint, content_hint = _filename_script(path), normal.infer_script(events)
            hint = named_hint or content_hint
            if named_hint and content_hint in ("zh-Hans", "zh-Hant") and named_hint != content_hint:
                hint = ""
                result.issues.append(f"参考文件字形标签与正文不一致，未采用：{path}")
            file = ReferenceFile(str(path), hashlib.sha256(raw).hexdigest(), decoded.encoding, hint, events)
            if source_path is not None and path == source_path and source.sha256 and file.sha256 != source.sha256:
                result.issues.append(f"原字幕路径字节已变化；仍以工程导入时的原事件配对：{path}")
        except (OSError, UnicodeError, ValueError) as exc:
            result.issues.append(f"参考文件无法读取：{path}（{exc}）")
            file = None
        parsed[path] = file
        _cancel(runner)
        return file

    def eligible(path: Path, *, direct: bool = False) -> bool:
        candidate_episode = _episode(path.name)
        if expected_episode and candidate_episode and not _same_episode(expected_episode, candidate_episode):
            if direct:
                result.issues.append(f"参考文件集号与原字幕不符：{path}")
            return False
        if not direct and not candidate_episode:
            if source_path is None or _pair_stem(source_path) != _pair_stem(path):
                return False
        if re.search(r"(?i)(?:[._-]ja(?:[-_]jp)?|[._-]jp)\.", path.name):
            return False
        return True

    candidates = []
    target_paths = [requested] if explicit_file else [p for p in paths if p.parent == directory]
    for path in target_paths:
        _cancel(runner)
        if not eligible(path, direct=explicit_file):
            continue
        named_script = _filename_script(path)
        if named_script and named_script != target_script:
            if explicit_file:
                result.issues.append(f"参考文件字形标签与目标不符：{path}")
            continue
        file = read(path)
        if file is None:
            continue
        if path in generated:
            result.issues.append(f"跳过工程生成的字幕，不能作为原版参考：{path}")
            continue
        if file.script_hint != target_script:
            if explicit_file or not named_script:
                result.issues.append(f"无法确认参考文件为目标字形 {target_script}：{path}")
            continue
        alignment = _align(original, file.events, normal, runner)
        if not alignment.credible:
            result.issues.append(f"参考文件与原始正文/顺序证据不足，未采用：{path}（唯一文字锚点 {alignment.anchors}）")
            continue
        candidates.append((file, alignment))
    # Multiple byte/visible-equivalent copies are safe; differing translations
    # require an explicit path instead of a filename/mtime/coverage guess.
    distinct = {}
    for file, alignment in candidates:
        identity = tuple((e.kind, e.start, e.end, e.text) for e in file.events)
        distinct.setdefault(identity, (file, alignment))
    if len(distinct) != 1:
        if distinct:
            result.issues.append("同集存在多个不同的目标字形参考，无法唯一选择；请指定参考文件：" +
                                 "；".join(file.path for file, _ in distinct.values()))
        else:
            result.issues.append(f"未找到可可靠采用的 {target_script} 原版参考；可指定同集参考文件或所在文件夹（只查所选目录）。")
        return result
    result.target, alignment = next(iter(distinct.values()))
    result.target_matches = dict(alignment.matches)

    # A baseline needs native opposite-script filename pairing with the chosen
    # target, plus its own source alignment.  Arbitrary older/newer editions are
    # never declared an unedited original merely because the times look alike.
    baselines = []
    target_path = Path(result.target.path)
    for path in paths:
        _cancel(runner)
        if path == target_path or path == source_path or not eligible(path):
            continue
        if not source_script or source_script == target_script or _filename_script(path) != source_script:
            continue
        if not _filename_script(target_path) or _pair_stem(path) != _pair_stem(target_path):
            continue
        file = read(path)
        if file is None or path in generated:
            continue
        source_to_base = _align(original, file.events, normal, runner)
        base_to_target = _align(file.events, result.target.events, normal, runner)
        if source_to_base.credible and base_to_target.credible:
            baselines.append((file, source_to_base, base_to_target))
    distinct_bases = {}
    for file, a, b in baselines:
        identity = tuple((e.kind, e.start, e.end, e.text) for e in file.events)
        distinct_bases.setdefault(identity, (file, a, b))
    if len(distinct_bases) == 1:
        result.baseline, to_base, to_target = next(iter(distinct_bases.values()))
        result.baseline_matches = dict(to_base.matches)
        composed = {i: to_target.matches[j] for i, j in to_base.matches.items() if j in to_target.matches}
        for i, j in composed.items():
            if i in result.target_matches and result.target_matches[i] != j:
                del result.target_matches[i]
                result.baseline_matches.pop(i, None)
                result.issues.append(f"原事件 {i} 的三方配对互相冲突，保留原导出回退。")
            else:
                result.target_matches[i] = j
    elif len(distinct_bases) > 1:
        result.issues.append("存在多个不同的原版同字形底稿，未猜测其版本；使用导入原文作底稿。")
    if result.baseline is None:
        result.issues.append("未找到可证明为原配版本的同字形底稿；使用工程导入原文作三方底稿。")
    matched = sum(e.kind == "Dialogue" and e.i in result.target_matches for e in original)
    total = sum(e.kind == "Dialogue" and bool(normal.key(e.text)) for e in original)
    if matched < total:
        result.issues.append(f"目标参考可靠匹配 {matched}/{total} 条原始正文；其余重复、拆并句或证据不足的事件沿用正常导出。")
    return result
