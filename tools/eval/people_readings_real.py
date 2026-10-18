"""Real character-reading evaluation; inputs read-only, projects only under TEMP.

Default invocation fetches Bangumi 1453 through the normal core API. An optional
--bangumi-json replays the retained response for a reproducible offline comparison.
Prints the JSON summary; detailed occurrences and API evidence stay under TEMP.
No GUI, media decoding, semantic model, or writes to user projects.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from subkit.ass.document import AssDocument
from subkit.core import bangumi
from subkit.core.cc import analyze_event
from subkit.core.readings import (anchor_positions, annotate_document, pieces_of,
                                  visible_map)
from subkit.core.workspace import Project
from subkit.reading.records import romaji_to_kana
from subkit.sources.cn import classify
from subkit.sources.srt import load_srt
from subkit.textutil import kata_to_hira

MEDIA = Path('./sample-media/S02E003 - [LoliHouse] Shoujo Kakumei Utena - 16 [BDRip 1440x1080 HEVC-10bit AACx2].mkv')


def romaji_hiragana(text):
    """Tool-only experiment: preserve explicit long vowels, reject unparsed Latin."""
    text = unicodedata.normalize("NFC", text).lower()
    text = text.translate(str.maketrans({"ā": "aa", "ī": "ii", "ū": "uu", "ē": "ee", "ō": "ou"}))
    if not re.fullmatch(r"[a-z']+(?:\s+[a-z']+)*", text.strip()):
        return ""
    kana = romaji_to_kana(text.strip())
    return kana if re.fullmatch(r"[ぁ-ゖ\s]+", kana) else ""


def reference(sources):
    values, labels, raw_labels = defaultdict(set), {}, []
    for episode, spans in sources:
        for sp in spans:
            if sp.kind != "speaker_label" or not (sp.data or {}).get("readings"):
                continue
            pairs = sp.data["readings"]
            raw_labels.append((episode, sp.value))
            for surface, kana in pairs:
                values[surface].add(kata_to_hira(kana))
    # A contiguous kanji hint can accidentally absorb a surname: 桐生冬芽(とうが).
    # Retain that source evidence but exclude it from gold instead of treating the
    # analyzer's greedy span as an authoritative pronunciation of the full name.
    ambiguous = []
    for surface in list(values):
        for prefix in list(values):
            if prefix != surface and surface.startswith(prefix) and len(prefix) >= 2:
                if all(not r.startswith(p) for r in values[surface] for p in values[prefix]):
                    ambiguous.append({"surface": surface, "readings": sorted(values[surface]),
                                      "known_prefix": prefix, "reason": "原料提示可能仅属于名字，整名参考排除"})
                    del values[surface]
                    break
    # Use all episodes' label hints to complete partially annotated labels.
    known = {s: next(iter(v)) for s, v in values.items() if len(v) == 1}
    for episode, raw_label in raw_labels:
        label = raw_label.removesuffix("たち").strip()
        pos, out, parts = 0, [], []
        while pos < len(label):
            if label[pos].isspace():
                pos += 1
                continue
            candidates = [s for s in known if label.startswith(s, pos)]
            if candidates:
                surface = max(candidates, key=len)
                kana = known[surface]
            else:
                match = re.match(r"[ぁ-ヺー]+", label[pos:])
                if not match:
                    break
                surface = match[0]
                kana = kata_to_hira(surface)
                values[surface].add(kana)
            pos += len(surface)
            parts.append([surface, kana])
            out.append(kana)
        else:
            compact = "".join(label.split())
            values[compact].add("".join(out))
            labels[compact] = {"parts": parts, "boundaries": [sum(len(s) for s, r in parts[:i])
                              for i in range(1, len(parts))], "episode": episode}
    conflicts = {s: sorted(v) for s, v in values.items() if len(v) != 1}
    return {s: next(iter(v)) for s, v in sorted(values.items()) if len(v) == 1}, labels, conflicts, ambiguous


def measure(doc, gold, labels):
    records = []
    for member in doc.members.values():
        if member.lang != "ja" or member.lane != "original" or member.role not in ("dialogue", "top"):
            continue
        vt, ids = visible_map(member)
        pieces = []
        for ann in doc.annotations.values():
            if ann.member != member.id or ann.kind != "ruby" or ann.status == "hidden":
                continue
            pos = anchor_positions(ann, ids)
            if pos:
                pieces.extend((pos[0] + a, pos[0] + b, r, ann.origin) for a, b, r in pieces_of(ann))
        for surface, expected in gold.items():
            for match in re.finditer(re.escape(surface), vt.text):
                a, b = match.span()
                overlapping = sorted(p for p in pieces if p[0] < b and p[1] > a)
                crossing = any(pa < a or pb > b for pa, pb, _, _ in overlapping)
                pos, reading, blocks = a, [], []
                for pa, pb, kana, origin in overlapping:
                    if pa < pos or pb > b:
                        continue
                    reading.extend([kata_to_hira(vt.text[pos:pa]), kata_to_hira(kana)])
                    blocks.append([vt.text[pa:pb], kana])
                    pos = pb
                reading.append(kata_to_hira(vt.text[pos:b]))
                produced = None if crossing else "".join(reading)
                boundaries = {p[1] - a for p in overlapping[:-1]}
                allowed = set(labels.get(surface, {}).get("boundaries", []))
                segmentation = ("crossing_boundary" if crossing else "kana_only" if not overlapping
                                else "one_block" if len(overlapping) == 1 else
                                "family_given_blocks" if boundaries <= allowed else "wrong_split")
                records.append({"surface": surface, "produced": produced, "gold": expected,
                                "correct": produced == expected, "segmentation": segmentation,
                                "origins": sorted(set(p[3] for p in overlapping)),
                                "blocks": blocks, "member": member.id, "offset": a, "text": vt.text})
    total = len(records)
    wrong = [r for r in records if not r["correct"]]
    examples = {}
    for r in wrong:
        key = (r["surface"], r["produced"], r["segmentation"])
        examples.setdefault(key, {k: r[k] for k in ("surface", "produced", "gold", "segmentation", "blocks", "text")})
    return {"occurrences": total, "correct": total - len(wrong), "wrong": len(wrong),
            "segmentation": dict(Counter(r["segmentation"] for r in records)),
            "origin_occurrences": dict(Counter(origin for r in records for origin in r["origins"])),
            "wrong_examples": list(examples.values())}, records


def evaluate(media=MEDIA, recorded=None):
    started = time.monotonic()
    temp_root = Path(os.environ["TEMP"]).resolve()
    scratch = Path(tempfile.mkdtemp(prefix="people-eval-", dir=temp_root))
    assert scratch.is_relative_to(temp_root)
    original, translation = media.with_suffix(".jp.srt"), media.with_suffix(".zh-Hant.ass")
    extra = sorted(media.parent.glob("* - 17 *.jp.srt"))
    inputs = [original, translation, *extra]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    project = Project.create(scratch / "project", "Utena E16 reading evaluation")
    ep = project.add_episode("E16", number=16)
    ep.set_video(media)
    inferred = classify(AssDocument.load(translation), {"dialogue": ["Default"]})
    cfg = {"dialogue": ["Default"], "top": [], "song_jp": [], "song_cn": []}
    for style, role in inferred.style_roles.items():
        if style != "Default" and role in ("top", "song_jp", "song_cn"):
            cfg[role].append(style)
    src = ep.add_source(original, lang="ja", lane="original")
    ep.add_source(translation, lang="zh-Hant", lane="translation", cn_cfg=cfg)
    gold_sources = [("E16", src.spans)]
    for path in extra:
        spans = [sp for i, ev in enumerate(load_srt(path))
                 for sp in analyze_event(i, ev.text, "srt").spans]
        gold_sources.append(("E17", spans))
    gold, labels, conflicts, ambiguous = reference(gold_sources)
    doc = ep.merge()
    before, before_records = measure(doc, gold, labels)
    if recorded:
        response = json.loads(Path(recorded).read_text(encoding="utf-8"))
    else:
        subject, characters, details = bangumi.fetch_subject("1453")
        response = {"subject": subject, "characters": characters, "details": details}
    (scratch / "bangumi.json").write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
    book = project.character_book()
    rows = bangumi.build_candidates(response["subject"], response["characters"], response["details"],
                                    book, project.languages, traditional=True)
    selected = [i for i, r in enumerate(rows) if r["checked"]]
    bangumi.apply_import(book, rows, selected)
    annotate_document(doc, ep.sources, ep.reading_engine(doc))
    after, after_records = measure(doc, gold, labels)
    imported = []
    for row in rows:
        if not row["checked"]:
            continue
        info = bangumi._info(response["details"].get(str(row["bangumi"]), response["details"].get(row["bangumi"], {})))
        form = row["names"]["ja"]
        imported.append({"ja": row["ja_name"], "zh": row["zh_name"], "romaji": row["romaji"],
                         "kana": info.get("假名") or info.get("纯假名", ""), "relation": row["relation"],
                         "parts": [form.family, form.given],
                         "readings": [form.family_reading, form.given_reading]})
    experiment_rows = copy.deepcopy(rows)
    filled = []
    for row in experiment_rows:
        form = row["names"]["ja"]
        kana = romaji_hiragana(row["romaji"])
        if kana and row["checked"]:
            parts = kana.split()
            if form.family and len(parts) == 2:
                for attr, value in zip(("family_reading", "given_reading"), parts):
                    if not getattr(form, attr):
                        setattr(form, attr, value)
                        filled.append([row["ja_name"], attr, value])
            elif not form.family and not form.given_reading:
                form.given_reading = "".join(parts)
                filled.append([row["ja_name"], "given_reading", form.given_reading])
    from subkit.core.people import CharacterBook
    experiment = CharacterBook(book.path)
    bangumi.apply_import(experiment, experiment_rows, selected)
    annotate_document(doc, ep.sources, ep.reading_engine(doc))
    converted, converted_records = measure(doc, gold, labels)
    evidence = {"before": before_records, "after": after_records, "romaji_fill": converted_records}
    (scratch / "occurrences.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    per_surface = []
    for surface, kana in gold.items():
        variants = [[r for r in rs if r["surface"] == surface]
                    for rs in (before_records, after_records, converted_records)]
        per_surface.append({"surface": surface, "gold": kana, "occurrences": len(variants[0]),
                            "correct": [sum(r["correct"] for r in rs) for rs in variants]})
    unchanged = hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    ep.save()
    return {"subject": response["subject"], "reference": {"files": [str(original), *map(str, extra)],
            "conflicts": conflicts, "ambiguous_hints_excluded": ambiguous, "surfaces": per_surface},
            "counting": "对白/顶部对白中逐字匹配全名及姓名部分，重叠分别计数；含纯假名，不含标签/歌词。跨边界注音无法分配，计错。correct 数组依次为空人物表/默认导入/工具内罗马字补空。",
            "style_roles": inferred.style_roles, "merge_semantic": False, "before": before,
            "after": after, "romaji_fill": converted, "romaji_fields_filled": filled,
            "bangumi_characters": imported, "validation_ok": ep.validate().ok,
            "proposals": ["本集未见罗马字补读音的准确率收益，不修改产品代码。",
                          "在实际出现缺读音角色的集数继续评测，再决定是否加入导入预览；ō 可能对应 ou 或 oo，无法仅凭罗马字确定。",
                          "人工复核未拆姓名（姫宮アンシー、高槻枝織、千唾馬宮）及原料/Bangumi 拼写差异（珠璃/樹璃、全角Ｂ/Ｃ与E/F）；不猜测别名。"],
            "source_bytes_unchanged": unchanged, "scratch": str(scratch),
            "seconds": round(time.monotonic() - started, 3)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media", type=Path, default=MEDIA)
    parser.add_argument("--bangumi-json", type=Path)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(evaluate(args.media, args.bangumi_json), ensure_ascii=False, separators=(",", ":")))
