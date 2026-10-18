"""Evaluate shipped C-key hints from real rough subtitles, without gold candidates.

Run from the project root: python tools/eval/hint_rough_real.py
Only decoded features are cached, under TEMP. Media and subtitles are read-only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np
from opencc import OpenCC
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from subkit import autotime, proc
from subkit.ass import AssDocument
from subkit.ass.text import plain_text
from subkit.audio import HOP_MS, load_features, waveform_cache_file
from subkit.sources.cn import classify
from subkit.sources.jp import JpOptions, clean_line
from subkit.sources.srt import load_srt
from tools.eval.build_utena import UTENA, norm

DEFAULT_PADDING = (100, 160)  # waveform.HABIT_INITIAL_MS; no Qt/settings needed.
_SIMPLIFY = OpenCC("t2s")


def pair_rows(source: list[dict], gold: list[dict], *, language: str) -> list[int]:
    """Extend build_utena's monotonic text matcher; times here are milliseconds."""
    def normalized(row):
        text = row["text"]
        if language == "ja":
            text = " ".join(clean_line(t, JpOptions()).text for t in text.splitlines())
        else:
            text = _SIMPLIFY.convert(text)
        return norm(text)

    a, b = [normalized(r) for r in source], [normalized(r) for r in gold]
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), np.float64)
    back = np.zeros((n + 1, m + 1), np.uint8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best, how = dp[i - 1, j], 1
            if dp[i, j - 1] > best:
                best, how = dp[i, j - 1], 2
            similarity = 0.0
            if a[i - 1] and b[j - 1] and abs(source[i - 1]["start"] - gold[j - 1]["start"]) <= 15000:
                similarity = 1 - Levenshtein.normalized_distance(a[i - 1], b[j - 1])
            if similarity >= .6 and dp[i - 1, j - 1] + similarity > best:
                best, how = dp[i - 1, j - 1] + similarity, 3
            dp[i, j], back[i, j] = best, how
    result = [-1] * n
    i, j = n, m
    while i and j:
        how = back[i, j]
        if how == 3:
            result[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif how == 1:
            i -= 1
        else:
            j -= 1
    return result


def inputs(ep: int) -> tuple[Path, Path, Path, Path]:
    # Prefer Recheck, like build_utena.final_path; also support the S02 rename.
    for folder in (UTENA / "Done/Publish Candidate/Recheck", UTENA / "Done/Publish Candidate"):
        candidates = [p for p in folder.glob("*.zh-Hant.ass") if f" - {ep:02d} [" in p.name]
        if len(candidates) > 1:
            raise ValueError(f"ambiguous gold E{ep:02d}: {candidates}")
        if candidates:
            gold = candidates[0]
            stem = gold.name.removesuffix(".zh-Hant.ass")
            return (UTENA / "origin" / (stem + ".mkv"),
                    UTENA / "origin" / (stem + "-sub01.ass.bak"),
                    ROOT / "tools/eval/data/utena/jp" / f"E{ep:02d}.ja.srt", gold)
    raise FileNotFoundError(f"gold E{ep:02d}")


def event_row(index, event):
    return dict(id=f"event:{index}", start=event.start * 10, end=event.end * 10,
                text=plain_text(event.text, breaks=" "))


def read_rows(cn_path, jp_path, gold_path):
    cn_doc = AssDocument.load(cn_path)
    roles = classify(cn_doc, {"dialogue": ["Default"], "song_cn": ["OPCH", "ED"]}).roles
    cn = [event_row(i, e) for i, e in enumerate(cn_doc.events) if roles.get(i) in ("dialogue", "top")]
    # Keep all cues, including non-speech, empty and unmatched cues, in context/counts.
    jp = [dict(id=f"cue:{i}:index:{c.index}", start=c.start * 10, end=c.end * 10, text=c.text)
          for i, c in enumerate(load_srt(jp_path))]
    final = AssDocument.load(gold_path)
    gold = {}
    for name, styles in (("A", {"CN", "CNSecondary", "CNTop"}),
                         ("B", {"JP", "JPSecondary", "JPTop"})):
        gold[name] = [event_row(i, e) for i, e in enumerate(final.events) if e.is_dialogue and e.style in styles]
    return {"A": cn, "B": jp}, gold


def generate(activity, rough: list[dict], padding: tuple[int, int]) -> list[dict]:
    """Candidate generation deliberately has no gold/mapping argument."""
    times = [(r["start"], r["end"]) for r in rough]
    result = []
    for i, current in enumerate(times):
        previous = times[i - 1] if i else None
        following = times[i + 1] if i + 1 < len(times) else None
        options = dict(lead_in_ms=padding[0], lead_out_ms=padding[1], previous=previous, following=following)
        suggestion = autotime.suggest_group_times(activity, *current, **options)
        adopted = tuple(c if s is None else s for c, s in zip(current, suggestion)) if suggestion else current
        again = autotime.suggest_group_times(activity, *adopted, **options) if suggestion else None
        stable = again is None or all(s is None or s == c for s, c in zip(again, adopted))
        result.append(dict(suggested_ms=suggestion, adopted_ms=adopted,
                           recomputed_ms=again, idempotent=stable))
    return result


def score_rows(rows: list[dict]) -> dict:
    """Count all source rows; unmatched edges are unknown, abstentions stay put."""
    result = dict(source_rows=len(rows), matched_rows=sum(r["gold_ms"] is not None for r in rows),
                  suggestions=sum(r["suggested_ms"] is not None for r in rows),
                  idempotence_checked=sum(r["suggested_ms"] is not None for r in rows),
                  idempotence_failed=sum(not r["idempotent"] for r in rows))
    for edge, index in (("start", 0), ("end", 1)):
        scores = dict(denominator=len(rows), matched=0, unknown=0, covered=0, abstained=0,
                      matched_covered=0, made_worse=0, made_better=0,
                      gold_checked=0, gold_suggested=0, gold_moved_beyond_200ms=0,
                      gold_kept=0, baseline_severe_gt_1000ms=0,
                      suggestion_severe_gt_1000ms=0, adopted_severe_gt_1000ms=0)
        for limit in (100, 200):
            for kind in ("baseline", "suggestion", "adopted"):
                scores[f"{kind}_within_{limit}ms"] = 0
        for row in rows:
            suggestion = row["suggested_ms"]
            value = suggestion[index] if suggestion else None
            scores["covered" if value is not None else "abstained"] += 1
            if row["gold_ms"] is None:
                scores["unknown"] += 1
                continue
            scores["matched"] += 1
            gold = row["gold_ms"][index]
            baseline = abs(row["rough_ms"][index] - gold)
            adopted = abs(row["adopted_ms"][index] - gold)
            error = abs(value - gold) if value is not None else None
            scores["matched_covered"] += error is not None
            scores["made_worse"] += baseline <= 200 and adopted > 200
            scores["made_better"] += baseline > 200 and adopted <= 200
            for kind, distance in (("baseline", baseline), ("suggestion", error), ("adopted", adopted)):
                if distance is not None:
                    scores[f"{kind}_severe_gt_1000ms"] += distance > 1000
                    for limit in (100, 200):
                        scores[f"{kind}_within_{limit}ms"] += distance <= limit
            gold_hint = row["gold_suggested_ms"]
            gold_value = gold_hint[index] if gold_hint else None
            scores["gold_checked"] += 1
            scores["gold_suggested"] += gold_value is not None
            spoiled = gold_value is not None and abs(gold_value - gold) > 200
            scores["gold_moved_beyond_200ms"] += spoiled
            scores["gold_kept"] += not spoiled
        result[edge] = scores
    return result


def evaluate(activity, rough, gold, mapping, padding):
    candidates = generate(activity, rough, padding)
    rows = []
    for i, (source, candidate, j) in enumerate(zip(rough, candidates, mapping)):
        answer = gold[j] if j >= 0 else None
        gold_hint = None
        if answer is not None:
            # Separate gold-kept experiment: only the current line uses gold time.
            # Neighbours stay rough; never put test answers into candidate context.
            previous = (rough[i - 1]["start"], rough[i - 1]["end"]) if i else None
            following = (rough[i + 1]["start"], rough[i + 1]["end"]) if i + 1 < len(rough) else None
            gold_hint = autotime.suggest_group_times(activity, answer["start"], answer["end"],
                        lead_in_ms=padding[0], lead_out_ms=padding[1], previous=previous, following=following)
        rows.append(dict(source_id=source["id"], gold_id=answer["id"] if answer else None,
                         rough_ms=[source["start"], source["end"]],
                         gold_ms=[answer["start"], answer["end"]] if answer else None,
                         gold_suggested_ms=gold_hint, **candidate))
    return dict(metrics=score_rows(rows), rows=rows)


def leave_one_out(samples: dict, excluded: int):
    training = sorted(ep for ep in samples if ep != excluded)
    onsets = [v for ep in training for v in samples[ep][0]]
    offsets = [v for ep in training for v in samples[ep][1]]
    calibration = autotime._hint_calibration(onsets, offsets, None, DEFAULT_PADDING)
    return dict(**asdict(calibration), training_episodes=training, excluded_episode=excluded)


def total_metrics(episodes, source, mode):
    return score_rows([r for ep in episodes for r in ep["sources"][source]["modes"][mode]["rows"]])


def write_report(report, output):
    report["method"] = dict(default_padding_ms=DEFAULT_PADDING, pairing_similarity_min=.6,
        pairing_max_start_distance_ms=15000, pairing="monotonic one-to-one text DP",
        rough_a_scope="classified dialogue/top events; excludes songs, signs and comments",
        rough_b_scope="all SRT cues, including non-speech/empty/unmatched",
        candidate_context="adjacent rough rows in source order; fixed during adoption",
        calibration="pool other completed episodes' unique gold JP dialogue spans",
        abstention="not a suggestion hit; unchanged rough edge in adopted score",
        unknown="unmatched source remains in full denominator, not labelled incorrect",
        gold_kept_context="gold current line, rough adjacent rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    lines = ["# 真实粗轴建议评测 · 2026-10-12", "",
             "仅评测现有 `suggest_group_times`，产品代码、版本与默认参数均未修改。", "",
             "## 输入与方法", "",
             "少女革命 E06–E15：A 为 origin 的字幕组中文 `.ass.bak`；B 为仓库日文 BD 重定时 SRT；答案为 Done/Publish Candidate（优先 Recheck）的用户成品。E14/15 使用 S02 文件名。原始视频路径直接交给 ffmpeg，第一音轨单声道 16 kHz；只在本次 TEMP 缓存音频特征。",
             "A 按 build_utena 的角色筛选只评对白/顶部对白，歌曲、标牌与注释不在其分母；B 保留全部 SRT cue。文字配对扩展 build_utena 的保序一对一 DP：简繁统一、日文按现有导入器清理说话人/读音、去标点，Levenshtein 相似度 ≥0.6，开始相差 ≤15 秒。来源 ID、答案起止、未配对行全部保留于 JSON；拆并句、多义重复句或偏差超过 15 秒可能无法配对，未做人审核。ASS 厘秒乘10，SRT按现有解析器四舍五入至10 ms；音频帧10 ms。",
             "候选只使用音频、粗轴当前行与粗轴文件自身相邻行（来源顺序，包括未配对行）。默认留白 100/160 ms；跨集校准以其他已完成集的成品日文对白提取边缘留白样本，按现有中位数/截断规则（至少各20样本，提前0–600、延后−200–600 ms）计算。留出集答案不参与校准；JSON列明训练集。",
             "每端的全分母是所有粗轴行；未配对行是未知而非判错。建议命中计数不把拒绝当成功；采用后命中将拒绝端保持粗轴。改坏=粗轴≤200 ms、采用后>200 ms；改好反之；严重偏差>1000 ms。幂等只检查有建议行，采用后在固定音频/参数/粗轴邻句下重算相等或拒绝。正确轴保持另从成品当前行出发，邻句仍用粗轴；未移动或拒绝均计保持。", "",
             "## 输入覆盖与耗时", "",
             "| 集 | A 配对/总行 | A 未配对答案 | B 配对/总行 | B 未配对答案 | 解码/特征秒 | 评测秒 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for ep in report["episodes"]:
        a, b = ep["sources"]["A"], ep["sources"]["B"]
        lines.append(f"| E{ep['episode']:02d} | {a['matched']}/{a['source_rows']} | {a['gold_unmatched']} | {b['matched']}/{b['source_rows']} | {b['gold_unmatched']} | {ep['decode_seconds']:.2f} | {ep['evaluation_seconds']:.2f} |")
    lines += ["", "## 各端结果（计数/全分母，均包含拒绝行）", "",
              "| 集/粗轴/留白 | 端 | 建议覆盖 | 粗轴≤100/200 | 建议≤100/200 | 采用后≤100/200 | 改坏/改好 | 严重：粗轴/建议/采用 | 正确轴移坏/已配对 | 幂等失败/有建议 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    scopes = [("合计", None)]
    for label, ep in scopes:
        for source in ("A", "B"):
            for mode in ("default", "loo"):
                metrics = ep["sources"][source]["modes"][mode]["metrics"] if ep else report["totals"][source][mode]
                for edge in ("start", "end"):
                    s = metrics[edge]
                    def hits(kind):
                        return f"{s[kind + '_within_100ms']}/{s[kind + '_within_200ms']} ({s['denominator']})"
                    lines.append(f"| {label}/{source}/{mode} | {edge} | {s['covered']}/{s['denominator']} | {hits('baseline')} | {hits('suggestion')} | {hits('adopted')} | {s['made_worse']}/{s['made_better']} | {s['baseline_severe_gt_1000ms']}/{s['suggestion_severe_gt_1000ms']}/{s['adopted_severe_gt_1000ms']} | {s['gold_moved_beyond_200ms']}/{s['gold_checked']} | {metrics['idempotence_failed']}/{metrics['idempotence_checked']} |")
    lines += ["", "各集相同指标、拒绝端/未知端计数与逐行证据完整保存在 JSON 的 `episodes[].sources.*.modes.*`。表中≤100/200列为两个命中计数，括号内为全分母；严重偏差和改坏/改好也使用此全分母，正确轴移坏另以已配对行作分母。", "", "## 结论与边界", ""]
    for source in ("A", "B"):
        for mode in ("default", "loo"):
            metrics = report["totals"][source][mode]
            start, end = metrics["start"], metrics["end"]
            lines.append(f"{source}/{mode}：采用后≤200 ms起端 {start['baseline_within_200ms']}→{start['adopted_within_200ms']}，末端 {end['baseline_within_200ms']}→{end['adopted_within_200ms']}（全分母 {metrics['source_rows']}，配对 {metrics['matched_rows']}）；改坏起/末 {start['made_worse']}/{end['made_worse']}。幂等失败 {metrics['idempotence_failed']}/{metrics['idempotence_checked']}。")
    lines += ["", "本批数据中，中文粗轴起端有净改善，末端改善较小；日文粗轴采用建议后两端≤200 ms总数没有净增加。两种输入均存在改坏原本正确端的情况，不能据此推荐整集无条件采用。", "",
              "上述数字描述这批真实输入上相对用户成品的时间距离，不能直接换算人工用时、采用意愿或声学边界正确率。建议命中率需结合覆盖、配对未知数与改坏数看；幂等可与错误结果同时成立。默认留白此前已参考过少女革命成品，因此不是完全独立封存测试；跨集校准隔离本集，但也只覆盖同作品。",
              "日文粗轴包含音效/音乐/空行，均保留全分母；仅配对到对白答案的行可评分。一对一匹配不处理拆并句，答案未配对与粗轴未配对均需后续抽查。正确轴保持使用粗轴邻句，测的是本实验上下文，不能代表全成品邻句流程。未比较 CTC、未测试 CPU 模型延迟、其他作品/语言/音轨、整季规模上限或人工 GUI。",
              f"本次完成集：{report['completed_episodes']}；未完成：{report['skipped']}；总耗时 {report['elapsed_seconds']:.2f} 秒。",
              "复跑：`python tools/eval/hint_rough_real.py --episodes 6 7 8 9 10 11 12 13 14 15`。`--max-seconds` 控制开始新集的预算，未完成集可通过 `--episodes` 指定；LOO训练集随本次完成集变化，正式比较应复跑全部集。",
              f"机器可读结果：[JSON]({output.name})（逐行证据、校准、来源路径/哈希、环境与耗时）。"]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", nargs="+", type=int, default=list(range(6, 16)))
    parser.add_argument("--max-seconds", type=float, default=480)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/eval/HINT-ROUGH-2026-10-12.json")
    args = parser.parse_args()
    if len(set(args.episodes)) != len(args.episodes):
        parser.error("episodes must be unique")
    started = time.perf_counter()
    cache = Path(os.environ["TEMP"]) / "hint-rough-features"
    ffmpeg = proc.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=10)
    report = dict(schema_version=1, environment=dict(platform=platform.platform(), python=platform.python_version(),
                  numpy=np.__version__, ffmpeg=ffmpeg.stdout.decode("utf-8", errors="replace").splitlines()[0],
                  audio_track=1, feature_hop_ms=HOP_MS, feature_cache=str(cache)),
                  requested_episodes=args.episodes, completed_episodes=[], skipped=[], episodes=[])
    work, samples = {}, {}
    for ep in args.episodes:
        if time.perf_counter() - started >= args.max_seconds:
            report["skipped"].append(dict(episode=ep, reason="time budget before starting episode"))
            continue
        try:
            video, cn, jp, final = inputs(ep)
            for path in (video, cn, jp, final):
                if not path.is_file():
                    raise FileNotFoundError(path)
        except FileNotFoundError as error:
            report["skipped"].append(dict(episode=ep, reason=str(error)))
            continue
        source, gold = read_rows(cn, jp, final)
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (cn, jp, final)}
        video_stat = video.stat()
        hit = waveform_cache_file(video, 1, cache).exists()
        begin = time.perf_counter()
        features = load_features(video, 1, cache)
        decode_seconds = time.perf_counter() - begin
        activity = autotime._snap_activity(features)
        del features
        begin = time.perf_counter()
        mappings = {s: pair_rows(source[s], gold[s], language="zh" if s == "A" else "ja") for s in source}
        unique_gold_times = sorted({(r["start"], r["end"]) for r in gold["B"]})
        samples[ep] = autotime._hint_padding_samples(activity, unique_gold_times)
        data = dict(episode=ep, paths=dict(video=str(video), rough_a=str(cn), rough_b=str(jp), gold=str(final)),
                    subtitle_sha256=before, video_size=video_stat.st_size, video_mtime_ns=video_stat.st_mtime_ns,
                    decode_seconds=decode_seconds, feature_cache_hit=hit, sources={})
        for s in source:
            mapping = mappings[s]
            matched = sum(j >= 0 for j in mapping)
            data["sources"][s] = dict(source_rows=len(source[s]), matched=matched,
                    unmatched=len(source[s]) - matched, gold_rows=len(gold[s]), gold_unmatched=len(gold[s]) - matched,
                    unmatched_source_ids=[r["id"] for r, j in zip(source[s], mapping) if j < 0],
                    unmatched_gold_ids=[r["id"] for j, r in enumerate(gold[s]) if j not in mapping],
                    modes=dict(default=evaluate(activity, source[s], gold[s], mapping, DEFAULT_PADDING)))
        data["evaluation_seconds"] = time.perf_counter() - begin
        report["episodes"].append(data)
        report["completed_episodes"].append(ep)
        work[ep] = activity, source, gold, mappings
        after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (cn, jp, final)}
        data["inputs_unchanged"] = before == after and (video_stat.st_size, video_stat.st_mtime_ns) == (video.stat().st_size, video.stat().st_mtime_ns)
        if not data["inputs_unchanged"]:
            raise RuntimeError("input changed during evaluation")
        print(f"E{ep:02d}: A {data['sources']['A']['matched']}/{len(source['A'])}; B {data['sources']['B']['matched']}/{len(source['B'])}; decode/features {decode_seconds:.3f}s", flush=True)
    for data in report["episodes"]:
        ep = data["episode"]
        activity, source, gold, mappings = work[ep]
        calibration = leave_one_out(samples, ep)
        data["loo_calibration"] = calibration
        begin = time.perf_counter()
        for s in source:
            data["sources"][s]["modes"]["loo"] = evaluate(activity, source[s], gold[s], mappings[s],
                                    (calibration["lead_in_ms"], calibration["lead_out_ms"]))
        data["evaluation_seconds"] += time.perf_counter() - begin
    report["totals"] = {s: {mode: total_metrics(report["episodes"], s, mode) for mode in ("default", "loo")} for s in ("A", "B")}
    report["elapsed_seconds"] = time.perf_counter() - started
    write_report(report, args.output)
    print(json.dumps(dict(episodes=report["completed_episodes"], skipped=report["skipped"],
                         elapsed_seconds=report["elapsed_seconds"], totals=report["totals"]), ensure_ascii=False, indent=1))
    return 0 if report["episodes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
