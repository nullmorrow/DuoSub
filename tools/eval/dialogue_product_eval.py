"""Evaluate product dialogue alignment on Utena E06-E15, leave one episode out.

Research float16 emissions/vocals in .local-cache/kalign-eval are reused, but
edges are always computed by the product. Missing audio caches use the whole
episode worker. Gold is used only for pairing, training other episodes and
scoring. No model loads at import time or while building subtitle rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from subkit import proc
from subkit.kalign import cache, dialogue, models, song, worker
from subkit.model.document import Document, Group, Member, Run
from subkit.sources.jp import JpOptions, clean_line
from tools.eval.hint_rough_real import inputs, pair_rows, read_rows, score_rows

CACHE = Path('./sample-media/kalign-eval')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", nargs="+", type=int, default=list(range(6, 16)))
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/eval/DIALOGUE-PRODUCT-2026-10-13.json")
    args = parser.parse_args(argv)
    if len(set(args.episodes)) != len(args.episodes):
        parser.error("episodes must be unique")
    return args


def build_lines(source, ep):
    """Rough rows alone determine text/windows; gold never enters generation.

    A uses the overlapping rough JP cue with the greatest overlap, as in the
    research evaluation; B uses its own cue. Preserve empty rows/denominators.
    """
    lines = {}
    doc = Document(f"E{ep:02d}")
    for name in ("A", "B"):
        out = []
        for i, row in enumerate(source[name]):
            text = row["text"]
            if name == "A":
                best, text = 0, ""
                for cue in source["B"]:
                    overlap = min(row["end"], cue["end"]) - max(row["start"], cue["start"])
                    if overlap > best:
                        best, text = overlap, cue["text"]
            text = " ".join(clean_line(t, JpOptions()).text for t in text.splitlines())
            mid = f"E{ep:02d}/{name}/{i}"
            member = Member(mid, mid, "ja", "original", "dialogue", row["start"], row["end"],
                            runs=[Run(text, ins=mid, a=0, b=len(text))])
            doc.members[mid] = member
            doc.groups.append(Group(mid, members=[mid]))
            kana, signature = dialogue.spoken_kana(doc, member)
            out.append(dialogue.DialogueLine(mid, mid, row["start"], row["end"], kana, signature))
        lines[name] = out
    return lines


def research_audio(directory):
    """Read research NPZs with closed handles; never load pickled edges."""
    directory = Path(directory)
    if not all((directory / name).is_file() for name in ("mms.npz", "kana.npz", "vocals16k.npy")):
        return None
    emissions = []
    for name in ("mms", "kana"):
        with np.load(directory / f"{name}.npz", allow_pickle=False) as z:
            star = int(z["star"])
            emissions.append(song.ModelEmission(name, z["lp"].astype(np.float32), float(z["frame_ms"]),
                             json.loads(str(z["vocab"])), int(z["blank"]), None if star < 0 else star))
    vocals = np.load(directory / "vocals16k.npy", allow_pickle=False).astype(np.float32)
    return emissions, models.rms_db_10ms(vocals, rate=16000)


def keyframes(directory, video):
    path = Path(directory) / "keyframes.npy"
    if path.is_file():
        return np.load(path, allow_pickle=False)
    out = proc.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time,flags",
                    "-of", "csv=p=0", str(video)], capture_output=True, text=True, encoding="utf-8", check=True).stdout
    values = [float(t) * 1000 for t, _, flags in (line.partition(",") for line in out.splitlines()) if "K" in flags and t]
    values = np.unique(np.round(np.asarray(values)))
    cache.write_array(path, values)
    return values


def episode_data(ep, cache_dir):
    video, cn, jp, final = inputs(ep)
    source, gold = read_rows(cn, jp, final)
    mapping = {s: pair_rows(source[s], gold[s], language="zh" if s == "A" else "ja") for s in ("A", "B")}
    lines = build_lines(source, ep)
    all_lines = lines["A"] + lines["B"]
    directory = Path(cache_dir) / f"E{ep:02d}"
    audio = research_audio(directory)
    if audio is not None:
        edges = dialogue.align_dialogue(all_lines, *audio)
        audio_source = "research emissions"
    else:
        request = worker.Request(str(video), 1, None, str(cache_dir / "product"), all_lines, {}, "dialogue")
        records = worker.run_request(request)
        edges = [None if row["speech_start"] is None else
                 dialogue.DialogueEdge(**{k: v for k, v in row.items() if k != "signature"}) for row in records]
        audio_source = "product worker"
    n = len(lines["A"])
    return dict(source=source, gold=gold, mapping=mapping, lines=lines,
                edges={"A": edges[:n], "B": edges[n:]}, keyframes=keyframes(directory, video), audio_source=audio_source)


def learn_leave_one_out(data, excluded, name):
    edges, manual = [], []
    for ep, d in data.items():
        if ep == excluded:
            continue
        gold = d["gold"][name]
        for line, edge, j in zip(d["lines"][name], d["edges"][name], d["mapping"][name]):
            if j < 0:
                continue
            edges.append(edge)
            manual.append(dict(mid=line.mid, start=gold[j]["start"], end=gold[j]["end"],
                               next_start=gold[j + 1]["start"] if j + 1 < len(gold) else None))
    return dialogue.learn_habit(edges, manual)


def scoring_rows(d, name, suggestions):
    rows = []
    for row, j, sug in zip(d["source"][name], d["mapping"][name], suggestions):
        current = (row["start"], row["end"])
        answer = d["gold"][name][j] if j >= 0 else None
        rows.append(dict(source_id=row["id"], rough_ms=current,
                         gold_ms=(answer["start"], answer["end"]) if answer else None,
                         suggested_ms=sug, adopted_ms=sug or current, idempotent=True, gold_suggested_ms=None))
    return rows


def evaluate(data):
    report = dict(results={}, parameters={}, audio_sources={ep: d["audio_source"] for ep, d in data.items()})
    for name in ("A", "B"):
        for previous in ("current", "suggested", None):
            label = previous or "none"
            rows = []
            for ep, d in data.items():
                lead_in, lead_out, source, n_start, n_end = learn_leave_one_out(data, ep, name)
                report["parameters"][f"{name}/E{ep:02d}"] = dict(lead_in=lead_in, lead_out=lead_out,
                    source=source, n_start=n_start, n_end=n_end, training_episodes=[e for e in data if e != ep])
                times = [(r["start"], r["end"]) for r in d["source"][name]]
                suggestions = dialogue.suggest_times(d["edges"][name], times, d["keyframes"],
                                  lead_in_ms=lead_in, lead_out_ms=lead_out, previous=previous)
                rows += scoring_rows(d, name, suggestions)
            metrics = score_rows(rows)
            report["results"][f"{name}/previous={label}"] = metrics
            st, en = metrics["start"], metrics["end"]
            print(f"{name} previous={label:9s} start ≤200 {st['baseline_within_200ms']}→{st['adopted_within_200ms']} "
                  f"(≤100 {st['baseline_within_100ms']}→{st['adopted_within_100ms']}) worse/better {st['made_worse']}/{st['made_better']} | "
                  f"end ≤200 {en['baseline_within_200ms']}→{en['adopted_within_200ms']} "
                  f"(≤100 {en['baseline_within_100ms']}→{en['adopted_within_100ms']}) worse/better {en['made_worse']}/{en['made_better']} | "
                  f"n={st['denominator']} matched={st['matched']} covered={st['covered']}", flush=True)
    report["limitations"] = "Real audio accuracy only; C-key idempotence and UI adoption are not evaluated here."
    return report


def main(argv=None):
    args = parse_args(argv)
    started = time.perf_counter()
    data = {}
    for ep in args.episodes:
        data[ep] = episode_data(ep, args.cache_dir)
        print(f"E{ep:02d} ready ({data[ep]['audio_source']})", flush=True)
    report = evaluate(data)
    report.update(episodes=args.episodes, elapsed_seconds=time.perf_counter() - started)
    cache.write_json(args.output, report)
    print(f"elapsed {report['elapsed_seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
