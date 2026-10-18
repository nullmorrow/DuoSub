"""Read-only dialogue cache projection; no models and no document writes."""
from __future__ import annotations

import json
from pathlib import Path

from subkit.audio import ffmpeg_audio_index
from subkit.core.readings import visible_map
from subkit.kalign import cache, dialogue, tokens

# Particle offsets per visible text. Validating a line's signature needs them, and computing them one line
# at a time creates a Sudachi dictionary each time (~70 ms); prefetch() fills this in one batch.
_PARTICLES: dict[str, frozenset] = {}


def _originals(doc, records):
    for member in doc.members.values():
        if member.id in records and member.lane == "original":
            yield member


def prefetch(doc, records) -> None:
    """Compute particle offsets for every aligned original line in one dictionary load."""
    texts = list(dict.fromkeys(visible_map(m)[0].text for m in _originals(doc, records)))
    missing = [t for t in texts if t not in _PARTICLES]
    if missing:
        for text, found in zip(missing, tokens.particle_positions_many(missing)):
            _PARTICLES[text] = frozenset(found)


def signature(doc, member) -> str:
    text = visible_map(member)[0].text
    if text not in _PARTICLES:
        _PARTICLES[text] = frozenset(tokens.particle_positions_many([text])[0])
    return dialogue.spoken_kana(doc, member, particles=set(_PARTICLES[text]))[1]


def load_records(video, audio_track, cache_dir):
    if not video:
        return {}
    try:
        path = cache.cache_directory(Path(cache_dir) / "kalign", video, ffmpeg_audio_index(audio_track), None) / "dialogue.json"
        return records_from_json(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, KeyError):
        return {}


def records_from_json(rows):
    """Treat malformed cache entries as absent, never as usable boundaries."""
    if not isinstance(rows, list):
        raise ValueError("invalid dialogue results")
    records = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("mid"), str) or not isinstance(row.get("signature"), str):
            continue
        a, b = row.get("speech_start"), row.get("speech_end")
        if a is not None or b is not None:
            if type(a) is not int or type(b) is not int or b < a:
                continue
        flags = row.get("flags", [])
        if not isinstance(flags, list) or any(not isinstance(f, str) for f in flags):
            continue
        try:
            conf = float(row.get("conf", 0))
        except (ValueError, TypeError):
            continue
        records[row["mid"]] = dict(row, conf=conf, flags=flags)
    return records


def valid_edges(doc, group, records):
    edges, stale = [], False
    if group.kind != "dialogue" or "hidden" in group.flags:
        return edges, stale
    for member in doc.members_of(group):
        row = records.get(member.id)
        if (row is None or member.lane != "original" or member.lang != "ja"
                or member.role == "comment" or member.source_kind == "Comment"):
            continue
        if signature(doc, member) != row["signature"]:
            stale = True
            continue
        if row.get("speech_start") is not None:
            edges.append(dialogue.DialogueEdge(member.id, row["speech_start"], row["speech_end"], row["conf"], row["flags"]))
    return edges, stale


def group_edge(doc, group, records):
    edges, _ = valid_edges(doc, group, records)
    if not edges:
        return None
    return dialogue.DialogueEdge(edges[0].mid, min(e.speech_start for e in edges),
                                max(e.speech_end for e in edges), max(e.conf for e in edges),
                                list(dict.fromkeys(f for e in edges for f in e.flags)))


def manual_edges(doc, group):
    """Manual group-extreme edges, with the same legacy rules as hint learning."""
    start, end = doc.group_time(group)
    manual = {}
    for name, time in (("start", start), ("end", end)):
        manual[name] = "manual" if any(
            getattr(m, name) == time and (
                m.extra.get("time_src", {}).get(name) == "manual"
                or name not in m.extra.get("time_src", {}) and (
                    group.state == "confirmed" or "time_src" not in m.extra and m.extra.get("time_manual")))
            for m in doc.members_of(group)) else "other"
    return manual


def habit_samples(doc, records):
    """Group-extreme manual edges, including legacy provenance, as energy calibration.

    Keep all neighbours and explicit next_start so pooling episodes cannot join ends.
    A multi-member group contributes one start/end sample, rather than overweighting it.
    """
    groups = sorted((g for g in doc.groups if g.kind == "dialogue" and g.members), key=doc.group_time)
    prefetch(doc, records)
    edges, rows = [], []
    for i, group in enumerate(groups):
        start, end = doc.group_time(group)
        edge = group_edge(doc, group, records)
        mid = f"{doc.episode}:{group.id}"
        manual = manual_edges(doc, group)
        if edge:
            edges.append(dialogue.DialogueEdge(mid, edge.speech_start, edge.speech_end, edge.conf, edge.flags))
        rows.append(dict(mid=mid, start=start, end=end, time_src=manual,
                         next_start=doc.group_time(groups[i + 1])[0] if i + 1 < len(groups) else None))
    return edges, rows
