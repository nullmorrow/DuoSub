"""Qt-free request assembly and edit detection, shared by GUI and evaluation."""
from __future__ import annotations

from subkit.core import karaoke
from subkit.core.readings import visible_map
from subkit.core.songs import song_span
from . import cache, dialogue, song, worker


def line_signature(doc, mid):
    member = doc.members.get(mid)
    if member is None:
        return None
    view = karaoke.effective(doc, member)
    return (visible_map(member)[0].text, tuple((b.sung, b.a, b.b) for b in view.beats))


def build_lines(doc, gids, *, fixed=True, overwrite=False, free=()):
    """Evaluation passes fixed=False so reference timing cannot constrain predictions.

    Protected lines keep their real (timed) beats as narrow windows, as context for their neighbours.
    Lines in ``free`` (the line the user explicitly asked to redo) and placeholder beats (no real time,
    e.g. a kanji block whose reading changed) are never fixed.
    """
    idx = karaoke.KIndex(doc)
    free = set(free)
    lines, hand, signatures, references = [], [], {}, []
    for gid in gids:
        for member in doc.members_of(doc.group(gid)):
            if member.lane != "original" or member.role != "song" or member.source_kind == "Comment":
                continue
            view = karaoke.effective(doc, member, idx=idx)
            if not view.beats:
                continue
            protected = view.state == "confirmed" or any(b.by in ("tap", "drag", "manual") for b in view.beats)
            anchor = fixed and protected and not overwrite and member.id not in free
            beats = [song.BeatInput(b.sung, from_ruby=b.ann is not None, a=b.a,
                     fixed=(b.t0, b.t1) if anchor and b.timed and b.t1 is not None else None) for b in view.beats]
            lines.append(song.LineInput(member.id, member.start, member.end, beats, visible_map(member)[0].text))
            signatures[member.id] = line_signature(doc, member.id)
            references.append({"mid": member.id, "start": member.start,
                               "beats": [{"t0": b.t0, "by": b.by} for b in view.beats]})
            hand.extend({"mid": member.id, "index": i, "t0": b.t0, "by": b.by}
                        for i, b in enumerate(view.beats) if b.by in ("tap", "drag") and b.t0 is not None)
    lines.sort(key=lambda line: (line.start, line.end))
    refs = {r["mid"]: r for r in references}
    return lines, hand, signatures, [refs[line.mid] for line in lines]


def build_song_request(doc, gid, *, video, audio_track, cache_dir, options):
    gids = song_span(doc, gid)
    options = dict(options)
    requested_gids = options.pop("requested_gids", [gid])
    # 这一句 is an explicit redo whose adoption replaces the whole line: never anchor it to its old times.
    free_requested = options.pop("free_requested", False)
    free = [m for g in requested_gids for m in doc.group(g).members] if free_requested else ()
    lines, hand, signatures, _refs = build_lines(doc, gids, overwrite=options.get("overwrite", False), free=free)
    if not lines:
        raise ValueError("这一首歌没有可对齐的原声拍")
    requested = [line.mid for line in lines if doc.members[line.mid].group in requested_gids]
    options["hand_beats"] = hand
    request = worker.Request(str(video), audio_track,
        cache.rounded_clip((max(0, min(l.start for l in lines) - 2000), max(l.end for l in lines) + 2000)),
        str(cache_dir), lines, options)
    return request, {"requested_ids": requested, "signatures": signatures}


def build_dialogue_request(doc, *, video, audio_track, cache_dir):
    """All visible Japanese original dialogue rows, using group time windows."""
    lines, members, signatures = [], {}, {}
    for group in doc.groups:
        if group.kind != "dialogue" or "hidden" in group.flags:
            continue
        for member in doc.members_of(group):
            if (member.lane != "original" or member.lang != "ja"
                    or member.role == "comment" or member.source_kind == "Comment"
                    or not visible_map(member)[0].text.strip()):
                continue
            kana, signature = dialogue.spoken_kana(doc, member)
            if not kana:
                continue
            start, end = doc.group_time(group)
            lines.append(dialogue.DialogueLine(group.id, member.id, start, end, kana, signature))
            members.setdefault(group.id, []).append(member.id)
            signatures[member.id] = signature
    lines.sort(key=lambda line: (line.start, line.end))
    if not lines:
        raise ValueError("这一集没有可对齐的日文台词")
    return (worker.Request(str(video), audio_track, None, str(cache_dir), lines, {}, kind="dialogue"),
            {"members": members, "signatures": signatures})
