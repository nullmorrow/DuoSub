"""Qt-free song ranges shared by effect editing and KTV layout."""
from __future__ import annotations

from ..model.document import Document


def song_span(doc: Document, gid: str) -> list[str]:
    """Return neighbouring lyrics of one family, ignoring interleaved non-song groups.

    Document order is retained. A different family or a lyric gap over 15 seconds
    ends the span; overlap and exactly 15 seconds remain in the same song.
    """
    songs = []
    for group in doc.groups:
        if group.kind != "song":
            continue
        members = [doc.members[mid] for mid in group.members
                   if doc.members[mid].role == "song" and doc.members[mid].source_kind != "Comment"]
        family = next((m.style.split(".", 1)[0] for m in members
                       if m.style.split(".", 1)[0] in ("OP", "ED", "IN")), None)
        songs.append((group.id, family, min((m.start for m in members), default=0),
                      max((m.end for m in members), default=0)))
    at = next((i for i, song in enumerate(songs) if song[0] == gid), None)
    if at is None or songs[at][1] is None:
        return []
    family = songs[at][1]
    first = last = at
    while first > 0 and songs[first - 1][1] == family and songs[first][2] - songs[first - 1][3] <= 15000:
        first -= 1
    while last + 1 < len(songs) and songs[last + 1][1] == family and songs[last + 1][2] - songs[last][3] <= 15000:
        last += 1
    return [song[0] for song in songs[first:last + 1]]
