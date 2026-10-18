"""One occurrence's manual per-kanji ruby, including explicit empty decisions."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from ..languages import get_language
from ..model.document import Annotation
from ..model.ids import new_id
from ..textutil import is_kanji
from .readings import anchor_of, anchor_positions, visible_map

if TYPE_CHECKING:
    from .ops import Editor


def split(editor: Editor, mid: str, a: int, b: int, readings: list[str]) -> int:
    doc = editor.doc
    member = doc.members[mid]
    if get_language(member.lang).annotation_system != "ruby":
        return 0
    vt, ids = visible_map(member)
    if not 0 <= a < b <= len(ids):
        raise ValueError("读音范围已改变，请重新选中")
    positions = [i for i in range(a, b) if is_kanji(vt.text[i])]
    if len(positions) < 2 or len(readings) != len(positions) or any(not isinstance(r, str) for r in readings):
        raise ValueError("请为范围内每个汉字填写一个读音（可以留空）")
    if any(ids[i] is None for i in positions):
        raise ValueError("读音范围没有可用的字符身份")
    old = []
    for ann in doc.annotations_of(mid):
        if ann.kind != "ruby":
            continue
        rng = anchor_positions(ann, ids)
        if rng is not None and rng[0] < b and a < rng[1]:
            if rng[0] < a or rng[1] > b:
                raise ValueError("请先选中整个读音范围，再按单个汉字拆开")
            old.append((rng, ann))

    def apply(tx):
        tx.keeps_text = True
        tx.member(mid)
        for _rng, ann in old:
            tx.annotation(ann.id)
            del doc.annotations[ann.id]
        for i, reading in zip(positions, readings):
            text = reading.strip()
            layout = next((copy.deepcopy(ann.layout) for (x, y), ann in old if x <= i < y), {})
            tx.add_annotation(Annotation(
                new_id("a_"), "ruby", mid, anchor_of([ids[i]]), vt.text[i], text,
                status="confirmed" if text else "hidden", origin="manual",
                candidates=[text] if text else [], parts=[[0, 1, text]] if text else [], layout=layout))
        return len(positions)

    return editor.history.run("按单个汉字拆开", apply)
