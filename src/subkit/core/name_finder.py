"""Read-only dialogue-name candidates; suggestions never create characters."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from ..ass.text import VisibleText
from ..languages import get_language, normalize_language_code
from .cc import normalize_speaker

_WORDS = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)
_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_STARTERS = frozenset("""i the a an and but or so then yes no okay ok well hello hi hey
    please thanks thank you we he she it they this that these those my your his her
    our their what why when where who how if now here there today tomorrow yesterday
    monday tuesday wednesday thursday friday saturday sunday
    je tu il elle nous vous ils elles le la les un une et mais oui non alors bonjour
    ich du er sie es wir ihr der die das ein eine und aber ja nein hallo
    yo tú el ella nosotros vosotros ellos ellas hola sí gracias
    eu você ele ela nós vocês eles elas olá sim obrigado
    io lui lei voi loro ciao grazie sì""".split())


def _excluded(book) -> set[str]:
    values = [*book.crowd, *book.not_person, *book.not_name]
    for ch in book.characters:
        values.extend(ch.aliases)
        for form in ch.names.values():
            values.extend([form.compose(), form.compose("short"), form.family, form.given])
    return {normalize_speaker(x) for x in values if x}


def _japanese(text):
    from sudachipy import SplitMode
    from ..reading.engine import Analyzer

    # C combines some full names into 人名-一般; A retains surname/given boundaries.
    morphs = [m for m in Analyzer.get().tokenize(text, mode=SplitMode.A) if m.surface.strip()]
    for i, m in enumerate(morphs):
        if m.pos[:3] != ("名詞", "固有名詞", "人名"):
            continue
        kind = {"姓": "family", "名": "given"}.get(m.pos[3], "person")
        yield m.surface, kind, m.reading
        if kind == "family" and i + 1 < len(morphs):
            n = morphs[i + 1]
            gap = text[m.end:n.start]
            if (n.pos[:4] == ("名詞", "固有名詞", "人名", "名")
                    and "\n" not in gap and (not gap or gap.isspace())):
                yield text[m.start:n.end], "person", m.reading + n.reading


def _latin(text, lang):
    punctuation = get_language(lang, text).sentence_punctuation
    for match in _WORDS.finditer(text):
        word = match.group()
        if (len(word) < 2 or not word[0].isupper() or not any(c.islower() for c in word)
                or not all("LATIN" in unicodedata.name(c, "") for c in word if c.isalpha())
                or word.casefold() in _STARTERS):
            continue
        prefix = text[text.rfind("\n", 0, match.start()) + 1:match.start()].rstrip(" \t\r\"'“”‘’「」『』()[]—–-")
        if not prefix or prefix[-1] in punctuation:
            continue
        yield word, "person", None


def _substrings(texts):
    return {run[i:i + size] for text in texts for run in _HAN.findall(text)
            for size in (2, 3, 4) for i in range(len(run) - size + 1)}


def _translation(candidate_groups, chinese):
    total = Counter(word for words in chinese.values() for word in words)
    together = Counter(word for gid in candidate_groups for word in chinese.get(gid, ()))
    ranked = []
    for word, count in together.items():
        precision = count / total[word]
        coverage = count / len(candidate_groups)
        if count >= 2 and coverage >= 0.6 and precision >= 0.6:
            ranked.append((count * precision, count, len(word), word))
    ranked.sort(reverse=True)
    if not ranked:
        return None
    best = ranked[0]
    # Equal-scoring unrelated phrases are ambiguous; shorter pieces of the same name are not.
    if any(row[:2] == best[:2] and row[3] not in best[3] for row in ranked[1:]):
        return None
    return best[3]


def find_names(doc, sources, book, languages) -> list[dict]:
    """Candidates in visible document order (Latin recurrence >= 2), then by count.

    ``sources`` is the episode source map, ``languages`` the project's lane map.
    Examples contain member/group IDs. Only current visible dialogue is scanned.
    """
    excluded = _excluded(book)
    candidates, groups, chinese = {}, {}, {}
    for group in doc.groups:
        cn = []
        for member in doc.members_of(group):
            if member.role not in ("dialogue", "top", "alt", "song") or member.source_kind == "Comment":
                continue
            source = next((sources[r.src] for r in member.runs if r.src in sources), None)
            lang = normalize_language_code(member.lang or (source.lang if source else languages[member.lane]))
            text = "\n".join(VisibleText(member.text).lines())
            if lang.startswith("zh-"):
                cn.append(text)
            if lang == "ja":
                found = _japanese(text)
            elif get_language(lang, text).word_spaces:
                found = _latin(text, lang)
            else:
                continue
            for surface, kind, reading in found:
                if normalize_speaker(surface) in excluded:
                    continue
                key = (lang, surface, kind)
                item = candidates.setdefault(key, {"text": surface, "lang": lang, "kind": kind,
                    "reading": reading, "count": 0, "examples": [], "episodes": [doc.episode]})
                item["count"] += 1
                example = {"group": group.id, "member": member.id, "text": text}
                if example not in item["examples"] and len(item["examples"]) < 3:
                    item["examples"].append(example)
                groups.setdefault(key, set()).add(group.id)
        chinese[group.id] = _substrings(cn)
    out = []
    for key, item in candidates.items():
        if item["lang"] != "ja" and item["count"] < 2:
            continue
        if item["lang"] == "ja":
            translation = _translation(groups[key], chinese)
            if translation:
                item["suggested_translation"] = translation
        out.append(item)
    return sorted(out, key=lambda item: -item["count"])
