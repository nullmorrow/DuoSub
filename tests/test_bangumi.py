"""Offline recorded API responses plus explicit synthetic splitting boundaries."""

import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit.core.bangumi import (BangumiError, USER_AGENT, apply_import,
                                BangumiCancelled, build_candidates, fetch_subject, parse_subject_id)
from subkit.core.ops import Editor
from subkit.core.people import CharacterBook, NameForm
from subkit.model.document import Document, Group, Member, Run

FIXTURE = json.loads((Path(__file__).with_name("bangumi_fixture.json")).read_text(encoding="utf-8"))
LANGUAGES = {"original": "ja", "translation": "zh-Hans"}


def synthetic(name="幸平創真", chinese="幸平创真", kana="ゆきひら そうま", romaji="Yukihira Souma"):
    # Synthetic responses keep fallback and conflicting readings deterministic.
    info = [{"key": "别名", "value": [{"k": "日文名", "v": name},
             {"k": "罗马字", "v": romaji}, {"k": "假名", "v": kana}]}]
    if chinese:
        info.append({"key": "简体中文名", "value": chinese})
    return ([{"id": 1, "name": name, "relation": "主角", "actors": [{"name": "松岡禎丞"}]}],
            {1: {"infobox": info}})


class BangumiTests(unittest.TestCase):
    def test_cancel_between_character_requests_discards_partial_fetch(self):
        stopped = [False]
        paths = []
        def read(path, timeout):
            paths.append(path)
            if path == "/subjects/2903":
                return FIXTURE["subject"]
            if path.endswith("/characters"):
                return FIXTURE["characters"]
            stopped[0] = True
            return FIXTURE["details"][path.rsplit("/", 1)[1]]
        with patch("subkit.core.bangumi._read", side_effect=read):
            with self.assertRaises(BangumiCancelled):
                fetch_subject("2903", delay=0, cancelled=lambda: stopped[0])
        self.assertEqual(len(paths), 3)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.book = CharacterBook(Path(temp.name) / "dict" / "characters.json")
        # A preview or apply must never secretly access the network.
        network = patch("subkit.core.bangumi.urlopen", side_effect=AssertionError("unexpected network"))
        network.start()
        self.addCleanup(network.stop)

    def candidates(self, **kwargs):
        chars, details = synthetic(**kwargs)
        return build_candidates({}, chars, details, self.book, LANGUAGES)

    def test_links_numbers_and_rejected_hosts_or_paths(self):
        for value in ("2903", " 02903 ", "https://bangumi.tv/subject/2903/",
                      "http://bgm.tv/subject/2903", "https://chii.in/subject/2903/?x=1#top"):
            self.assertEqual(parse_subject_id(value), 2903)
        for value in ("", "0", "-1", "１２", "https://bangumi.tv.evil/subject/2903",
                      "https://evil" + "@" + "bangumi.tv/subject/2903", "https://bgm.tv/character/2903",
                      "file://bangumi.tv/subject/2903", "https://bgm.tv/subject/1/extra", "1.5"):
            with self.subTest(value=value), self.assertRaisesRegex(BangumiError, "无法从 Bangumi 读取："):
                parse_subject_id(value)

    def test_recorded_preview_order_missing_chinese_and_no_writes(self):
        before = self.book.to_json()
        rows = build_candidates(FIXTURE["subject"], FIXTURE["characters"], FIXTURE["details"],
                                self.book, LANGUAGES)
        self.assertEqual(len(rows), 14)
        self.assertEqual([r["bangumi"] for r in rows], [c["id"] for c in FIXTURE["characters"]])
        self.assertEqual(sum(bool(r["zh_name"]) for r in rows), 2)
        self.assertNotIn("zh-Hans", rows[0]["names"])
        self.assertEqual(rows[0]["cv"], "中村千絵")
        self.assertEqual(rows[0]["names"]["ja"].given, "ユージェニー")
        self.assertEqual(rows[0]["names"]["ja"].family, "ド・ダングラール")
        self.assertEqual(self.book.to_json(), before)
        self.assertFalse(self.book.path.exists())

    def test_soma_split_verified_readings_and_chinese_boundary(self):
        row = self.candidates()[0]
        ja, zh = row["names"]["ja"], row["names"]["zh-Hans"]
        self.assertEqual((ja.family, ja.given, ja.family_reading, ja.given_reading),
                         ("幸平", "創真", "ゆきひら", "そうま"))
        self.assertEqual((zh.family, zh.given), ("幸平", "创真"))
        self.assertEqual(self.candidates(romaji="Sōma Yukihira")[0]["names"]["ja"].family, "幸平")

    def test_sudachi_surname_and_romaji_disagreement_falls_back(self):
        form = self.candidates(name="薙切えりな", chinese="薙切绘里奈", kana="なきり えりな",
                               romaji="Nakiri Erina")[0]["names"]["ja"]
        self.assertEqual((form.family, form.given), ("薙切", "えりな"))
        row = self.candidates(romaji="Someone Else")[0]
        self.assertEqual(row["names"]["ja"].given, "幸平創真")
        self.assertEqual(row["names"]["ja"].family, "")
        self.assertEqual(row["names"]["zh-Hans"].given, "幸平创真")
        self.assertEqual(self.candidates(kana="ゆきひらそうま", romaji="")[0]["names"]["ja"].family, "")

    def test_katakana_given_first_and_uncertain_chinese_whole(self):
        row = self.candidates(name="シェルドン・クーパー", chinese="谢尔顿·库珀", kana="", romaji="Sheldon Cooper")[0]
        ja, zh = row["names"]["ja"], row["names"]["zh-Hans"]
        self.assertEqual((ja.given, ja.family, ja.order, ja.separator),
                         ("シェルドン", "クーパー", "given_first", "・"))
        self.assertEqual(ja.given_reading, "しぇるどん")
        self.assertEqual((zh.given, zh.family, zh.compose()), ("谢尔顿", "库珀", "谢尔顿·库珀"))
        uncertain = self.candidates(name="シェルドン・クーパー", chinese="谢尔顿", kana="", romaji="Sheldon Cooper")[0]
        self.assertEqual(uncertain["names"]["zh-Hans"].given, "谢尔顿")

    def test_mononym_pure_kana_second_chinese_and_guest_defaults(self):
        chars, details = synthetic(name="めぐみん", chinese="", kana="", romaji="Megumin")
        details[1]["infobox"] = [{"key": "别名", "value": [
            {"k": "纯假名", "v": "メグミン"}, {"k": "第二中文名", "v": "惠惠"}]}]
        chars[0]["relation"] = "客串"
        row = build_candidates({}, chars, details, self.book, LANGUAGES)[0]
        self.assertFalse(row["checked"])
        self.assertEqual(row["names"]["ja"].given_reading, "めぐみん")
        self.assertEqual(row["zh_name"], "惠惠")
        chars[0]["relation"] = "配角"
        self.assertTrue(build_candidates({}, chars, details, self.book, LANGUAGES)[0]["checked"])

    def test_opencc_s2tw_and_only_project_chinese_language(self):
        chars, details = synthetic(chinese="幸平创真")
        rows = build_candidates({}, chars, details, self.book,
                                {"original": "ja", "translation": "zh-TW"}, True)
        self.assertEqual(rows[0]["zh_name"], "幸平創真")
        self.assertEqual(rows[0]["names"]["zh-Hant"].given, "創真")
        self.assertNotIn("zh-Hans", rows[0]["names"])
        rows = build_candidates({}, chars, details, self.book, {"original": "ja", "translation": "en"})
        self.assertEqual(set(rows[0]["names"]), {"ja"})

    def test_exact_japanese_or_alias_matching_only_and_ambiguity(self):
        ch = self.book.add_character({"ja": "別人", "zh-Hans": "幸平創真"}, aliases=["幸平創真"])
        self.assertEqual(self.candidates()[0]["match"], ch.id)
        self.assertIsNone(self.candidates(name="幸平 創真")[0]["match"])
        self.book.add_character({"ja": "幸平創真"})
        self.assertIsNone(self.candidates()[0]["match"])
        self.book.characters[0].aliases.clear()
        self.book.characters.pop()
        self.assertIsNone(self.candidates()[0]["match"])

    def test_selected_create_fill_only_empty_source_persistence_and_no_aliases(self):
        existing = self.book.add_character({"ja": NameForm(family="幸平", given="創真", family_reading="人工")},
                                           aliases=["創真"], persona={"tone": "保留"})
        existing.extra["source"] = {"other": "keep"}
        rows = self.candidates()
        self.assertEqual(apply_import(self.book, rows, [0, 0]), [existing.id])
        updated = self.book.character(existing.id)
        self.assertEqual(updated.names["ja"].family_reading, "人工")
        self.assertEqual(updated.names["ja"].given_reading, "そうま")
        self.assertEqual(updated.extra["source"], {"other": "keep", "bangumi": 1})
        self.assertEqual(updated.aliases, ["創真"])
        self.assertEqual(updated.persona["tone"], "保留")
        raw = self.book.path.read_bytes()
        self.assertEqual(CharacterBook.load(self.book.path).to_json(), self.book.to_json())
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        new = self.candidates(name="めぐみん", chinese="", kana="めぐみん", romaji="Megumin")
        self.assertEqual(apply_import(self.book, new, []), [])
        self.assertEqual(self.book.path.read_bytes(), raw)
        cid = apply_import(self.book, new, [0])[0]
        self.assertEqual(self.book.character(cid).extra["source"], {"bangumi": 1})
        self.assertEqual(set(self.book.character(cid).names), {"ja"})

    def test_whole_existing_name_not_duplicated_and_manual_match_kept(self):
        existing = self.book.add_character({"ja": "幸平創真"})
        apply_import(self.book, self.candidates(), [0])
        self.assertEqual(self.book.character(existing.id).full_name("ja"), "幸平創真")
        self.assertEqual(self.book.character(existing.id).names["ja"].given_reading, "ゆきひらそうま")
        rows = self.candidates()
        rows[0]["match"] = existing.id
        self.book.character(existing.id).extra["source"]["bangumi"] = 88
        apply_import(self.book, rows, [0])
        self.assertEqual(self.book.character(existing.id).extra["source"]["bangumi"], 88)

    def test_empty_name_preserves_manual_settings_and_changed_name_gets_no_wrong_reading(self):
        ch = self.book.add_character({"ja": NameForm(short="custom", short_custom="保留", separator="·")})
        ch.extra["source"] = None
        rows = self.candidates()
        rows[0]["match"] = ch.id
        apply_import(self.book, rows, [0])
        form = self.book.character(ch.id).names["ja"]
        self.assertEqual((form.short, form.short_custom, form.separator), ("custom", "保留", "·"))
        form.family = "人工姓"
        form.family_reading = ""
        apply_import(self.book, rows, [0])
        self.assertEqual(self.book.character(ch.id).names["ja"].family_reading, "")
        self.assertEqual(self.book.character(ch.id).extra["source"], {"bangumi": 1})

    def test_invalid_manual_match_and_save_failure_leave_book_unchanged(self):
        rows = self.candidates()
        rows[0]["match"] = "chr_missing"
        with self.assertRaises(ValueError):
            apply_import(self.book, rows, [0])
        self.assertEqual(self.book.characters, [])
        rows[0]["match"] = None
        with patch.object(self.book, "save", side_effect=OSError("disk full")), self.assertRaises(OSError):
            apply_import(self.book, rows, [0])
        self.assertEqual(self.book.characters, [])

    def test_one_undo_redo_restores_file_links_and_manual_speaker(self):
        ch = self.book.add_character({"ja": "別人"})
        group = Group("g1", members=["m1", "m2"])
        doc = Document("E1", groups=[group])
        for mid, speaker in (("m1", {"label": "幸平創真", "character": None}),
                             ("m2", {"label": "幸平創真", "character": ch.id, "by": "user"})):
            doc.members[mid] = Member(mid, "g1", "ja", "original", "dialogue", 0, 1000,
                                      runs=[Run("こんにちは")], speaker=speaker)
        ed = Editor(doc, {}, character_book=self.book)
        raw, before = self.book.path.read_bytes(), self.book.to_json()
        imported = apply_import(self.book, self.candidates(), [0], editor=ed)
        self.assertEqual(len(ed.history.done), 1)
        self.assertEqual(doc.members["m1"].speaker["character"], imported[0])
        self.assertEqual(doc.members["m2"].speaker["character"], ch.id)
        after = self.book.to_json()
        ed.history.undo()
        self.assertEqual(self.book.path.read_bytes(), raw)
        self.assertEqual(self.book.to_json(), before)
        self.assertIsNone(doc.members["m1"].speaker["character"])
        ed.history.redo()
        self.assertEqual(self.book.to_json(), after)
        self.assertEqual(doc.members["m1"].speaker["character"], imported[0])

    def test_undo_initial_import_removes_previously_absent_file(self):
        ed = Editor(Document("E1"), {}, character_book=self.book)
        apply_import(self.book, self.candidates(), [0], editor=ed)
        ed.history.undo()
        self.assertFalse(self.book.path.exists())
        self.assertEqual(self.book.characters, [])

    def test_fetch_recorded_responses_headers_timeout_delay_and_closed_handles(self):
        responses = [FIXTURE["subject"], FIXTURE["characters"],
                     *(FIXTURE["details"][str(c["id"])] for c in FIXTURE["characters"])]
        handles = [io.BytesIO(json.dumps(x, ensure_ascii=False).encode("utf-8")) for x in responses]
        with patch("subkit.core.bangumi.urlopen", side_effect=handles) as opened, patch("subkit.core.bangumi.time.sleep") as sleep:
            subject, chars, details = fetch_subject("2903", timeout=3, delay=0.1)
        self.assertEqual(len(details), 14)
        self.assertEqual(len(chars), 14)
        self.assertEqual(subject["name"], "巖窟王")
        self.assertEqual(opened.call_count, 16)
        self.assertEqual(sleep.call_count, 14)
        self.assertTrue(all(h.closed for h in handles))
        for call in opened.call_args_list:
            self.assertEqual(call.args[0].get_header("User-agent"), USER_AGENT)
            self.assertEqual(call.kwargs["timeout"], 3)
        self.assertNotIn("images", chars[0])
        self.assertNotIn("summary", details[1984])

    def test_fetch_clear_network_json_and_shape_errors(self):
        errors = [URLError("offline"), TimeoutError("timed out"), HTTPError("url", 404, "Not Found", {}, None)]
        for error in errors:
            with patch("subkit.core.bangumi.urlopen", side_effect=error), self.assertRaisesRegex(BangumiError, "无法从 Bangumi 读取："):
                fetch_subject("2903")
        for raw in (b"invalid JSON", b"[]"):
            with patch("subkit.core.bangumi.urlopen", side_effect=[io.BytesIO(raw), io.BytesIO(b"[]")]), self.assertRaises(BangumiError):
                fetch_subject("2903")


if __name__ == "__main__":
    unittest.main()
