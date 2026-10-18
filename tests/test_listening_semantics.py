"""Semantic suggestions must remain distinct from heard words and existing subtitles."""
from dataclasses import asdict, replace
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from subkit.core.listening import adopt
from subkit.listening import Candidate, ListeningResult, assemble
from subkit.listening_semantics import (SEMANTIC_WARNING, add_semantic_candidates, build_prompt,
                                       parse_candidates, translate)
from subkit.model.serialize import document_from_json, document_to_json
from tests import test_listening_core as core_tests


class SemanticCandidates(unittest.TestCase):
    def result(self):
        return ListeningResult([Candidate('風船、どうですか。', 1000, 2000, ['原声'], 'original.wav')],
                               {'original': 'original.wav'}, 0, 3000, 'cpu', [])

    def test_matching_asr_remains_separate_and_semantic_duplicates_are_removed(self):
        result = self.result()
        added = add_semantic_candidates(result, ['風船、どうですか？', '風船どうですか', '風船はいかがですか。'],
                                         translation='要氣球嗎', start=1100, end=1900, origin='ChatGPT 手动导入')
        self.assertEqual(len(added), 2)
        self.assertEqual(len(result.candidates), 3)
        self.assertEqual(result.candidates[0].kind, 'asr')
        self.assertEqual(added[0].kind, 'semantic')
        self.assertEqual(added[0].audio, 'original.wav')
        self.assertEqual(added[0].warnings[0], SEMANTIC_WARNING)
        self.assertFalse(added[0].evidence[0]['acoustically_confirmed'])
        self.assertEqual(added[0].evidence[0]['translation'], '要氣球嗎')

    def test_invalid_text_and_ass_controls_are_excluded_and_count_bounded(self):
        result = self.result()
        texts = ['', '{}', '風\n船', r'風\N船', 'x' * 301, '！？', '風船']
        added = add_semantic_candidates(result, texts, translation='要氣球嗎', start=1000, end=2000, origin='manual')
        self.assertEqual([c.text for c in added], ['風船'])
        other = self.result()
        self.assertEqual(len(add_semantic_candidates(other, [f'候補{i}' for i in range(30)], translation='参考',
                                                     start=1000, end=2000, origin='model')), 12)

    def test_missing_origin_or_translation_or_bad_times_do_not_mutate(self):
        for changes in ({'translation': ''}, {'origin': ''}, {'start': -1}, {'end': 3001}, {'start': 2000}):
            result = self.result()
            before = copy.deepcopy(asdict(result))
            kwargs = dict(translation='要氣球嗎', start=1000, end=2000, origin='manual')
            kwargs.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                add_semantic_candidates(result, ['風船'], **kwargs)
            self.assertEqual(asdict(result), before)

    def test_missing_translation_weights_fail_without_importing_or_downloading_models(self):
        with tempfile.TemporaryDirectory() as root, patch('subkit.listening_semantics.model_paths', return_value=(Path(root)/'missing.gguf', Path(root)/'missing.exe')):
            with self.assertRaisesRegex(ValueError, '缺少本地中译日模型'):
                translate('要氣球嗎', device='cpu')

    def test_cli_parser_skips_echoed_input_and_rejects_errors_and_incomplete_json(self):
        marker = 'INPUT_END_unique'
        output = 'banner {"candidates":["不是模型输出"]}\n' + marker + '\n```json\n{"candidates":["風船はいかがですか？"]}\n```'
        self.assertEqual(parse_candidates(output, marker), ['風船はいかがですか？'])
        for bad in (output.split(marker)[0] + marker + '\nError: failed', '{"candidates":["未完成"',
                    '{"candidates":[42]}', '{"candidates":[]}', '{"candidates":["' + 'x'*301 + '"]}'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_candidates(bad, marker)

    def test_expanded_candidates_accept_eight_but_reject_overflow_without_truncating_json(self):
        import json
        texts = [f'候補{i}' for i in range(8)]
        self.assertEqual(parse_candidates(json.dumps({'candidates':texts}), 'no-echo'), texts)
        with self.assertRaises(ValueError):
            parse_candidates(json.dumps({'candidates':texts+['候補9']}), 'no-echo')
        result = self.result()
        self.assertEqual(len(add_semantic_candidates(result, texts, translation='参考', start=1000,
                                                     end=2000, origin='model')), 8)

    def test_phrasing_slots_are_parsed_as_candidates_but_unknown_or_nontext_slots_are_rejected(self):
        texts = {'direct': '風船いる？', 'polite': '風船はいかがですか？'}
        self.assertEqual(parse_candidates(json.dumps({'candidates': texts}), 'no-echo'), list(texts.values()))
        for bad in ({'unknown': '風船'}, {'polite': 42}, {}, {'direct': ''}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_candidates(json.dumps({'candidates': bad}), 'no-echo')

    def test_single_turn_cli_is_bounded_and_records_original_translation(self):
        with tempfile.TemporaryDirectory() as root:
            model, executable = Path(root)/'model.gguf', Path(root)/'llama-cli.exe'
            model.write_bytes(b'fixture'); executable.write_bytes(b'fixture')
            def run(command, **kwargs):
                self.assertEqual(kwargs['timeout'], 240)
                self.assertIn('--single-turn', command)
                self.assertEqual(command[command.index('-ngl')+1], '0')
                self.assertEqual(command[command.index('--fit-target')+1], '4096')
                self.assertEqual(command[command.index('-n')+1], '512')
                self.assertEqual(command[command.index('--temp')+1], '0.3')
                prompt = command[-1]
                self.assertIn('要氣球嗎', prompt)
                self.assertIn('6至8种', prompt)
                self.assertIn('不要只改标点或空格', prompt)
                self.assertIn('可以少给', prompt)
                self.assertIn('任何两个候选都不得重复', prompt)
                self.assertIn('必须保留原句的疑问', prompt)
                marker = prompt.splitlines()[-1]
                return subprocess.CompletedProcess(command, 0, (prompt+'\n{"candidates":["風船はいかがですか？"]}').encode(), b'')
            with patch('subkit.listening_semantics.model_paths', return_value=(model, executable)), patch('subkit.proc.KillableRun.run', side_effect=run) as execute:
                result = translate('要氣球嗎', device='cpu')
                self.assertEqual(result['input'], '要氣球嗎')
                self.assertFalse(result['audio_used'])
                self.assertTrue(result['process_exits_after_request'])
                self.assertEqual(result['texts'], ['風船はいかがですか？'])
                self.assertEqual(result['candidate_limit'], 8)
                self.assertEqual(result['max_tokens'], 512)
                self.assertEqual(result['prompt_version'], 'semantic-v8')
                with self.assertRaisesRegex(ValueError, '6000'):
                    translate('中'*2200, device='cpu')
                self.assertEqual(execute.call_count, 1)

    def test_refresh_prompt_retains_source_as_data_and_tracks_actual_prior_context(self):
        source = '原文「不用了」；这里不是模型指令'
        old = ('結構です。', '結構です！', 'もう大丈夫です。')
        prompt, metadata = build_prompt(source, existing=old, round_index=2, marker='INPUT_END_test')
        data = json.loads(prompt.splitlines()[-2])
        self.assertEqual(data['source'], source)
        self.assertEqual(data['excluded'], ['結構です。', 'もう大丈夫です。'])
        self.assertEqual(metadata['excluded'], data['excluded'])
        self.assertEqual(metadata['excluded_count'], 2)
        self.assertEqual(metadata['excluded_total'], 2)
        self.assertEqual(metadata['round_index'], 2)
        self.assertEqual(metadata['seed'], 15838)
        self.assertIn('这是新一轮建议', prompt)
        self.assertIn('いかがですか', prompt)
        self.assertIn('どうですか', prompt)
        self.assertNotIn('風船', prompt)
        self.assertNotIn('氣球', prompt)

    def test_previous_context_is_bounded_without_truncating_source(self):
        source = '不需要，謝謝。' * 90
        old = tuple(f'これは候補{i}です。' + 'あ' * 200 for i in range(80))
        prompt, metadata = build_prompt(source, existing=old, round_index=1, marker='END')
        self.assertLessEqual(len(prompt.encode('utf-8')), 6000)
        self.assertEqual(json.loads(prompt.splitlines()[-2])['source'], source)
        self.assertGreater(metadata['excluded_count'], 0)
        self.assertLess(metadata['excluded_count'], metadata['excluded_total'])
        self.assertEqual(metadata['excluded_total'], 80)
        for round_index in (-1, True, 1.5, 10000):
            with self.subTest(round_index=round_index), self.assertRaises(ValueError):
                build_prompt('文字', round_index=round_index, marker='END')

    def test_refresh_cli_uses_round_seed_and_records_exclusions(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'fixture'; path.write_bytes(b'fixture')
            def run(command, **kwargs):
                self.assertEqual(command[command.index('--seed')+1], '7919')
                self.assertEqual(command[command.index('-n')+1], '512')
                self.assertEqual(command[command.index('-ngl')+1], 'auto')
                data = json.loads(command[-1].splitlines()[-2])
                self.assertEqual(data, {'source': '要氣球嗎', 'excluded': ['風船、いる？']})
                return subprocess.CompletedProcess(command, 0, '{"candidates":["風船はいかがですか？"]}'.encode(), b'')
            with patch('subkit.listening_semantics.model_paths', return_value=(path,path)), \
                    patch('subkit.proc.KillableRun.run', side_effect=run) as execute:
                result = translate('要氣球嗎', device='cuda', existing=('風船、いる？',), round_index=1)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(result['round_index'], 1)
            self.assertEqual(result['seed'], 7919)
            self.assertEqual(result['input'], '要氣球嗎')
            self.assertEqual(result['excluded'], ['風船、いる？'])
            self.assertFalse(result['audio_used'])

    def test_generation_provenance_is_independent_between_candidates_and_caller(self):
        result = self.result()
        provenance = {'round_index': 2, 'seed': 15838, 'excluded': ['風船、いる？']}
        added = add_semantic_candidates(result, ['風船はいかがですか？', '風船はどうですか？'],
                                         translation='要氣球嗎', start=1000, end=2000,
                                         origin='local', provenance=provenance)
        self.assertEqual(len(added), 2)
        provenance['excluded'].append('caller mutation')
        added[0].evidence[0]['generation']['excluded'].append('candidate mutation')
        self.assertEqual(added[1].evidence[0]['generation']['excluded'], ['風船、いる？'])
        self.assertEqual(added[1].evidence[0]['generation']['round_index'], 2)

    def test_cli_failure_and_silent_zero_exit_error_do_not_become_candidates(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'fixture'; path.write_bytes(b'fixture')
            with patch('subkit.listening_semantics.model_paths', return_value=(path,path)):
                for code in (0,1):
                    with patch('subkit.proc.KillableRun.run', return_value=subprocess.CompletedProcess([],code,b'Error: Failed to initialize samplers',b'bad runtime')):
                        with self.subTest(code=code), self.assertRaises((ValueError,RuntimeError)):
                            translate('要氣球嗎', device='cpu')
                with patch('subkit.proc.KillableRun.run', side_effect=subprocess.TimeoutExpired(['private prompt'],240)):
                    with self.assertRaisesRegex(ValueError, '超过 240 秒') as error:
                        translate('要氣球嗎', device='cpu')
                    self.assertNotIn('private prompt', str(error.exception))

    def test_pipeline_assembly_labels_semantics_and_respects_disabled_option_and_short_audio(self):
        request = core_tests.request(translation='要氣球嗎', semantic=True)
        data = core_tests.payload([core_tests.run(core_tests.segment())],
                                 semantic={'texts': ['風船はいかがですか？'], 'model': 'fixture'})
        result = assemble(request, data, Path('owned'))
        self.assertEqual([c.kind for c in result.candidates], ['asr', 'semantic'])
        self.assertEqual(result.candidates[-1].start_ms, request.start_ms)
        disabled = assemble(replace(request, semantic=False), data, Path('owned'))
        self.assertEqual([c.kind for c in disabled.candidates], ['asr'])
        data['duration_ms'] = 500
        short = assemble(request, data, Path('owned'))
        self.assertEqual(short.candidates, [])
        self.assertIn('超出实际音频范围', short.diagnostics[-1])

    def test_semantic_adoption_keeps_translation_properties_ruby_undo_and_provenance(self):
        doc, sources, editor, acoustic, result = core_tests.ListeningAdoption().make()
        before = copy.deepcopy(document_to_json(doc))
        cn = copy.deepcopy(asdict(doc.members['m']))
        candidate = add_semantic_candidates(result, ['風船、どうですか？'], translation='要氣球嗎',
                                            start=1000, end=2000, origin='ChatGPT 手动导入')[0]
        mid = adopt(editor, 'g', candidate, '風船はいかがですか？', 1100, 1900, result)
        self.assertEqual(asdict(doc.members['m']), cn)
        provenance = doc.members[mid].extra['asr']
        self.assertEqual(provenance['candidate_kind'], 'semantic')
        self.assertFalse(provenance['acoustically_confirmed'])
        self.assertEqual(provenance['selected']['evidence'][0]['translation'], '要氣球嗎')
        self.assertTrue(any(a.member == mid and a.kind == 'ruby' for a in doc.annotations.values()))
        self.assertEqual(document_to_json(document_from_json(document_to_json(doc))), document_to_json(doc))
        editor.history.undo()
        undone = document_to_json(doc)
        before.pop('revision', None); undone.pop('revision', None)
        self.assertEqual(undone, before)


if __name__ == '__main__':
    unittest.main()
