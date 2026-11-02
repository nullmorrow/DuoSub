"""Privacy checks reject leaks without printing sensitive values."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('public_privacy_check', ROOT / 'tools/privacy_check.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class PrivacyCheck(unittest.TestCase):
    def test_only_pseudonymous_noreply_email_is_allowed(self):
        self.assertEqual(checker.content_issues('README.md', checker.PUBLIC_EMAIL.encode()), [])
        email = 'person' + '@' + 'example.test'
        self.assertIn('email address', checker.content_issues('README.md', email.encode()))

    def test_only_ai_co_author_trailers_may_carry_a_service_email(self):
        service = 'noreply' + '@' + 'anthropic.com'
        other = 'person' + '@' + 'example.test'
        trailers = f'Fix\n\nCo-Authored-By: Claude Opus 5.5 <{service}>\nCo-Authored-By: GPT-6.1 Sol (OpenAI Codex)\n'
        self.assertEqual(checker.content_issues('commit-message.txt', trailers.encode()), [])
        for text in (f'Fix\n\nContact {service} for details\n',                       # not a trailer line
                     f'Fix\n\nCo-Authored-By: Someone <{other}>\n',                  # not the allowed address
                     f'Fix\n\nCo-Authored-By: Claude <{service}> and {other}\n'):    # extra address on the line
            with self.subTest(text=text):
                self.assertIn('email address', checker.content_issues('commit-message.txt', text.encode()))
        self.assertIn('email address', checker.content_issues('README.md', trailers.encode()))   # files stay strict

    def test_personal_home_paths_are_rejected(self):
        for text in ('C:' + '/Users/' + 'person/data', '/' + 'home/' + 'person/data', '/' + 'Users/' + 'person/data',
                     'C:' + '/Users/' + '用户名/data', '/' + 'home/' + '使用者/data'):
            self.assertIn('personal home path', checker.content_issues('file.txt', text.encode()))

    def test_system_paths_and_synthetic_drive_paths_are_not_personal_homes(self):
        for text in ('C:/Windows/Fonts', 'C:/Program Files/Tool', 'Z:/gone/font.ttf'):
            self.assertEqual(checker.content_issues('fixture.py', text.encode()), [])

    def test_credentials_and_private_key_markers_are_rejected(self):
        samples = ('gh' + 'p_' + 'A' * 40, 'sk-' + 'proj-' + 'a' * 32,
                   'AK' + 'IA' + 'A' * 16, '-----BEGIN ' + 'DSA ' + 'PRIVATE KEY-----')
        for value in samples:
            self.assertIn('credential or private key', checker.content_issues('file.txt', value.encode()))

    def test_private_local_files_and_data_are_rejected(self):
        for name in ('duosub.local.json', '.privacy-blocklist.local.json', '.env', 'Projects/p/document.json',
                     'migration/old.bundle', 'reports/run.txt', 'project/journal/backup.json', 'media/clip.wav'):
            self.assertIn('private or generated file', checker.content_issues(name, b''))

    def test_forced_internal_documents_and_evaluation_data_are_rejected(self):
        for name in ('tools/eval/data/results.json', 'docs/eval/notes.json',
                     'docs/CLAUDE-HANDOFF-private.md', 'docs/ui-redesign/current/screen.png'):
            self.assertIn('private or generated file', checker.content_issues(name, b''))

    def test_synthetic_baseline_unit_tests_and_documented_example_are_allowed(self):
        for name in ('tests/baseline/test_core.py', 'duosub.local.example.json', 'docs/manual/index.html'):
            self.assertEqual(checker.content_issues(name, b'public data'), [])

    def test_private_blocklist_matches_contents_and_filenames(self):
        blocked = ('previous' + '-identity',)
        self.assertIn('legacy private identifier', checker.content_issues('file.txt', blocked[0].encode(), blocked))
        self.assertIn('legacy private identifier', checker.content_issues(blocked[0] + '.txt', b'', blocked))

    def test_arbitrary_binary_is_rejected(self):
        self.assertIn('unreviewed binary', checker.content_issues('docs/result.bin', b'\x00\xff'))

    def test_commit_identity_requires_both_name_and_email(self):
        current = f'{checker.PUBLIC_NAME} <{checker.PUBLIC_EMAIL}> 1 +0000'
        self.assertTrue(checker.check_identity(current))
        self.assertFalse(checker.check_identity(current.replace(checker.PUBLIC_NAME, 'different', 1)))


if __name__ == '__main__':
    unittest.main()
