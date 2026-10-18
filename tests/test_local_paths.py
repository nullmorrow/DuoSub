"""Machine configuration stays private while workers and GUI share portable paths."""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from subkit.local_paths import local_value


class LocalPaths(unittest.TestCase):
    def test_explicit_environment_wins_without_reading_file(self):
        with patch.dict(os.environ, {'SUBKIT_TEST_PATH': 'external/model'}, clear=True), patch.object(Path, 'read_text') as read:
            self.assertEqual(local_value('SUBKIT_TEST_PATH', 'fallback'), 'external/model')
            read.assert_not_called()

    def test_explicit_empty_environment_stays_empty(self):
        with patch.dict(os.environ, {'SUBKIT_TEST_PATH': ''}, clear=True):
            self.assertEqual(local_value('SUBKIT_TEST_PATH', 'fallback'), '')

    def test_private_file_is_read_from_launcher_directory(self):
        paths = []
        def read(path, **kwargs):
            paths.append(path)
            return json.dumps({'SUBKIT_TEST_PATH': 'private/model'})
        with patch.dict(os.environ, {}, clear=True), patch.object(Path, 'read_text', read):
            self.assertEqual(local_value('SUBKIT_TEST_PATH', 'fallback'), 'private/model')
        self.assertEqual(paths, [Path(__file__).resolve().parents[1] / 'duosub.local.json'])

    def test_missing_and_unreadable_files_fall_back(self):
        for error in (FileNotFoundError(), PermissionError()):
            with patch.dict(os.environ, {}, clear=True), patch.object(Path, 'read_text', side_effect=error):
                self.assertEqual(local_value('SUBKIT_TEST_PATH', Path('fallback')), 'fallback')

    def test_malformed_or_wrong_types_are_ignored(self):
        for text in ('{', '[]', '{"SUBKIT_TEST_PATH":3}', '{"SUBKIT_TEST_PATH":null}', '{"SUBKIT_TEST_PATH":"a\\u0000b"}'):
            with patch.dict(os.environ, {}, clear=True), patch.object(Path, 'read_text', return_value=text):
                self.assertEqual(local_value('SUBKIT_TEST_PATH', 'fallback'), 'fallback')

    def test_empty_private_path_can_disable_an_optional_location(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(Path, 'read_text', return_value='{"SUBKIT_TEST_PATH":""}'):
            self.assertEqual(local_value('SUBKIT_TEST_PATH', 'fallback'), '')


if __name__ == '__main__':
    unittest.main()
