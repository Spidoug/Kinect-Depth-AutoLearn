"""Dependency-free regression tests for helper state and process restart logic."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).with_name('autolearn_helper.py')
spec = importlib.util.spec_from_file_location('autolearn_helper_under_test', MODULE_PATH)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class HelperTests(unittest.TestCase):
    def test_atomic_json_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sub' / 'state.json'
            helper.atomic_json(path, {'state': 'ready', 'label': 'depth'})
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')),
                             {'state': 'ready', 'label': 'depth'})
            self.assertEqual(list(path.parent.glob('*.tmp')), [])

    def test_missing_bridge_is_rate_limited(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(helper, 'BIN_DIR', Path(directory)):
                helper.bridges.clear()
                helper.bridge_retry_after.clear()
                self.assertIsNone(helper.ensure_bridge('missing_bridge', 38173))
                retry = helper.bridge_retry_after['missing_bridge']
                self.assertIsNone(helper.ensure_bridge('missing_bridge', 38173))
                self.assertEqual(retry, helper.bridge_retry_after['missing_bridge'])

    def test_rejects_invalid_training_request(self):
        with self.assertRaises(ValueError):
            helper.launch_training({'dataset': '', 'trainer': ''})
        with self.assertRaises(ValueError):
            helper.launch_training([])


if __name__ == '__main__':
    unittest.main()
