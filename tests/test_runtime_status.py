from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.runtime_status import RuntimeStatusError, update_status


class RuntimeStatusTest(unittest.TestCase):
    def test_recursive_merge_and_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "status.json"
            first = update_status(
                path,
                repo_commit="abc123",
                jobs={"last_24h": 2, "errors": 0},
                submissions={"point": {"status_code": 200}},
            )
            second = update_status(
                path,
                jobs={"last_24h": 3},
                submissions={"quantile": {"status_code": 200}},
            )
            self.assertIn("updated_at", first)
            self.assertTrue(second["updated_at"].endswith("Z"))
            self.assertEqual(second["jobs"], {"last_24h": 3, "errors": 0})
            self.assertEqual(set(second["submissions"]), {"point", "quantile"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), second)

    def test_rejects_nested_secret_without_leaking_value(self):
        secret = "extremely-private-value"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "status.json"
            with self.assertRaises(RuntimeStatusError) as caught:
                update_status(path, model={"arena_api_key": secret})
            self.assertIn("model.arena_api_key", str(caught.exception))
            self.assertNotIn(secret, str(caught.exception))
            self.assertFalse(path.exists())

    def test_existing_secret_is_removed_on_next_safe_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "status.json"
            path.write_text(
                json.dumps(
                    {
                        "repo_commit": "abc",
                        "smtp_password": "old-secret",
                        "nested": {"access_token": "old-token", "safe": True},
                    }
                ),
                encoding="utf-8",
            )
            result = update_status(path, worker_kind="forecast")
            self.assertNotIn("smtp_password", result)
            self.assertNotIn("access_token", result["nested"])
            self.assertTrue(result["nested"]["safe"])

    def test_invalid_existing_json_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "status.json"
            original = "not-json"
            path.write_text(original, encoding="utf-8")
            with self.assertRaises(RuntimeStatusError):
                update_status(path, jobs={"errors": 1})
            self.assertEqual(path.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
