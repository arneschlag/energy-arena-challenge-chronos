from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import finetune


class AdapterPromotionTest(unittest.TestCase):
    @staticmethod
    def _candidate(root: Path, name: str, value: str) -> Path:
        candidate = root / name
        candidate.mkdir()
        (candidate / "marker").write_text(value, encoding="utf-8")
        return candidate

    def test_recurring_promotion_atomically_switches_current_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._candidate(root, "candidate-one", "one")
            current = finetune._promote_candidate(root, first, "G1_C1", "one")
            self.assertTrue(current.is_symlink())
            self.assertEqual((current / "marker").read_text(encoding="utf-8"), "one")

            second = self._candidate(root, "candidate-two", "two")
            current = finetune._promote_candidate(root, second, "G1_C1", "two")
            self.assertTrue(current.is_symlink())
            self.assertEqual((current / "marker").read_text(encoding="utf-8"), "two")
            rollback = root / "rollback-two"
            self.assertTrue(rollback.is_symlink())
            self.assertEqual((rollback / "marker").read_text(encoding="utf-8"), "one")

    def test_legacy_current_directory_is_retained_as_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._candidate(root, "current", "legacy")
            candidate = self._candidate(root, "candidate", "new")
            current = finetune._promote_candidate(root, candidate, "G3_C5", "migration")
            self.assertTrue(current.is_symlink())
            self.assertEqual((current / "marker").read_text(encoding="utf-8"), "new")
            self.assertEqual(
                (root / "rollback-migration" / "marker").read_text(encoding="utf-8"),
                "legacy",
            )


class FineTuneStatusTest(unittest.TestCase):
    def test_success_updates_summary_model_metadata(self):
        manifest = {
            "repo_commit": "commit123",
            "base_model": "amazon/chronos-2",
            "training_mode": "LoRA",
            "adapter_digest": "sha256:adapterdigest",
            "train_start": "2025-07-17",
            "train_end": "2026-07-17",
            "num_steps": 300,
        }
        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "status.json"
            env = {
                "PROD_CONFIG": "G3_C5",
                "REPO_COMMIT": "commit123",
                "STATUS_PATH": str(status_path),
            }
            with (
                patch.dict(os.environ, env, clear=False),
                patch.object(finetune, "_fit_and_promote", return_value=manifest),
            ):
                returned = finetune.run()

            self.assertEqual(returned, manifest)
            state = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(state["worker_kind"], "forecast-G3_C5")
            self.assertEqual(state["training_mode"], "LoRA")
            self.assertEqual(state["adapter_digest"], "sha256:adapterdigest")
            self.assertEqual(state["model"]["base_model"], "amazon/chronos-2")
            self.assertEqual(state["fine_tune"]["status"], "ok")
            self.assertEqual(state["fine_tune"]["num_steps"], 300)

    def test_failure_records_only_exception_type(self):
        leaked_value = "must-never-enter-status"
        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "status.json"
            env = {
                "PROD_CONFIG": "G1_C1",
                "REPO_COMMIT": "commit456",
                "STATUS_PATH": str(status_path),
            }
            with (
                patch.dict(os.environ, env, clear=False),
                patch.object(
                    finetune,
                    "_fit_and_promote",
                    side_effect=RuntimeError(leaked_value),
                ),
                self.assertRaises(RuntimeError),
            ):
                finetune.run()

            raw = status_path.read_text(encoding="utf-8")
            state = json.loads(raw)
            self.assertEqual(state["fine_tune"]["status"], "error")
            self.assertEqual(state["fine_tune"]["error_type"], "RuntimeError")
            self.assertNotIn(leaked_value, raw)


if __name__ == "__main__":
    unittest.main()
