from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from pipeline import arena


TARGET = "2026-07-18T00:00:00+02:00"
OPEN = "2026-07-17T09:00:00Z"
DEADLINE = "2026-07-17T11:00:00Z"


def challenges():
    anchor = {
        "next_target_start": TARGET,
        "submission_window_opens_at": OPEN,
        "next_submission_deadline": DEADLINE,
    }
    return {"20": dict(anchor), "22": dict(anchor), "24": dict(anchor)}


def result(fmt: str, code: int, ok: bool) -> dict:
    return {
        "challenge_id": arena.DELU_CHALLENGES[fmt],
        "status_code": code,
        "ok": ok,
        "target_start": TARGET,
    }


class ArenaRunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.state = root / "pipeline_state.json"
        self.status = root / "status.json"
        self.globals = patch.multiple(
            arena,
            PROD_CONFIG="G1_C1",
            MODEL_LABEL="Chronos-2 G1_C1",
            REPO_COMMIT="commit123",
            STATE=self.state,
            STATUS=self.status,
            SUBMIT_ENABLED=True,
            ARENA_API_KEY="test-only-key",
            SUBMIT_FORMATS=("point", "quantile", "ensemble"),
            SUBMIT_LEAD_MINUTES=60,
        )
        self.globals.start()
        self.addCleanup(self.globals.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_per_format_dedup_retries_only_failed_format_and_merges_status(self):
        calls = []

        def fake_submit(target, formats, open_challenges):
            self.assertEqual(target, TARGET)
            self.assertIsNotNone(open_challenges)
            calls.append(tuple(formats))
            if len(calls) == 1:
                return {
                    "point": result("point", 200, True),
                    "quantile": result("quantile", 503, False),
                    "ensemble": result("ensemble", 201, True),
                }
            return {"quantile": result("quantile", 200, True)}

        status_base = {
            "repo_commit": "commit123",
            "worker_kind": "forecast-G1_C1",
            "training_mode": "LoRA",
            "adapter_digest": "sha256:digest",
        }
        with (
            patch.object(arena, "open_challenges", return_value=challenges()),
            patch.object(arena, "_now_utc", return_value=pd.Timestamp("2026-07-17T10:30Z")),
            patch.object(arena, "submit", side_effect=fake_submit),
            patch.object(arena, "_status_base", return_value=status_base),
            patch.object(
                arena,
                "_freshness",
                return_value={"load": {"latest": "2026-07-17T10:00:00Z", "age_hours": 0.5}},
            ),
        ):
            first = arena.run_if_due()
            second = arena.run_if_due()

        self.assertIn("failed=['quantile']", first)
        self.assertIn("failed=[]", second)
        self.assertEqual(
            calls,
            [("point", "quantile", "ensemble"), ("quantile",)],
        )
        state = json.loads(self.state.read_text(encoding="utf-8"))
        self.assertEqual(
            state["submitted_live_by_format"],
            {"point": TARGET, "quantile": TARGET, "ensemble": TARGET},
        )
        status = json.loads(self.status.read_text(encoding="utf-8"))
        self.assertEqual(set(status["submissions"]), {"point", "quantile", "ensemble"})
        self.assertEqual(status["submissions"]["quantile"]["status_code"], 200)
        self.assertEqual(status["jobs"]["errors"], 0)
        self.assertIsNone(status["jobs"]["last_error"])
        self.assertNotIn("test-only-key", self.status.read_text(encoding="utf-8"))

    def test_dry_run_makes_no_submission_http_request(self):
        values = [[1.0], [2.0]]
        with (
            patch.multiple(arena, SUBMIT_ENABLED=False, ARENA_API_KEY="unused-key"),
            patch.object(arena.requests, "post") as post,
        ):
            code, body = arena._post("20", TARGET, values)
        self.assertEqual((code, body), (0, "dry-run"))
        post.assert_not_called()

    def test_submission_window_and_fresh_data_lead_boundary(self):
        with (
            patch.multiple(arena, SUBMIT_FORMATS=("point",)),
            patch.object(arena, "open_challenges", return_value=challenges()),
            patch.object(
                arena,
                "_now_utc",
                side_effect=(
                    pd.Timestamp("2026-07-17T08:59:59Z"),
                    pd.Timestamp("2026-07-17T09:30:00Z"),
                    pd.Timestamp("2026-07-17T10:00:00Z"),
                    pd.Timestamp("2026-07-17T11:00:00Z"),
                ),
            ),
            patch.object(
                arena,
                "submit",
                return_value={"point": result("point", 200, True)},
            ) as submit,
            patch.object(
                arena,
                "_status_base",
                return_value={"repo_commit": "commit123", "worker_kind": "forecast-G1_C1"},
            ),
            patch.object(arena, "_freshness", return_value={"load": {"status": "ok"}}),
        ):
            before_open = arena.run_if_due()
            too_early = arena.run_if_due()
            at_earliest = arena.run_if_due()
            at_deadline = arena.run_if_due()

        self.assertTrue(before_open.startswith("not-in-window"))
        self.assertTrue(too_early.startswith("waiting-for-fresh-data"))
        self.assertIn("accepted=['point']", at_earliest)
        self.assertTrue(at_deadline.startswith("not-in-window"))
        submit.assert_called_once_with(TARGET, ["point"], challenges())

    def test_missing_required_area_is_reported_as_missing_freshness(self):
        populated = pd.Series(
            [1.0], index=pd.DatetimeIndex(["2026-07-17T10:00:00Z"])
        )
        empty = pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"))
        cfg = SimpleNamespace(areas=("area-a", "area-b"))
        with (
            patch.object(arena.configs, "get", return_value=cfg),
            patch.object(arena.data, "clear_caches"),
            patch.object(arena.data, "target", side_effect=(populated, empty)),
            patch.object(arena, "_now_utc", return_value=pd.Timestamp("2026-07-17T10:30Z")),
        ):
            freshness = arena._freshness("G1_C1")
        self.assertEqual(freshness["load"], {"latest": None, "status": "fehlt"})


if __name__ == "__main__":
    unittest.main()
