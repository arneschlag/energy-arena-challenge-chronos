from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pipeline.summary import G1, G3, LoadedState, build_daily_summary, load_state, render_model_block


class SummaryTest(unittest.TestCase):
    def test_two_explicit_blocks_and_only_relevant_freshness(self):
        common = {
            "repo_commit": "abc1234",
            "worker_kind": "forecast-worker",
            "model": {
                "base_model": "amazon/chronos-2",
                "parameter_count": 120_000_000,
            },
            "target_start": "2026-07-18T00:00:00+02:00",
            "submissions": {
                "point": {"challenge_id": 20, "status_code": 200},
                "quantile": {"challenge_id": 22, "status_code": 200},
            },
            "jobs": {"last_24h": 25, "errors": 0},
        }
        g1 = dict(
            common,
            training_mode="zero-shot",
            data_freshness={
                "actual": {"latest": "2026-07-17T09:00:00Z", "age_hours": 0.5},
                "weather": {"latest": "DO-NOT-SHOW"},
            },
            # Muss wegen der Allowlist vollstaendig ignoriert werden.
            arena_api_key="very-secret-key",
        )
        g3 = dict(
            common,
            training_mode="LoRA",
            adapter_digest="sha256:adapter123",
            data_freshness={
                "load": {"latest": "load-ts", "age_hours": 0.2},
                "forecast_weather": {"latest": "weather-ts", "age_hours": -36.0},
                "aux_price": {"latest": "price-ts", "age_hours": 2.0},
                "aux_solar": {"latest": "solar-ts", "age_hours": 3.0},
                "aux_wind": {"latest": "wind-ts", "age_hours": 4.0},
            },
        )
        result = build_daily_summary(
            LoadedState(Path("g1.json"), g1),
            LoadedState(Path("g3.json"), g3),
            now=datetime(2026, 7, 17, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")),
            env={},
        )
        self.assertIn("G1_C1 — Chronos 2 univariat", result.body)
        self.assertIn("G3_C5 — Chronos 2 multivariat (Group Attention, C5)", result.body)
        self.assertEqual(
            result.body.count(
                "Basismodell: amazon/chronos-2 (120 Mio. Parameter)"
            ),
            2,
        )
        self.assertIn("36.0 h voraus", result.body)
        self.assertIn("zero-shot; Adapter: keiner", result.body)
        self.assertIn("LoRA; Adapter-Digest: sha256:adapter123", result.body)
        self.assertEqual(result.body.count("Worker-Kind: forecast-worker"), 2)
        self.assertIn("point#20: 200", result.body)
        g1_block, g3_block = result.body.split("G3_C5", 1)
        self.assertNotIn("Wetter:", g1_block)
        for label in ("Last:", "Wetter:", "Preis:", "Solar:", "Wind:"):
            self.assertIn(label, g3_block)
        self.assertNotIn("very-secret-key", result.body)

    def test_missing_and_invalid_states_are_robust(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = load_state(Path(tmp) / "missing.json")
            self.assertEqual(missing.data, {})
            self.assertEqual(missing.issue, "Statusdatei fehlt")

            invalid_path = Path(tmp) / "invalid.json"
            invalid_path.write_text("{not json", encoding="utf-8")
            invalid = load_state(invalid_path)
            self.assertIn("ungültiges JSON", invalid.issue or "")

            summary = build_daily_summary(missing, invalid, env={})
            self.assertIn("Status: Statusdatei fehlt", summary.body)
            self.assertIn("keine Submissionresultate gemeldet", summary.body)

    def test_directory_state_path_and_environment_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pipeline_state.json"
            path.write_text(json.dumps({"last_submitted_live": "2026-07-18"}), encoding="utf-8")
            loaded = load_state(tmp)
            block = render_model_block(
                G1,
                loaded,
                {
                    "G1_REPO_COMMIT": "deadbee",
                    "G1_TRAINING_MODE": "zero-shot",
                    "ARENA_API_KEY": "must-not-leak",
                },
            )
            self.assertIn("Repo-Commit: deadbee", block)
            self.assertIn("Liefertag: 2026-07-18", block)
            self.assertNotIn("must-not-leak", block)

    def test_zero_jobs_is_not_treated_as_missing(self):
        block = render_model_block(
            G3,
            LoadedState(None, {"jobs": {"last_24h": 0, "errors": 0}}),
            {},
        )
        self.assertIn("0 gelaufen, 0 Fehler", block)

    def test_worker_transition_fields_are_useful_fallbacks(self):
        block = render_model_block(
            G1,
            LoadedState(
                None,
                {
                    "last_result": "point#20:200 quantile#22:200",
                    "job_status": "ok",
                    "last_duration_seconds": 12.25,
                },
            ),
            {},
        )
        self.assertIn("point#20:200 quantile#22:200", block)
        self.assertIn("letzter Status ok; Dauer 12.2 s", block)


if __name__ == "__main__":
    unittest.main()
