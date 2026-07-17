from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from loaders import config


class AtomicCsvTest(unittest.TestCase):
    def test_success_replaces_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "series.csv"
            destination.write_text("old\n", encoding="utf-8")

            config.atomic_to_csv(
                pd.DataFrame({"date": ["2026-07-17"], "value": [1.0]}),
                destination,
                index=False,
            )

            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                "date,value\n2026-07-17,1.0\n",
            )

    def test_failed_write_keeps_last_complete_file(self):
        class BrokenFrame:
            @staticmethod
            def to_csv(handle, **_kwargs):
                handle.write("partial")
                raise RuntimeError("simulated write failure")

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "series.csv"
            destination.write_text("complete\n", encoding="utf-8")

            with self.assertRaises(RuntimeError):
                config.atomic_to_csv(BrokenFrame(), destination, index=False)

            self.assertEqual(destination.read_text(encoding="utf-8"), "complete\n")
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
