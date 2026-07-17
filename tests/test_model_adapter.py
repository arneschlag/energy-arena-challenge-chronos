from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from experiments import model


class AdapterFailClosedTests(unittest.TestCase):
    @staticmethod
    def _write_adapter(path: Path, config_name: str = "G1_C1") -> None:
        path.mkdir(parents=True)
        (path / "adapter_model.safetensors").write_bytes(b"test-adapter-weights")
        adapter_cfg = {
            "peft_type": "LORA",
            "r": model.LORA_R,
            "lora_alpha": model.LORA_ALPHA,
            "target_modules": list(model.LORA_TARGET_MODULES),
            "base_model_name_or_path": model.MODEL,
        }
        (path / "adapter_config.json").write_text(
            json.dumps(adapter_cfg), encoding="utf-8"
        )
        manifest = {
            "config": config_name,
            "base_model": model.MODEL,
            "training_mode": "LoRA",
            "rank": model.LORA_R,
            "alpha": model.LORA_ALPHA,
            "adapter_digest": model._sha256(path / "adapter_model.safetensors"),
        }
        (path / "production_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    def test_missing_required_adapter_never_publishes_zero_shot_singleton(self):
        class StubPipeline:
            calls = 0

            def __init__(self):
                self.quantiles = model.CHRONOS_QUANTILE_LEVELS

            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                cls.calls += 1
                return cls()

        chronos = types.SimpleNamespace(Chronos2Pipeline=StubPipeline)
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.dict(sys.modules, {"chronos": chronos}),
            mock.patch.dict(
                "os.environ",
                {"ADAPTER_DIR": str(Path(directory) / "adapters"),
                 "REQUIRE_ADAPTER": "true", "DEVICE": "cpu"},
                clear=False,
            ),
            mock.patch.object(model, "_PIPE", None),
            mock.patch.object(model, "_PIPE_METADATA", None),
        ):
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "REQUIRE_ADAPTER"):
                    model.pipe()
                self.assertIsNone(model._PIPE)
                self.assertIsNone(model._PIPE_METADATA)

        # Fail closed before allocating/loading the large base model.
        self.assertEqual(StubPipeline.calls, 0)

    def test_valid_adapter_uses_official_chronos_loader_and_manifest(self):
        class StubPipeline:
            calls = []

            def __init__(self):
                self.quantiles = model.CHRONOS_QUANTILE_LEVELS

            @classmethod
            def from_pretrained(cls, path, **kwargs):
                cls.calls.append((path, kwargs))
                return cls()

        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapters" / "current"
            self._write_adapter(adapter)
            chronos = types.SimpleNamespace(Chronos2Pipeline=StubPipeline)
            with (
                mock.patch.dict(sys.modules, {"chronos": chronos}),
                mock.patch.dict(
                    "os.environ",
                    {
                        "ADAPTER_DIR": str(adapter.parent),
                        "REQUIRE_ADAPTER": "true",
                        "PROD_CONFIG": "G1_C1",
                        "DEVICE": "cpu",
                    },
                    clear=False,
                ),
                mock.patch.object(model, "_PIPE", None),
                mock.patch.object(model, "_PIPE_METADATA", None),
            ):
                loaded = model.pipe()
                metadata = model.active_model_metadata()

        self.assertIsInstance(loaded, StubPipeline)
        self.assertEqual(StubPipeline.calls, [(str(adapter), {"device_map": "cpu"})])
        self.assertEqual(metadata["config"], "G1_C1")
        self.assertEqual(metadata["training_mode"], "LoRA")

    def test_adapter_for_other_worker_is_rejected_before_model_load(self):
        class StubPipeline:
            calls = 0

            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                cls.calls += 1
                raise AssertionError("loader must not be called")

        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapters" / "current"
            self._write_adapter(adapter, config_name="G3_C5")
            chronos = types.SimpleNamespace(Chronos2Pipeline=StubPipeline)
            with (
                mock.patch.dict(sys.modules, {"chronos": chronos}),
                mock.patch.dict(
                    "os.environ",
                    {
                        "ADAPTER_DIR": str(adapter.parent),
                        "REQUIRE_ADAPTER": "true",
                        "PROD_CONFIG": "G1_C1",
                        "DEVICE": "cpu",
                    },
                    clear=False,
                ),
                mock.patch.object(model, "_PIPE", None),
                mock.patch.object(model, "_PIPE_METADATA", None),
                self.assertRaisesRegex(RuntimeError, "Worker erwartet"),
            ):
                model.pipe()

        self.assertEqual(StubPipeline.calls, 0)

    def test_pipeline_without_native_quantile_contract_is_rejected(self):
        class StubPipeline:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

        chronos = types.SimpleNamespace(Chronos2Pipeline=StubPipeline)
        with (
            mock.patch.dict(sys.modules, {"chronos": chronos}),
            mock.patch.dict(
                "os.environ",
                {"REQUIRE_ADAPTER": "false", "DEVICE": "cpu", "ADAPTER_DIR": ""},
                clear=False,
            ),
            mock.patch.object(model, "_PIPE", None),
            mock.patch.object(model, "_PIPE_METADATA", None),
            self.assertRaisesRegex(RuntimeError, "Quantillevels"),
        ):
            model.pipe()


if __name__ == "__main__":
    unittest.main()
