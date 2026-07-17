"""Standalone production LoRA fit for ``G1_C1`` or ``G3_C5``.

The fit is written to a staging directory and only promoted after the expected
rank/alpha, files and schema have been validated.  Existing adapters remain
available as timestamped rollback directories.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from experiments import configs, data, model
from .locking import data_read_lock, gpu_lock
from .runtime_status import RuntimeStatusError, update_status


ALLOWED_CONFIGS = frozenset({"G1_C1", "G3_C5"})
HORIZON = 155


def _fit_task(config_name: str, train_end: pd.Timestamp, train_start: pd.Timestamp):
    cfg = configs.get(config_name)
    frames = {area: model.area_frame(cfg, area) for area in cfg.areas}
    if list(frames) != list(cfg.areas):
        raise RuntimeError(f"unvollstaendiges Trainingsschema: {list(frames)}")
    if cfg.gran == "joint":
        task = model.build_joint_fit_task(frames, train_end, train_start=train_start)
        target_count = sum(len(frame[1]) for frame in frames.values())
    else:
        if len(frames) != 1:
            raise RuntimeError(f"{config_name}: genau ein Whole-Area-Frame erwartet")
        frame = next(iter(frames.values()))
        task = model.build_fit_task(
            frame[0], train_end, frame[1], frame[2], frame[3], train_start=train_start
        )
        target_count = len(frame[1])
    if task is None:
        raise RuntimeError("Trainingsfenster ist zu kurz oder unvollstaendig")
    return cfg, task, target_count


def _latest_common_target(config_name: str) -> pd.Timestamp:
    cfg = configs.get(config_name)
    latest = []
    for area in cfg.areas:
        series = data.target(area).dropna()
        if series.empty:
            raise RuntimeError(f"keine Istlast fuer {area}")
        latest.append(series.index.max())
    return min(latest).floor("15min")


def _validate_checkpoint(path: Path, config_name: str) -> dict:
    metadata = model.validate_lora_checkpoint(path)
    return {
        "config": config_name,
        **metadata,
        "rank": model.LORA_R,
        "alpha": model.LORA_ALPHA,
    }


def _fsync_directory(path: Path) -> None:
    """Persistiere Verzeichnis-Metadaten nach Rename/Symlink-Swap."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_manifest(path: Path, manifest: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _promote_candidate(adapter_root: Path, candidate: Path, config_name: str,
                       stamp: str) -> Path:
    """Mache einen validierten Checkpoint ueber ``current`` atomar sichtbar.

    Checkpoints liegen unveraenderlich in ``release-*``-Verzeichnissen.
    ``current`` ist ein relativer Symlink und wird mit ``os.replace`` in einem
    Schritt gewechselt; der vorherige Release bleibt als ``rollback-*`` erhalten.
    """
    current = adapter_root / "current"
    release = adapter_root / f"release-{config_name}-{stamp}"
    temporary_link = adapter_root / f".current-{stamp}.tmp"
    rollback = adapter_root / f"rollback-{stamp}"
    if release.exists() or os.path.lexists(temporary_link) or os.path.lexists(rollback):
        raise RuntimeError(f"Adapter-Ziel fuer Zeitstempel {stamp} existiert bereits")

    previous_release: str | None = None
    legacy_backup: Path | None = None
    if current.is_symlink():
        raw_target = os.readlink(current)
        resolved = (current.parent / raw_target).resolve()
        if resolved.parent != adapter_root.resolve() or not resolved.is_dir():
            raise RuntimeError("current verweist nicht auf einen gueltigen lokalen Adapter-Release")
        previous_release = resolved.name
    elif current.exists():
        if not current.is_dir():
            raise RuntimeError("current ist weder Adapter-Verzeichnis noch Symlink")
        # Einmalige Migration des frueheren Directory-Layouts. Der gemeinsame
        # GPU-Lock verhindert dabei parallele Inferenz.
        legacy_backup = rollback
    elif os.path.lexists(current):
        raise RuntimeError("current ist ein defekter Symlink")

    candidate.rename(release)
    os.symlink(release.name, temporary_link)
    try:
        if legacy_backup is not None:
            current.rename(legacy_backup)
        os.replace(temporary_link, current)
        _fsync_directory(adapter_root)
    except Exception:
        if os.path.lexists(temporary_link):
            temporary_link.unlink()
        if legacy_backup is not None and legacy_backup.exists() and not current.exists():
            legacy_backup.rename(current)
        raise

    if previous_release is not None:
        os.symlink(previous_release, rollback)
        _fsync_directory(adapter_root)
    return current


def _status_path() -> Path:
    base = Path(os.environ.get("APP_BASE", "/app/state"))
    return Path(os.environ.get("STATUS_PATH", base / "status.json"))


def _update_runtime_status(**fields) -> None:
    """Statusfehler duerfen einen gueltigen Adapter nicht verhindern."""
    try:
        update_status(_status_path(), **fields)
    except (OSError, RuntimeStatusError) as exc:
        print(
            f"[WARN] Fine-Tuning-Status nicht schreibbar ({type(exc).__name__})",
            file=sys.stderr,
            flush=True,
        )


def _fit_and_promote(config_name: str) -> dict:
    adapter_root = Path(os.environ.get("ADAPTER_DIR", "/app/adapters"))
    adapter_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    staging = adapter_root / f".staging-{config_name}-{stamp}"
    steps = int(os.environ.get("FT_STEPS", "300"))
    batch = int(os.environ.get("FT_BATCH_SIZE", "8"))
    lr = float(os.environ.get("FT_LR", "1e-4"))
    ctx_steps = int(os.environ.get("FT_CONTEXT", str(63 * 96)))
    months = int(os.environ.get("FT_TRAIN_WINDOW_MONTHS", "12"))

    with gpu_lock(), data_read_lock():
        data.clear_caches()
        train_end = _latest_common_target(config_name)
        train_start = train_end - pd.DateOffset(months=months)
        cfg, task, target_count = _fit_task(config_name, train_end, train_start)

        # Never train a new adapter on top of the currently deployed adapter.
        saved_env = {name: os.environ.get(name) for name in
                     ("ADAPTER_DIR", "ADAPTER_PATH", "REQUIRE_ADAPTER")}
        os.environ.pop("ADAPTER_DIR", None)
        os.environ.pop("ADAPTER_PATH", None)
        os.environ["REQUIRE_ADAPTER"] = "false"
        model.reset_pipe()
        try:
            model.fit_lora(
                [task], HORIZON, ctx_steps, num_steps=steps, lr=lr,
                batch_size=batch, out_dir=staging,
            )
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        finally:
            model.reset_pipe()
            for name, value in saved_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

        try:
            candidate = staging / "finetuned-ckpt"
            manifest = _validate_checkpoint(candidate, config_name)
            manifest.update(
                repo_commit=os.environ.get("REPO_COMMIT", "unknown"),
                trained_at=datetime.now(timezone.utc).isoformat(),
                train_start=str(train_start),
                train_end=str(train_end),
                train_window_months=months,
                num_steps=steps,
                batch_size=batch,
                learning_rate=lr,
                context_steps=ctx_steps,
                horizon=HORIZON,
                granularity=cfg.gran,
                areas=list(cfg.areas),
                target_count=target_count,
            )
            _write_manifest(candidate / "production_manifest.json", manifest)
            # Das Manifest ist Teil des Produktionsvertrags und wird vor dem
            # Swap gegen Adapter und Worker-Config geprueft.
            model.validate_lora_checkpoint(
                candidate, expected_config=config_name, require_manifest=True
            )
            _promote_candidate(adapter_root, candidate, config_name, stamp)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    print(
        f"LoRA bereit: config={config_name} mode=LoRA digest={manifest['adapter_digest']} "
        f"train_end={train_end}",
        file=sys.stderr,
        flush=True,
    )
    return manifest


def run() -> dict:
    """Adapter trainieren und den geheimnisfreien Status atomar aktualisieren."""
    config_name = os.environ.get("PROD_CONFIG", "").strip()
    if config_name not in ALLOWED_CONFIGS:
        raise SystemExit(f"PROD_CONFIG muss eine von {sorted(ALLOWED_CONFIGS)} sein")
    repo_commit = os.environ.get("REPO_COMMIT", "unknown").strip() or "unknown"
    started_at = datetime.now(timezone.utc).isoformat()
    common = {
        "repo_commit": repo_commit,
        "worker_kind": f"forecast-{config_name}",
    }
    _update_runtime_status(
        **common,
        fine_tune={"status": "running", "started_at": started_at, "error_type": None},
    )
    try:
        manifest = _fit_and_promote(config_name)
    except Exception as exc:
        _update_runtime_status(
            **common,
            fine_tune={
                "status": "error",
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                # Nur der Typ wird persistiert; Exception-Texte koennen fremde
                # Pfade, URLs oder Zugangsdaten enthalten.
                "error_type": type(exc).__name__,
            },
        )
        raise

    model_status = {
        "base_model": manifest.get("base_model", model.MODEL),
        "parameter_count": manifest.get(
            "parameter_count", model.MODEL_PARAMETER_COUNT
        ),
        "training_mode": manifest.get("training_mode", "LoRA"),
        "adapter_digest": manifest.get("adapter_digest"),
    }
    _update_runtime_status(
        **common,
        model=model_status,
        training_mode=model_status["training_mode"],
        adapter_digest=model_status["adapter_digest"],
        fine_tune={
            "status": "ok",
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error_type": None,
            "train_start": manifest.get("train_start"),
            "train_end": manifest.get("train_end"),
            "num_steps": manifest.get("num_steps"),
        },
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
