"""Reiner Forecast-/Submission-Scheduler fuer genau G1_C1 oder G3_C5.

Der Prozess startet keine Loader und schreibt nicht in ``data/``. Der erste
Forecast wird absichtlich erst beim konfigurierten Cron-Termin erzeugt. Eine
containeruebergreifende POSIX-Sperre serialisiert alle GPU-Laeufe.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from experiments import configs
from loaders import config as lc
from .locking import data_read_lock, gpu_lock
from .runtime_status import RuntimeStatusError, update_status

ALLOWED_CONFIGS = frozenset({"G1_C1", "G3_C5"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(**updates) -> None:
    """Geheimnisfreien Worker-Status fuer den Summary-Dienst aktualisieren."""
    base = Path(os.environ.get("APP_BASE", "/app/state"))
    path = Path(os.environ.get("STATUS_PATH", base / "status.json"))
    fields = {
        "worker_kind": os.environ.get("WORKER_KIND", os.environ.get("PROD_CONFIG", "")),
        "model_label": os.environ.get("MODEL_LABEL", ""),
        "prod_config": os.environ.get("PROD_CONFIG", ""),
        "base_model": os.environ.get("MODEL_ID", "amazon/chronos-2"),
        "parameter_count": 120_000_000,
        "training_mode": os.environ.get("TRAINING_MODE", "zero-shot"),
        **updates,
    }
    repo_commit = os.environ.get("REPO_COMMIT", "").strip()
    adapter_digest = os.environ.get("ADAPTER_DIGEST", "").strip()
    if repo_commit:
        fields["repo_commit"] = repo_commit
    if adapter_digest:
        fields["adapter_digest"] = adapter_digest
    try:
        update_status(path, **fields)
    except (OSError, RuntimeStatusError) as exc:
        print(f"[WARN] Worker-Status nicht schreibbar ({type(exc).__name__})",
              file=sys.stderr, flush=True)


def _record_job(*, error: bool) -> None:
    """Gleitende, geheimnisfreie 24-h-Zaehler fuer die Tagesmail pflegen."""
    base = Path(os.environ.get("APP_BASE", "/app/state"))
    path = Path(os.environ.get("STATUS_PATH", base / "status.json"))
    try:
        current = update_status(path)
        cutoff = datetime.now(timezone.utc).timestamp() - 24 * 3600
        events = []
        for event in current.get("job_events", []):
            if not isinstance(event, dict):
                continue
            try:
                stamp = datetime.fromisoformat(str(event["at"]).replace("Z", "+00:00"))
            except (KeyError, TypeError, ValueError):
                continue
            if stamp.timestamp() >= cutoff:
                events.append({"at": stamp.isoformat(), "error": bool(event.get("error"))})
        events.append({"at": _utc_now(), "error": error})
        _write_status(job_events=events,
                      jobs={"last_24h": len(events),
                            "errors": sum(bool(event["error"]) for event in events)})
    except (OSError, RuntimeStatusError) as exc:
        print(f"[WARN] Job-Zaehler nicht schreibbar ({type(exc).__name__})",
              file=sys.stderr, flush=True)


def _last_target_start() -> str | None:
    path = Path(os.environ.get("APP_BASE", "/app/state")) / "pipeline_state.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(state, dict):
        return None
    return (state.get("last_target_start") or state.get("last_submitted_live")
            or state.get("last_submitted_dry"))


def _adapter_manifest_metadata() -> dict:
    manifest = Path(os.environ.get("ADAPTER_DIR", "/app/adapters")) / "current" \
        / "production_manifest.json"
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError, UnicodeError):
        # Einen Digest aus einem frueheren, inzwischen entfernten Adapter nicht
        # ueber den rekursiven Status-Merge konservieren.
        return {"adapter_digest": None}
    if not isinstance(value, dict):
        return {"adapter_digest": None}
    allowed = (
        "training_mode", "adapter_digest", "base_model", "parameter_count",
        "repo_commit",
    )
    return {name: value[name] for name in allowed if value.get(name)}


def _validate_role() -> tuple[str, str, str]:
    config_name = os.environ.get("PROD_CONFIG", "").strip()
    if config_name not in ALLOWED_CONFIGS:
        raise SystemExit(
            f"Worker verweigert PROD_CONFIG={config_name!r}; erlaubt: {sorted(ALLOWED_CONFIGS)}"
        )
    cfg = configs.get(config_name)
    if config_name == "G1_C1":
        # Explizite Invariante: Dieser Worker darf keinerlei Wetter-/Aux-Daten
        # benoetigen. Ein versehentlich veraenderter Registry-Eintrag stoppt ihn.
        if (cfg.weather is not None or cfg.aux_target or cfg.aux_known or cfg.aux_past
                or cfg.calendar or cfg.gran != "whole"):
            raise SystemExit("G1_C1 ist nicht mehr strikt univariat; Start abgebrochen")
    label = os.environ.get("MODEL_LABEL", f"Chronos-2 {config_name}").strip()
    minute = os.environ.get("SUBMIT_MINUTE", "20").strip()
    if not minute:
        raise SystemExit("SUBMIT_MINUTE darf nicht leer sein")
    return config_name, label, minute


def _run_submission(label: str) -> None:
    started = time.time()
    _write_status(job_status="running", last_job_started_at=_utc_now())
    try:
        print(f"[{label}] warte auf gemeinsamen GPU-Lock", file=sys.stderr, flush=True)
        # Erst GPU, dann Daten: Ein auf die GPU wartender Worker blockiert keinen
        # anstehenden Data-Refresh mit einem unnoetig lange gehaltenen Leselock.
        with gpu_lock(), data_read_lock():
            print(f"[{label}] GPU- und Daten-Leselock erhalten", file=sys.stderr, flush=True)
            # Spaeter Import: kein Chronos-/Arena-Import und insbesondere kein
            # Forecast beim Container-Startup.
            from . import arena
            result = arena.run_if_due()
        print(f"[{label}] GPU- und Daten-Leselock freigegeben", file=sys.stderr, flush=True)
        fields = {
            "job_status": "ok",
            "last_job_finished_at": _utc_now(),
            "last_result": str(result),
            "last_error_type": None,
            "last_duration_seconds": round(time.time() - started, 3),
        }
        # arena.run_if_due schreibt Submissionresultate, Datenfrische und die
        # tatsaechlich aktive Adapter-Metadaten bereits atomar in denselben Status.
        target = _last_target_start()
        if target:
            fields["target_start"] = target
        _write_status(**fields)
        _record_job(error=False)
        print(f"[ok] {label}: {result} ({time.time() - started:.0f}s)",
              file=sys.stderr, flush=True)
    except Exception as exc:
        _write_status(job_status="error", last_job_finished_at=_utc_now(),
                      last_error_type=type(exc).__name__,
                      last_duration_seconds=round(time.time() - started, 3))
        _record_job(error=True)
        print(f"[FEHLER] {label}:\n{traceback.format_exc()}",
              file=sys.stderr, flush=True)


def _run_finetune(label: str) -> None:
    started = time.time()
    _write_status(finetune_status="running", last_finetune_started_at=_utc_now())
    try:
        # pipeline.finetune haelt selbst denselben gpu->data-Lockvertrag.
        from . import finetune
        manifest = finetune.run()
        metadata = {name: manifest[name] for name in
                    (
                        "training_mode", "adapter_digest", "base_model",
                        "parameter_count", "repo_commit",
                    )
                    if manifest.get(name)}
        _write_status(finetune_status="ok", last_finetune_finished_at=_utc_now(),
                      last_finetune_error_type=None,
                      last_finetune_duration_seconds=round(time.time() - started, 3),
                      **metadata)
        _record_job(error=False)
        print(f"[ok] {label} Fine-Tuning ({time.time() - started:.0f}s)",
              file=sys.stderr, flush=True)
    except Exception as exc:
        _write_status(finetune_status="error", last_finetune_finished_at=_utc_now(),
                      last_finetune_error_type=type(exc).__name__,
                      last_finetune_duration_seconds=round(time.time() - started, 3))
        _record_job(error=True)
        print(f"[FEHLER] {label} Fine-Tuning:\n{traceback.format_exc()}",
              file=sys.stderr, flush=True)


def main() -> None:
    lc.load_dotenv()
    config_name, label, minute = _validate_role()
    finetune_enabled = os.environ.get("FINETUNE_ENABLED", "false").lower() == "true"
    finetune_months = os.environ.get("FINETUNE_MONTHS", "1,4,7,10").strip()
    finetune_day = os.environ.get("FINETUNE_DAY", "1").strip()
    finetune_hour = os.environ.get("FINETUNE_HOUR_UTC", "7").strip()
    finetune_minute = os.environ.get("FINETUNE_MINUTE_UTC", "10").strip()
    state_dir = Path(os.environ.get("APP_BASE", "/app/state"))
    state_dir.mkdir(parents=True, exist_ok=True)
    data_mode = "read-only mount (durch Compose erzwungen)"
    print(f"### Forecast-Worker bereit | {label} | config={config_name}"
          f" | submit=UTC *:{minute} | finetune="
          f"{'Monate ' + finetune_months + ', Tag ' + finetune_day + ', UTC '
             + finetune_hour + ':' + finetune_minute if finetune_enabled else 'aus'}"
          f" | data={data_mode}",
          file=sys.stderr, flush=True)
    _write_status(job_status="scheduled", scheduler_timezone="UTC",
                  submit_minute=minute, finetune_enabled=finetune_enabled,
                  finetune_months=finetune_months if finetune_enabled else None,
                  finetune_day=finetune_day if finetune_enabled else None,
                  finetune_hour_utc=finetune_hour if finetune_enabled else None,
                  finetune_minute_utc=finetune_minute if finetune_enabled else None,
                  started_at=_utc_now(), **_adapter_manifest_metadata())

    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(lambda: _run_submission(label), CronTrigger(minute=minute),
                      id=f"submit-{config_name.lower().replace('_', '-')}",
                      max_instances=1, coalesce=True, misfire_grace_time=600)
    if finetune_enabled:
        scheduler.add_job(lambda: _run_finetune(label),
                          CronTrigger(month=finetune_months, day=finetune_day,
                                      hour=finetune_hour, minute=finetune_minute),
                          id=f"finetune-{config_name.lower().replace('_', '-')}",
                          max_instances=1, coalesce=True, misfire_grace_time=3600)
    print("Kein Startup-Submit/-Fine-Tuning; erster Lauf erst am Scheduler-Termin.",
          file=sys.stderr, flush=True)
    scheduler.start()


if __name__ == "__main__":
    main()
