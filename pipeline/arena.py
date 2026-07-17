"""Safe Energy-Arena submission for one explicitly named Chronos-2 worker.

Each worker owns a separate API key, persistent dedup state and status file.
Forecasts are only attempted close to the published deadline.  A format is
marked submitted only after its own successful HTTP response, so partial
failures can be retried without duplicating already accepted formats.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from experiments import configs, data, model
from loaders import config as lc
from . import forecast
from .runtime_status import update_status


lc.load_dotenv()
ARENA_BASE = os.environ.get("ARENA_BASE", "https://api.energy-arena.org")
ARENA_API_KEY = os.environ.get("ARENA_API_KEY", "")
SUBMIT_ENABLED = os.environ.get("SUBMIT_ENABLED", "false").lower() == "true"
SUBMIT_FORMATS = tuple(os.environ.get("SUBMIT_FORMATS", "point quantile ensemble").split())
DELU_CHALLENGES = {"point": "20", "quantile": "22", "ensemble": "24"}
PROD_CONFIG = os.environ.get("PROD_CONFIG", "").strip()
MODEL_LABEL = os.environ.get("MODEL_LABEL", f"Chronos-2 {PROD_CONFIG}").strip()
REPO_COMMIT = os.environ.get("REPO_COMMIT", "unknown").strip()
STATE_DIR = Path(os.environ.get("APP_BASE", "/app/state"))
STATE = Path(os.environ.get("PIPELINE_STATE_PATH", STATE_DIR / "pipeline_state.json"))
STATUS = Path(os.environ.get("STATUS_PATH", STATE_DIR / "status.json"))
SUBMIT_LEAD_MINUTES = int(os.environ.get("SUBMIT_LEAD_MINUTES", "60"))


def _state() -> dict:
    try:
        value = json.loads(STATE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(value, dict):
        raise RuntimeError("Pipeline-State ist kein JSON-Objekt")
    return value


def _write_state(value: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=STATE.parent,
            prefix=f".{STATE.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, STATE)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _headers():
    return {"X-API-Key": ARENA_API_KEY} if ARENA_API_KEY else None


def _now_utc() -> pd.Timestamp:
    """Aktuelle UTC-Zeit als eigener Test-/Clock-Seam."""
    return pd.Timestamp.now(tz="UTC")


def open_challenges() -> dict:
    response = requests.get(
        f"{ARENA_BASE}/api/v1/challenges/open", headers=_headers(), timeout=20
    )
    response.raise_for_status()
    return {
        str(challenge["challenge_id"]): challenge
        for challenge in response.json().get("active_challenges", [])
    }


def _values(fmt: str, zquantiles: dict) -> list:
    if fmt == "quantile":
        values = forecast.delu_quantiles(zquantiles)
        return [values[:, t].round(3).tolist() for t in range(values.shape[1])]
    if fmt == "point":
        values = forecast.delu_point(zquantiles)
        return [round(float(value), 3) for value in values]
    if fmt == "ensemble":
        values = forecast.delu_ensemble(zquantiles, n=100)
        return [values[:, t].round(3).tolist() for t in range(values.shape[1])]
    raise ValueError(f"unbekanntes Submission-Format {fmt!r}")


def _post(challenge_id: str, target_start: str, values: list):
    if not (SUBMIT_ENABLED and ARENA_API_KEY):
        width = len(values[0]) if values and isinstance(values[0], list) else 1
        print(
            f"[DRY] {MODEL_LABEL} challenge {challenge_id}: {len(values)}x{width}",
            file=sys.stderr,
            flush=True,
        )
        return 0, "dry-run"
    response = requests.post(
        f"{ARENA_BASE}/api/v1/submissions",
        headers={"X-API-Key": ARENA_API_KEY},
        json={"challenge_id": challenge_id, "target_start": target_start, "values": values},
        timeout=30,
    )
    return response.status_code, response.text[:300]


def _latest(series) -> str | None:
    clean = series.dropna()
    return clean.index.max().isoformat() if not clean.empty else None


def _freshness(config_name: str) -> dict:
    """Non-secret freshness metadata tailored to the active model schema."""
    data.clear_caches()
    cfg = configs.get(config_name)
    now = _now_utc()

    def item(timestamp: str | None) -> dict:
        if timestamp is None:
            return {"latest": None, "status": "fehlt"}
        ts = pd.Timestamp(timestamp)
        return {
            "latest": timestamp,
            "age_hours": round((now - ts).total_seconds() / 3600, 2),
        }

    def common_latest(timestamps) -> str | None:
        values = list(timestamps)
        # Fuer ein gemeinsames Modell ist eine Quelle nicht frisch, sobald auch
        # nur ein benoetigtes Gebiet keine verwertbare Reihe besitzt.
        if not values or any(value is None for value in values):
            return None
        return min(values)

    result = {
        "load": item(common_latest(_latest(data.target(a)) for a in cfg.areas))
    }
    if config_name == "G3_C5":
        weather_latest = [_latest(data.weather(a, cfg.weather).iloc[:, 0]) for a in cfg.areas]
        result["weather"] = item(common_latest(weather_latest))
        for name in ("price", "solar", "wind"):
            areas = cfg.areas if name == "price" else [a for a in cfg.areas if a != lc.LU]
            timestamps = []
            for area in areas:
                series = data.aux(area, name)
                if series is not None:
                    timestamps.append(_latest(series))
                else:
                    timestamps.append(None)
            result[name] = item(common_latest(timestamps))
    return result


def _status_base() -> dict:
    metadata = model.active_model_metadata()
    return {
        "repo_commit": REPO_COMMIT,
        "worker_kind": f"forecast-{PROD_CONFIG}",
        "model": {
            "label": MODEL_LABEL,
            "base_model": metadata.get("base_model", model.MODEL),
            "parameter_count": metadata.get(
                "parameter_count", model.MODEL_PARAMETER_COUNT
            ),
            "training_mode": metadata.get("training_mode", "unknown"),
            "adapter_digest": metadata.get("adapter_digest"),
        },
        "training_mode": metadata.get("training_mode", "unknown"),
        "adapter_digest": metadata.get("adapter_digest"),
    }


def submit(target_start: str, formats, challenges: dict) -> dict:
    """Create one forecast and submit only the requested, still-pending formats."""
    zquantiles, _ = forecast.zone_quantiles(PROD_CONFIG, delivery_date=target_start)
    results = {}
    for fmt in formats:
        challenge_id = DELU_CHALLENGES[fmt]
        if challenge_id not in challenges:
            results[fmt] = {
                "challenge_id": challenge_id,
                "status": "not-open",
                "target_start": target_start,
            }
            continue
        code, response = _post(challenge_id, target_start, _values(fmt, zquantiles))
        ok = code == 0 or 200 <= code < 300
        results[fmt] = {
            "challenge_id": challenge_id,
            "status_code": code,
            "ok": ok,
            "target_start": target_start,
        }
        print(
            f"[{MODEL_LABEL}] {fmt} (#{challenge_id}) target_start={target_start} -> {code}",
            file=sys.stderr,
            flush=True,
        )
        # Response-Bodies koennen fremde Identifikatoren enthalten und werden
        # deshalb weder in Logs noch im Status persistiert.
    return results


def run_if_due() -> str:
    if PROD_CONFIG not in {"G1_C1", "G3_C5"}:
        raise RuntimeError(f"Live-Worker verweigert PROD_CONFIG={PROD_CONFIG!r}")
    if not 0 < SUBMIT_LEAD_MINUTES <= 24 * 60:
        raise RuntimeError("SUBMIT_LEAD_MINUTES muss zwischen 1 und 1440 liegen")
    try:
        challenges = open_challenges()
    except Exception as exc:
        update_status(
            STATUS, repo_commit=REPO_COMMIT, worker_kind=f"forecast-{PROD_CONFIG}",
            jobs={"errors": 1, "last_error": f"Arena nicht erreichbar ({type(exc).__name__})"},
        )
        return f"arena-unreachable: {type(exc).__name__}"

    quantile_id = DELU_CHALLENGES["quantile"]
    if quantile_id not in challenges:
        return "no-open-load-challenge"
    challenge = challenges[quantile_id]
    target = challenge["next_target_start"]
    now = _now_utc()
    opens = pd.Timestamp(challenge["submission_window_opens_at"]).tz_convert("UTC")
    deadline = pd.Timestamp(challenge["next_submission_deadline"]).tz_convert("UTC")
    earliest = deadline - pd.Timedelta(minutes=SUBMIT_LEAD_MINUTES)
    if not (opens <= now < deadline):
        return f"not-in-window (opens {opens}, deadline {deadline})"
    if now < earliest:
        return f"waiting-for-fresh-data (earliest {earliest}, deadline {deadline})"

    mode = "live" if (SUBMIT_ENABLED and ARENA_API_KEY) else "dry"
    state = _state()
    submitted = dict(state.get(f"submitted_{mode}_by_format", {}))
    requested = [fmt for fmt in SUBMIT_FORMATS if fmt in DELU_CHALLENGES]
    pending = [fmt for fmt in requested if submitted.get(fmt) != target]
    if not pending:
        return f"already-submitted ({mode}) {target}"

    started = datetime.now(timezone.utc).isoformat()
    try:
        results = submit(target, pending, challenges)
        for fmt, result in results.items():
            if result.get("ok"):
                submitted[fmt] = target
        state[f"submitted_{mode}_by_format"] = submitted
        state["last_target_start"] = target
        _write_state(state)
        failed_formats = [fmt for fmt, result in results.items() if not result.get("ok")]
        status_fields = _status_base()
        status_fields.update(
            target_start=target,
            submissions=results,
            data_freshness=_freshness(PROD_CONFIG),
            jobs={
                "last_run_started": started,
                "last_run_finished": datetime.now(timezone.utc).isoformat(),
                "errors": len(failed_formats),
                "last_error": (
                    "Submission fehlgeschlagen: " + ", ".join(failed_formats)
                    if failed_formats else None
                ),
            },
        )
        update_status(STATUS, **status_fields)
    except Exception as exc:
        update_status(
            STATUS, repo_commit=REPO_COMMIT, worker_kind=f"forecast-{PROD_CONFIG}",
            target_start=target,
            jobs={"last_run_started": started, "errors": 1,
                  "last_error": type(exc).__name__},
        )
        raise

    accepted = [fmt for fmt, result in results.items() if result.get("ok")]
    failed = [fmt for fmt, result in results.items() if not result.get("ok")]
    return f"{mode} target={target} accepted={accepted} failed={failed}"


if __name__ == "__main__":
    print(run_if_due())
