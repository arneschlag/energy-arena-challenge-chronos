"""Kombinierte, geheimnisfreie Tageszusammenfassung der Live-Modelle.

Die beiden Container schreiben ihren Status in getrennte JSON-Dateien.  Dieses
Modul liest nur explizit erlaubte Felder daraus; unbekannte Felder (insbesondere
API-Schluessel) werden weder formatiert noch geloggt.

Empfohlenes Statusformat (alle Felder sind optional)::

    {
      "repo_commit": "abc123...",
      "training_mode": "LoRA",
      "adapter_digest": "sha256:...",
      "target_start": "2026-07-18T00:00:00+02:00",
      "submissions": {
        "point": {"challenge_id": 20, "status_code": 200},
        "quantile": {"challenge_id": 22, "status_code": 200}
      },
      "data_freshness": {
        "load": {"latest": "2026-07-16T09:30:00Z", "age_hours": 0.7}
      },
      "jobs": {"last_24h": 25, "errors": 0}
    }

Fuer einen sanften Uebergang werden einige alte Feldnamen ebenfalls erkannt.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo


BERLIN = ZoneInfo("Europe/Berlin")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    heading: str
    freshness: tuple[tuple[str, tuple[str, ...]], ...]


G1 = ModelSpec(
    key="G1",
    heading="G1_C1 — Chronos 2 univariat",
    freshness=(("Last", ("load", "actual", "loads")),),
)
G3 = ModelSpec(
    key="G3",
    heading="G3_C5 — Chronos 2 multivariat (Group Attention, C5)",
    freshness=(
        ("Last", ("load", "actual", "loads")),
        ("Wetter", ("weather", "forecast_weather")),
        ("Preis", ("price", "aux_price")),
        ("Solar", ("solar", "aux_solar")),
        ("Wind", ("wind", "aux_wind")),
    ),
)


@dataclass(frozen=True)
class LoadedState:
    path: Path | None
    data: Mapping[str, Any]
    issue: str | None = None


@dataclass(frozen=True)
class DailySummary:
    subject: str
    body: str


def _env_first(env: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = env.get(name)
        if value and value.strip():
            return value.strip()
    return None


def state_path(spec: ModelSpec, env: Mapping[str, str] | None = None) -> Path:
    """Liefere den getrennten Statuspfad eines Modells.

    ``*_STATUS_PATH`` ist der kanonische Name, ``*_STATE_PATH`` bleibt als
    Alias verwendbar.  Die Defaults passen zu read-only State-Volumes in einem
    dritten Summary-Container.
    """
    env = os.environ if env is None else env
    raw = _env_first(env, f"{spec.key}_STATUS_PATH", f"{spec.key}_STATE_PATH")
    return Path(raw or f"/state/{spec.key.lower()}/status.json")


def load_state(path: str | os.PathLike[str]) -> LoadedState:
    """JSON-State lesen, ohne bei fehlenden oder defekten Dateien abzubrechen."""
    original = Path(path)
    candidate = original
    if candidate.is_dir():
        for name in ("status.json", "pipeline_status.json", "pipeline_state.json"):
            child = candidate / name
            if child.is_file():
                candidate = child
                break
        else:
            candidate = candidate / "status.json"
    try:
        raw = candidate.read_text(encoding="utf-8")
    except FileNotFoundError:
        return LoadedState(candidate, {}, "Statusdatei fehlt")
    except OSError:
        return LoadedState(candidate, {}, "Statusdatei nicht lesbar")
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeError):
        return LoadedState(candidate, {}, "Statusdatei enthält ungültiges JSON")
    if not isinstance(data, dict):
        return LoadedState(candidate, {}, "Statusdatei enthält kein JSON-Objekt")
    return LoadedState(candidate, data)


def _nested(data: Mapping[str, Any], *paths: str) -> Any:
    for path in paths:
        value: Any = data
        for part in path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                break
            value = value[part]
        else:
            if value is not None and value != "":
                return value
    return None


def _safe_text(value: Any, *, maximum: int = 160) -> str | None:
    """Nur kurze skalare Werte in die Mail uebernehmen."""
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return text[:maximum] if text else None


def _first_present(mapping: Mapping[str, Any], *names: str) -> Any:
    """Ersten vorhandenen Wert liefern; numerische Null ist dabei gueltig."""
    for name in names:
        if name in mapping and mapping[name] is not None and mapping[name] != "":
            return mapping[name]
    return None


def _metadata_value(
    state: Mapping[str, Any],
    spec: ModelSpec,
    env: Mapping[str, str],
    state_paths: tuple[str, ...],
    env_suffixes: tuple[str, ...],
) -> str | None:
    value = _safe_text(_nested(state, *state_paths))
    if value:
        return value
    names = [f"{spec.key}_{suffix}" for suffix in env_suffixes]
    names.extend(env_suffixes)
    return _env_first(env, *names)


def _format_mode(state: Mapping[str, Any], spec: ModelSpec, env: Mapping[str, str]) -> str:
    mode = _metadata_value(
        state,
        spec,
        env,
        (
            "training_mode",
            "adapter_mode",
            "model.training_mode",
            "model.adapter_mode",
            "model.mode",
            "mode",
        ),
        ("TRAINING_MODE", "ADAPTER_MODE", "MODEL_MODE"),
    ) or "nicht gemeldet"
    digest = _metadata_value(
        state,
        spec,
        env,
        ("adapter_digest", "model.adapter_digest", "adapter.digest", "adapter.sha256"),
        ("ADAPTER_DIGEST", "ADAPTER_SHA256"),
    )
    if digest:
        return f"{mode}; Adapter-Digest: {digest}"
    if mode.casefold().replace("_", "-") in {"zero-shot", "zeroshot"}:
        return f"{mode}; Adapter: keiner"
    return f"{mode}; Adapter-Digest: nicht gemeldet"


def _format_base_model(
    state: Mapping[str, Any], spec: ModelSpec, env: Mapping[str, str]
) -> str:
    model_id = _metadata_value(
        state,
        spec,
        env,
        ("model.base_model", "base_model", "model.id", "model_id"),
        ("BASE_MODEL", "MODEL_ID"),
    ) or "nicht gemeldet"
    raw_count = _nested(state, "model.parameter_count", "parameter_count")
    count = _number(raw_count)
    if count is None:
        return model_id
    if count >= 1_000_000:
        return f"{model_id} ({count / 1_000_000:.0f} Mio. Parameter)"
    return f"{model_id} ({count:.0f} Parameter)"


def _submission_lines(state: Mapping[str, Any]) -> list[str]:
    submissions = _nested(
        state, "submissions", "submission_results", "last_submissions", "submission.results"
    )
    rows: list[tuple[str, Any]] = []
    if isinstance(submissions, Mapping):
        rows = [(str(name), value) for name, value in submissions.items()]
    elif isinstance(submissions, list):
        for index, value in enumerate(submissions, start=1):
            if not isinstance(value, Mapping):
                continue
            name = _safe_text(
                value.get("format") or value.get("name") or value.get("target")
            ) or f"Eintrag {index}"
            rows.append((name, value))
    elif submissions is not None:
        text = _safe_text(submissions)
        if text:
            return [f"  {text}"]

    lines: list[str] = []
    for name, value in rows:
        label = name
        if isinstance(value, Mapping):
            challenge = _safe_text(_first_present(value, "challenge_id", "challenge"))
            raw_status = _first_present(value, "status_code", "http_status", "status")
            if raw_status is None and value.get("ok") is True:
                raw_status = "OK"
            elif raw_status is None and value.get("ok") is False:
                raw_status = "FEHLER"
            status = _safe_text(raw_status)
            target = _safe_text(value.get("target_start") or value.get("delivery_date"))
            if challenge:
                label += f"#{challenge}"
            detail = status or "Status nicht gemeldet"
            if target:
                detail += f"; Liefertag {target}"
        else:
            detail = _safe_text(value) or "Status nicht gemeldet"
        lines.append(f"  {label}: {detail}")
    if lines:
        return lines
    # Der Worker kann den kompakten Rueckgabestring der Arena zunaechst als
    # Uebergangsformat melden (z. B. ``point#20:200 quantile#22:200``).
    compact = _safe_text(_nested(state, "last_submission_result", "last_result"))
    return [f"  {compact}"] if compact else ["  keine Submissionresultate gemeldet"]


def _freshness_mapping(state: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _nested(state, "data_freshness", "freshness", "data.freshness", "data_age")
    return value if isinstance(value, Mapping) else {}


def _find_alias(mapping: Mapping[str, Any], aliases: tuple[str, ...]) -> Any:
    folded = {str(key).casefold(): value for key, value in mapping.items()}
    for alias in aliases:
        if alias.casefold() in folded:
            return folded[alias.casefold()]
    return None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_freshness(value: Any) -> str:
    if isinstance(value, Mapping):
        latest = _safe_text(
            _first_present(value, "latest", "available_until", "timestamp", "max_timestamp")
        )
        age = _number(value.get("age_hours") if "age_hours" in value else value.get("hours_old"))
        parts = []
        if latest:
            parts.append(f"bis {latest}")
        if age is not None:
            if age < 0:
                parts.append(f"{abs(age):.1f} h voraus")
            else:
                parts.append(f"{age:.1f} h alt")
        status = _safe_text(value.get("status"))
        if status:
            parts.append(status)
        return "; ".join(parts) or "nicht gemeldet"
    return _safe_text(value) or "nicht gemeldet"


def _job_line(state: Mapping[str, Any]) -> str:
    jobs = _nested(state, "jobs", "job_summary")
    jobs = jobs if isinstance(jobs, Mapping) else {}
    total = _safe_text(_first_present(jobs, "last_24h", "total_24h", "total"))
    if total is None:
        total = _safe_text(_nested(state, "jobs_last_24h"))
    errors_value = (
        jobs.get("errors")
        if "errors" in jobs
        else jobs.get("errors_24h", _nested(state, "job_errors"))
    )
    errors = _safe_text(errors_value)
    if total is None and errors is None:
        status = _safe_text(_nested(state, "job_status"))
        error_type = _safe_text(_nested(state, "last_error_type"))
        duration = _number(_nested(state, "last_duration_seconds"))
        parts = []
        if status:
            parts.append(f"letzter Status {status}")
        if error_type:
            parts.append(f"letzter Fehler {error_type}")
        if duration is not None:
            parts.append(f"Dauer {duration:.1f} s")
        return "; ".join(parts) or "nicht gemeldet"
    return f"{total or '?'} gelaufen, {errors or '?'} Fehler (letzte 24 h)"


def render_model_block(
    spec: ModelSpec,
    loaded: LoadedState,
    env: Mapping[str, str] | None = None,
) -> str:
    """Einen Modellblock rendern; nur allowlist-Felder verlassen den State."""
    env = os.environ if env is None else env
    state = loaded.data
    commit = _metadata_value(
        state,
        spec,
        env,
        ("repo_commit", "git_commit", "repo.commit", "build.repo_commit", "build.commit"),
        ("REPO_COMMIT", "GIT_COMMIT", "SOURCE_COMMIT"),
    ) or "nicht gemeldet"
    worker_kind = _metadata_value(
        state,
        spec,
        env,
        ("worker_kind", "model.worker_kind", "worker.kind", "service_role"),
        ("WORKER_KIND", "SERVICE_ROLE"),
    ) or "nicht gemeldet"
    target = _safe_text(
        _nested(
            state,
            "target_start",
            "delivery_date",
            "submission.target_start",
            "last_submitted_live",
        )
    ) or "nicht gemeldet"

    lines = [spec.heading]
    if loaded.issue:
        lines.append(f"Status: {loaded.issue}")
    lines.extend(
        (
            f"Repo-Commit: {commit}",
            f"Worker-Kind: {worker_kind}",
            f"Basismodell: {_format_base_model(state, spec, env)}",
            f"Modus: {_format_mode(state, spec, env)}",
            f"Liefertag: {target}",
            "Submissionresultate:",
        )
    )
    lines.extend(_submission_lines(state))
    lines.append("Datenfrische:")
    freshness = _freshness_mapping(state)
    for label, aliases in spec.freshness:
        lines.append(f"  {label}: {_format_freshness(_find_alias(freshness, aliases))}")
    lines.append(f"Jobs/Fehler: {_job_line(state)}")
    return "\n".join(lines)


def build_daily_summary(
    g1: LoadedState,
    g3: LoadedState,
    *,
    now: datetime | None = None,
    env: Mapping[str, str] | None = None,
) -> DailySummary:
    env = os.environ if env is None else env
    now = (now or datetime.now(tz=BERLIN)).astimezone(BERLIN)
    stamp = now.strftime("%Y-%m-%d %H:%M %Z")
    subject = f"Energy Arena Tages-Summary {now:%Y-%m-%d} — G1_C1 & G3_C5"
    body = "\n".join(
        (
            f"Tages-Summary {stamp}",
            "",
            render_model_block(G1, g1, env),
            "",
            render_model_block(G3, g3, env),
        )
    )
    return DailySummary(subject=subject, body=body + "\n")


def load_daily_summary(
    *,
    g1_path: str | os.PathLike[str] | None = None,
    g3_path: str | os.PathLike[str] | None = None,
    now: datetime | None = None,
    env: Mapping[str, str] | None = None,
) -> DailySummary:
    env = os.environ if env is None else env
    return build_daily_summary(
        load_state(g1_path or state_path(G1, env)),
        load_state(g3_path or state_path(G3, env)),
        now=now,
        env=env,
    )
