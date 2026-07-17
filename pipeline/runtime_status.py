"""Atomarer, nebenlaeufigkeitssicherer Writer fuer Modell-Laufzeitstatus.

Forecast-, Arena- und Fine-Tuning-Code koennen unabhaengige Teilinformationen
mit ``update_status(path, **fields)`` melden.  Updates werden rekursiv gemergt,
unter einem POSIX-File-Lock serialisiert und erst nach vollstaendigem Schreiben
per ``os.replace`` sichtbar.  Zugangsdaten gehoeren grundsaetzlich nicht in
diese Statusdateien.
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RuntimeStatusError(ValueError):
    """Statusdaten sind ungueltig oder koennten Zugangsdaten enthalten."""


_SENSITIVE_PARTS = (
    "api_key",
    "apikey",
    "api-token",
    "api_token",
    "access_token",
    "refresh_token",
    "authorization",
    "credential",
    "password",
    "passwd",
    "private_key",
    "secret",
    "smtp_pass",
    "cookie",
)


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key).strip().casefold().replace("-", "_")
    return any(part.replace("-", "_") in normalized for part in _SENSITIVE_PARTS)


def _find_sensitive(value: Any, prefix: str = "") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if _is_sensitive_key(key):
                return path
            found = _find_sensitive(child, path)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found = _find_sensitive(child, f"{prefix}[{index}]")
            if found:
                return found
    return None


def _without_sensitive(value: Any) -> Any:
    """Altbestand defensiv bereinigen, bevor er erneut geschrieben wird."""
    if isinstance(value, Mapping):
        return {
            str(key): _without_sensitive(child)
            for key, child in value.items()
            if not _is_sensitive_key(key)
        }
    if isinstance(value, list):
        return [_without_sensitive(child) for child in value]
    return value


def _merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        old = result.get(key)
        if isinstance(old, Mapping) and isinstance(value, Mapping):
            result[key] = _merge(dict(old), value)
        else:
            result[key] = value
    return result


def _read_existing(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise RuntimeStatusError("Bestehender Laufzeitstatus ist nicht lesbar") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeStatusError("Bestehender Laufzeitstatus enthält ungültiges JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeStatusError("Bestehender Laufzeitstatus ist kein JSON-Objekt")
    return _without_sensitive(value)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        # Auch den Directory-Eintrag nach Moeglichkeit dauerhaft machen.
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def update_status(path: str | os.PathLike[str], **fields: Any) -> dict[str, Any]:
    """Statusfelder atomar mergen und den vollstaendigen neuen State liefern.

    Verschachtelte Mappings werden rekursiv zusammengefuehrt.  Listen und
    skalare Werte ersetzen den bisherigen Wert.  ``updated_at`` wird immer auf
    einen UTC-Zeitstempel gesetzt.  Feldnamen, die nach Zugangsdaten aussehen,
    werden auch in verschachtelten Listen/Mappings abgewiesen.
    """
    sensitive = _find_sensitive(fields)
    if sensitive:
        # Nur den Feldpfad nennen, niemals dessen Inhalt.
        raise RuntimeStatusError(f"Vertrauliches Statusfeld nicht erlaubt: {sensitive}")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_name(f".{destination.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            current = _read_existing(destination)
            merged = _merge(current, fields)
            merged["updated_at"] = _timestamp()
            # Nicht-JSON-faehige Werte vor dem Ersetzen abweisen.
            try:
                json.dumps(merged, ensure_ascii=False)
            except (TypeError, ValueError) as exc:
                raise RuntimeStatusError("Laufzeitstatus ist nicht JSON-serialisierbar") from exc
            _atomic_json_write(destination, merged)
            return merged
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
