"""Energy-Arena-Submission der DE-LU-Total-Last.

Drei Formate aus EINEM Forecast-Lauf:
  point    (Challenge 20)  ein Wert je Step
  quantile (Challenge 22)  5 Quantile [2.5/25/50/75/97.5%] je Step
  ensemble (Challenge 24)  100 Member je Step

Live-POST nur wenn SUBMIT_ENABLED=true UND ARENA_API_KEY gesetzt — sonst Dry-Run.
API:  GET  {ARENA_BASE}/api/v1/challenges/open   (Header X-API-Key)
      POST {ARENA_BASE}/api/v1/submissions       {challenge_id, target_start, values}
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests

from loaders import config as lc
from . import forecast

lc.load_dotenv()
ARENA_BASE = os.environ.get("ARENA_BASE", "https://api.energy-arena.org")
ARENA_API_KEY = os.environ.get("ARENA_API_KEY", "")
SUBMIT_ENABLED = os.environ.get("SUBMIT_ENABLED", "false").lower() == "true"
SUBMIT_FORMATS = os.environ.get("SUBMIT_FORMATS", "point quantile ensemble").split()
DELU_CHALLENGES = {"point": "20", "quantile": "22", "ensemble": "24"}
REF_TZ = "Europe/Berlin"
STATE = Path(os.environ.get("APP_BASE", lc.BASE)) / "pipeline_state.json"


# --- kleiner JSON-State (Dedup der Abgaben) ---------------------------------

def _state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def _set_state(k, v):
    s = _state(); s[k] = v; STATE.write_text(json.dumps(s))


# --- Arena-API ---------------------------------------------------------------

def _headers():
    return {"X-API-Key": ARENA_API_KEY} if ARENA_API_KEY else None


def open_challenges() -> dict:
    r = requests.get(f"{ARENA_BASE}/api/v1/challenges/open", headers=_headers(), timeout=20)
    r.raise_for_status()
    return {c["challenge_id"]: c for c in r.json().get("active_challenges", [])}


def _values(fmt: str, zsamp: dict) -> list:
    if fmt == "quantile":
        q = forecast.delu_quantiles(zsamp)                      # (5, 96)
        return [q[:, t].round(3).tolist() for t in range(q.shape[1])]
    if fmt == "point":
        p = forecast.delu_point(zsamp)                          # (96,)
        return [round(float(v), 3) for v in p]
    if fmt == "ensemble":
        e = forecast.delu_ensemble(zsamp)                       # (100, 96)
        return [e[:, t].round(3).tolist() for t in range(e.shape[1])]
    raise ValueError(fmt)


def _post(challenge_id: str, target_start: str, values: list):
    if not (SUBMIT_ENABLED and ARENA_API_KEY):
        width = len(values[0]) if values and isinstance(values[0], list) else 1
        print(f"[DRY] challenge {challenge_id}: {len(values)}x{width} "
              f"(SUBMIT_ENABLED={SUBMIT_ENABLED}, key={'ja' if ARENA_API_KEY else 'nein'})",
              file=sys.stderr)
        return 0, "dry-run"
    r = requests.post(f"{ARENA_BASE}/api/v1/submissions", headers={"X-API-Key": ARENA_API_KEY},
                      json={"challenge_id": challenge_id, "target_start": target_start,
                            "values": values}, timeout=30)
    return r.status_code, r.text[:300]


def submit(target_start: str, formats=None) -> str:
    """Forecast fuer den Liefertag von `target_start` erstellen und die gewaehlten
    Formate einreichen (bzw. Dry-Run)."""
    formats = formats or SUBMIT_FORMATS
    zsamp, _ = forecast.zone_samples(delivery_date=target_start)
    ch = open_challenges()
    out = []
    for fmt in formats:
        cid = DELU_CHALLENGES.get(fmt)
        if cid is None or (ch and cid not in ch):
            continue
        code, resp = _post(cid, target_start, _values(fmt, zsamp))
        out.append(f"{fmt}#{cid}:{code}")
        print(f"  {fmt} (#{cid}) target_start={target_start} -> {code}", file=sys.stderr)
        if code >= 400:
            print(f"    response: {resp}", file=sys.stderr)
    return " ".join(out) or "no-matching-challenge"


def run_if_due() -> str:
    """Nur abgeben, wenn ein DE-LU-Quantil-Fenster offen und der Liefertag noch
    nicht (im aktuellen Modus) eingereicht wurde."""
    try:
        ch = open_challenges()
    except Exception as e:
        return f"arena-unreachable: {e}"
    cid = DELU_CHALLENGES["quantile"]
    if cid not in ch:
        return "no-open-challenge"
    c = ch[cid]
    now = pd.Timestamp.now(tz="UTC")
    opens = pd.Timestamp(c["submission_window_opens_at"]).tz_convert("UTC")
    deadline = pd.Timestamp(c["next_submission_deadline"]).tz_convert("UTC")
    target = c["next_target_start"]
    mode = "live" if (SUBMIT_ENABLED and ARENA_API_KEY) else "dry"
    key = f"last_submitted_{mode}"
    if _state().get(key) == target:
        return f"already-submitted ({mode}) {target}"
    if not (opens <= now < deadline):
        return f"not-in-window (opens {opens}, deadline {deadline})"
    res = submit(target)
    _set_state(key, target)
    return res


if __name__ == "__main__":
    print(run_if_due())
