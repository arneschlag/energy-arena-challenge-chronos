"""Produktions-Scheduler (vollautomatisch): Daten frisch halten + Submit.

Beim Start: fehlt der Datenbestand -> Voll-Backfill; danach einmal Refresh + Submit.
Dann APScheduler (UTC):
  :05 stuendlich  Daten-Refresh (Wetter inkl. naechste Tage, Last, Markt, Aggregation)
  :20 stuendlich  Arena-Submit, wenn ein DE-LU-Fenster offen und noch nicht abgegeben

Alles env-gesteuert (.env): ENTSOE_API_KEY (Pflicht), ARENA_API_KEY + SUBMIT_ENABLED
fuer echte Abgabe (sonst Dry-Run), PROD_CONFIG (Modell), DEVICE (cuda|cpu).

Start:  python -m pipeline.orchestrator
"""
from __future__ import annotations

import sys
import time
import traceback

from loaders import config as lc
from loaders import schedule as lsched
from . import arena


def _safe(fn, name):
    t0 = time.time()
    try:
        r = fn()
        print(f"[ok] {name}: {r if r is not None else ''} ({time.time()-t0:.0f}s)",
              file=sys.stderr, flush=True)
    except Exception:
        print(f"[FEHLER] {name}:\n{traceback.format_exc()}", file=sys.stderr, flush=True)


def _needs_backfill() -> bool:
    return not any(lc.DATA_LOADS.glob("*.csv"))


def job_data():
    lsched.run_refresh()          # Wetter (Luecken + naechste Tage), Last, Markt, Aggregation


def job_submit():
    return arena.run_if_due()


def main():
    lc.load_dotenv()
    print(f"### Produktions-Orchestrator Start {time.strftime('%Y-%m-%d %H:%M:%S')} "
          f"| SUBMIT_ENABLED={arena.SUBMIT_ENABLED} | model={__import__('os').environ.get('PROD_CONFIG','G1_C5')}",
          file=sys.stderr, flush=True)
    if _needs_backfill():
        print("Kein Datenbestand -> Voll-Backfill (einmalig, kann dauern)", file=sys.stderr, flush=True)
        _safe(lsched.run_once, "backfill")
    _safe(job_data, "data-refresh")
    _safe(job_submit, "submit")

    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    sched = BlockingScheduler(timezone="UTC")
    sched.add_job(lambda: _safe(job_data, "data-refresh"), CronTrigger(minute=5),
                  id="data", max_instances=1, misfire_grace_time=600)
    sched.add_job(lambda: _safe(job_submit, "submit"), CronTrigger(minute=20),
                  id="submit", max_instances=1, misfire_grace_time=600)
    print("Scheduler laeuft (UTC): :05 Daten, :20 Submit. Strg+C beendet.",
          file=sys.stderr, flush=True)
    sched.start()


if __name__ == "__main__":
    main()
