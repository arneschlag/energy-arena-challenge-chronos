"""Scheduler des einzigen schreibenden Daten-/Summary-Containers.

Dieser Prozess ist der einzige Produktionsdienst, der Loader aufruft. Forecast-
Worker mounten ``data/`` read-only und besitzen deshalb weder einen Loader-Job noch
einen Backfill-Pfad. Ein vorhandenes Mail-/Summary-Modul kann ueber
``SUMMARY_MODULE`` referenziert werden, ohne es an diesen Scheduler zu koppeln.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time

from loaders import config as lc
from loaders import schedule as lsched
from .locking import data_read_lock, data_write_lock


def _safe(fn, name: str):
    started = time.time()
    try:
        result = fn()
        suffix = "" if result is None else f": {result}"
        print(f"[ok] {name}{suffix} ({time.time() - started:.0f}s)",
              file=sys.stderr, flush=True)
    except BaseException as exc:
        # Ein einzelner externer Daten- oder Summary-Fehler darf den Scheduler nicht
        # beenden. KeyboardInterrupt/SystemExit werden im Hauptthread weitergereicht.
        if isinstance(sys.exc_info()[1], (KeyboardInterrupt, SystemExit)):
            raise
        # Externe ENTSO-E-Fehler koennen Request-URLs mit Token enthalten.
        print(f"[FEHLER] {name}: {type(exc).__name__}",
              file=sys.stderr, flush=True)


def _needs_backfill() -> bool:
    return not any(lc.DATA_LOADS.glob("*.csv"))


def job_refresh_all() -> None:
    with data_write_lock():
        lsched.run_refresh()


def job_loads() -> None:
    """Kleiner stuendlicher Tick fuer Istlast und ENTSO-E-Benchmark."""
    with data_write_lock():
        lsched.job_loads_refresh()


def job_market() -> None:
    """Preis/Solar/Wind vor dem Day-Ahead-Fenster aktualisieren."""
    with data_write_lock():
        lsched.job_market()


def job_weather() -> None:
    """Wetterforecast laden und anschliessend auf die Gebiete aggregieren."""
    with data_write_lock():
        lsched.job_weather_daily()


def job_summary() -> str:
    """Optionales bestehendes Summary-Modul als ``python -m MODUL`` starten."""
    module = os.environ.get("SUMMARY_MODULE", "").strip()
    if not module:
        return "deaktiviert (SUMMARY_MODULE leer)"
    args = shlex.split(os.environ.get("SUMMARY_ARGS", ""))
    with data_read_lock():
        subprocess.run([sys.executable, "-m", module, *args], check=True)
    return f"Modul {module}"


def main() -> None:
    lc.load_dotenv()
    load_minute = os.environ.get(
        "LOAD_REFRESH_MINUTE", os.environ.get("DATA_REFRESH_MINUTE", "45")
    )
    market_hours = os.environ.get("MARKET_HOURS_UTC", "8,9")
    market_minute = os.environ.get("MARKET_MINUTE_UTC", "20")
    weather_hours = os.environ.get("WEATHER_HOURS_UTC", "6,8,9")
    weather_minute = os.environ.get("WEATHER_MINUTE_UTC", "0")
    summary_hour = os.environ.get("SUMMARY_HOUR_LOCAL", "11")
    summary_minute = os.environ.get("SUMMARY_MINUTE_LOCAL", "55")
    summary_timezone = os.environ.get("SUMMARY_TIMEZONE", "Europe/Berlin")
    print("### Data/Summary-Orchestrator"
          f" | load=UTC *:{load_minute}"
          f" | market=UTC {market_hours}:{market_minute}"
          f" | weather=UTC {weather_hours}:{weather_minute}"
          f" | summary={summary_timezone} {summary_hour}:{summary_minute}"
          f" | models={os.environ.get('SUMMARY_MODEL_LABELS', 'nicht gesetzt')}",
          file=sys.stderr, flush=True)

    # Nur der Data-Service darf Erstbefuellung und initialen Refresh ausfuehren.
    if _needs_backfill():
        print("Kein Last-Datenbestand -> einmaliger Voll-Backfill", file=sys.stderr, flush=True)
        _safe(lambda: _run_backfill(), "backfill")
    _safe(job_refresh_all, "data-refresh/startup")

    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(lambda: _safe(job_loads, "loads-refresh"),
                      CronTrigger(minute=load_minute), id="loads",
                      max_instances=1, coalesce=True, misfire_grace_time=600)
    scheduler.add_job(lambda: _safe(job_market, "market-refresh"),
                      CronTrigger(hour=market_hours, minute=market_minute), id="market",
                      max_instances=1, coalesce=True, misfire_grace_time=3600)
    scheduler.add_job(lambda: _safe(job_weather, "weather-refresh"),
                      CronTrigger(hour=weather_hours, minute=weather_minute), id="weather",
                      max_instances=1, coalesce=True, misfire_grace_time=3600)
    if os.environ.get("SUMMARY_MODULE", "").strip():
        scheduler.add_job(lambda: _safe(job_summary, "summary"),
                          CronTrigger(hour=summary_hour, minute=summary_minute,
                                      timezone=summary_timezone),
                          id="summary", max_instances=1, coalesce=True,
                          misfire_grace_time=3600)
    print("Data/Summary-Scheduler laeuft; keine Forecasts oder Submissions in dieser Rolle.",
          file=sys.stderr, flush=True)
    scheduler.start()


def _run_backfill() -> None:
    with data_write_lock():
        lsched.run_once()


if __name__ == "__main__":
    main()
