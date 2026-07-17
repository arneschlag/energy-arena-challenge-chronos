"""Der EINE Scheduler der Daten-Pipeline (nur Daten, kein Training/Submit).

Drei Betriebsarten:
  --once     Voll-Backfill: alles einmal der Reihe nach (fuer den Erststart).
  --refresh  EIN inkrementeller Tick, dann Ende — fuer externe Planung (cron/systemd,
             z.B. alle 2 h): Punkt-Wetter rollierend (Luecken der letzten Tage +
             naechste 4 Tage) + Last-Refresh (letzte 5 Tage mergen) + Markt/Aux +
             Aggregation. Kein Voll-Jahres-Backfill.
  (default)  Daemon mit APScheduler (UTC), eigene Trigger:
             :05 stuendlich  -> loads.refresh_recent(5)  (Ist-Last + Prognose mergen)
             06:30 taeglich   -> market  (Preis/Solar/Wind/Residual neu bauen)
             07:00 taeglich   -> Punktwetter-Refresh + Aggregation

Punkt-Wetter- und Voll-Jahres-Last-Backfill sind Einmal-Laeufe (--once bzw. die
Einzelmodule) und laufen bewusst NICHT im laufenden Betrieb.

Aufruf:
    export ENTSOE_API_KEY=xxxx-...
    python -m loaders.schedule --once      # Erststart / Voll-Backfill
    python -m loaders.schedule --refresh   # ein Tick (extern alle 2 h planen)
    python -m loaders.schedule             # eigener Daemon
"""
from __future__ import annotations

import argparse
import sys

from . import aggregate, config, loads, market, weather


def _safe(fn, name: str) -> None:
    try:
        fn()
        print(f"[ok] {name}", file=sys.stderr)
    except Exception as exc:
        # ENTSO-E-Fehler koennen die Request-URL inklusive Token enthalten.
        # Deshalb nur den Exception-Typ, niemals den fremden Nachrichtentext loggen.
        print(f"[FEHLER] {name}: {type(exc).__name__}", file=sys.stderr)


def _token() -> str:
    token = config.entsoe_key()
    if not token:
        sys.exit("Kein ENTSO-E-Token. Setze ENTSOE_API_KEY (.env).")
    return token


# --- Jobs -------------------------------------------------------------------

def job_loads_refresh() -> None:
    loads.refresh_recent(days=5, token=_token())


def job_market() -> None:
    from entsoe import EntsoePandasClient

    client = EntsoePandasClient(api_key=_token())
    # Eine ausgefallene Quelle darf die beiden anderen nicht ueberspringen.
    _safe(lambda: market.build_price(client), "market.price")
    _safe(market.build_solar, "market.solar")
    _safe(market.build_wind, "market.wind")
    _safe(market.build_residual, "market.residual")


def job_weather_daily() -> None:
    weather.refresh_recent(past_days=7, forecast_days=4)   # Luecken + naechste Arena-Tage
    aggregate.aggregate_point()
    aggregate.analyse_diff()


def run_once() -> None:
    """Voll-Backfill fuer den Erststart."""
    token = _token()
    _safe(lambda: weather.download_point(config.DEFAULT_YEARS), "weather.point")
    _safe(lambda: loads.download(config.DEFAULT_YEARS, token), "loads.download")
    _safe(loads.reconcile_delu, "loads.reconcile_delu")
    _safe(job_market, "market")
    _safe(aggregate.aggregate_point, "aggregate.point")
    _safe(aggregate.analyse_diff, "aggregate.diff")


def run_refresh() -> None:
    """Ein inkrementeller Tick fuer den laufenden Betrieb (idempotent, dann Ende).
    Fuer externe Planung gedacht (cron/systemd-Timer, z.B. alle 2 h)."""
    _token()                                     # Token frueh pruefen
    # Punkt-Wetter rollierend: Luecken der letzten Tage schliessen + naechste Arena-Tage
    _safe(lambda: weather.refresh_recent(past_days=7, forecast_days=4), "weather.refresh")
    _safe(job_loads_refresh, "loads.refresh")    # letzte 5 Tage Ist+Prognose mergen
    _safe(job_market, "market")                  # Preis/Solar/Wind/Residual (bis heute)
    _safe(aggregate.aggregate_point, "aggregate.point")
    _safe(aggregate.analyse_diff, "aggregate.diff")


def run_daemon() -> None:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    sched = BlockingScheduler(timezone="UTC")
    sched.add_job(lambda: _safe(job_loads_refresh, "loads.refresh"),
                  CronTrigger(minute=5), id="loads", max_instances=1)
    sched.add_job(lambda: _safe(job_market, "market"),
                  CronTrigger(hour=6, minute=30), id="market", max_instances=1)
    sched.add_job(lambda: _safe(job_weather_daily, "weather+aggregate"),
                  CronTrigger(hour=7, minute=0), id="weather", max_instances=1)
    print("Scheduler laeuft (UTC). Strg+C zum Beenden.", file=sys.stderr)
    sched.start()


def main() -> None:
    p = argparse.ArgumentParser(description="Daten-Pipeline-Scheduler")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--once", action="store_true", help="Voll-Backfill statt Daemon")
    g.add_argument("--refresh", action="store_true",
                   help="ein inkrementeller Tick, dann Ende (extern planen)")
    a = p.parse_args()
    config.load_dotenv()
    if a.once:
        run_once()
    elif a.refresh:
        run_refresh()
    else:
        run_daemon()


if __name__ == "__main__":
    main()
