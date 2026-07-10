"""QUELLE 1 — Wetter je H3-Zelle (Open-Meteo), 15-minuetlich, 7 Variablen.

Zwei Produkte, beide JE ZELLE gespeichert (Aggregation -> aggregate.py):

  point     Punktprognose (deterministisch), 1 Request je Zelle & Jahr
            Endpoint: historical-forecast-api
            -> data/historical/{hex}_{jahr}.csv   (2023-2025 voll; 2026 bis Stichtag)

  refresh   rollierend fuer den Betrieb: Forecast-Endpoint mit past_days (Luecken der
            letzten Tage schliessen) + forecast_days (naechste 2 Tage), je Zelle in
            die Jahres-CSVs gemergt (neuer Wert gewinnt). -> data/historical/{hex}_{jahr}.csv

  ensemble  DWD ICON-D2-EPS, 20 Member, ~48 h Horizont (Wetter-Unsicherheit)
            Endpoint: ensemble-api  (nur ~90 Tage frei zurueck)
            -> data/historical_ens/{hex}_{datum}.csv   (Spalten: member, date, <vars>)

Aufruf:
    python -m loaders.weather point                    # Voll-Backfill je Jahr
    python -m loaders.weather point --years 2026
    python -m loaders.weather refresh                  # Luecken + naechste 2 Tage
    python -m loaders.weather refresh --past-days 30
    python -m loaders.weather ensemble                 # naechster Tag
    python -m loaders.weather ensemble --start 2026-06-05 --end 2026-06-07
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import config, grid

# --- Endpunkte / Parameter --------------------------------------------------
HIST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FCAST_URL = "https://api.open-meteo.com/v1/forecast"    # rollierend: past_days + forecast_days
ENS_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
ENS_MODEL = "icon_d2"                                   # DWD, 2.2 km, native 15-min
ENS_VARS = [v for v in config.WEATHER_VARS if v != "is_day"]  # is_day deterministisch
SLEEP = 0.4                                             # Free-Tier schonen


def _point_client():
    """Cache+Retry-Client fuer die Punktprognose (lazy — nur bei Bedarf)."""
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry
    cache = requests_cache.CachedSession(str(config.BASE / ".cache"), expire_after=3600)
    return openmeteo_requests.Client(session=retry(cache, retries=5, backoff_factor=0.2))


# --- Punktprognose ----------------------------------------------------------

def _point_response_to_df(response) -> pd.DataFrame:
    """Open-Meteo-Response -> DataFrame (Spaltenreihenfolge = WEATHER_VARS)."""
    m15 = response.Minutely15()
    data = {
        "date": pd.date_range(
            start=pd.to_datetime(m15.Time(), unit="s", utc=True),
            end=pd.to_datetime(m15.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=m15.Interval()),
            inclusive="left",
        )
    }
    for i, name in enumerate(config.WEATHER_VARS):
        data[name] = m15.Variables(i).ValuesAsNumpy()
    return pd.DataFrame(data)


def download_point(years: list[int]) -> None:
    """1 Request je Zelle & Jahr -> data/historical/{hex}_{jahr}.csv (skip-wenn-da)."""
    g = grid.download_cells()               # 224 dt. Zellen + LU-Solo-Zelle
    client = _point_client()
    config.DATA_HIST.mkdir(parents=True, exist_ok=True)
    total = len(g) * len(years)
    done = 0
    print(f"Punkt-Wetter: {len(g)} Zellen x {len(years)} Jahre = {total} Requests",
          file=sys.stderr)

    for cell in g.itertuples(index=False):
        for year in years:
            done += 1
            out = config.DATA_HIST / f"{cell.h3_index}_{year}.csv"
            if out.exists():
                continue                                # wiederaufnehmbar
            start, end = config.year_bounds(year)
            params = {"latitude": cell.lat, "longitude": cell.lng,
                      "start_date": start, "end_date": end,
                      "minutely_15": config.WEATHER_VARS}
            try:
                resp = client.weather_api(HIST_URL, params=params)[0]
            except Exception as exc:
                print(f"  FEHLER {cell.h3_index} {year}: {exc}", file=sys.stderr)
                time.sleep(2.0)
                continue
            df = _point_response_to_df(resp)
            df.insert(0, "tso", config.ZONE_LABEL.get(cell.zone, ""))
            df.insert(0, "h3_index", cell.h3_index)
            df.to_csv(out, index=False)
            print(f"  [{done}/{total}] {cell.h3_index} {year} -> {out.name} ({len(df)})",
                  file=sys.stderr)
            if not getattr(resp, "_from_cache", False):
                time.sleep(SLEEP)


# --- Rollierender Refresh (Luecken schliessen + naechste 2 Tage) -------------

def _merge_point_csv(h3_index: str, tso: str, df: pd.DataFrame) -> None:
    """df (date + WEATHER_VARS) nach Jahr in die historical/{hex}_{jahr}.csv mergen.
    Dedup nach date, NEUER Wert gewinnt (aktualisiert die Vorhersage)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df.insert(0, "tso", tso)
    df.insert(0, "h3_index", h3_index)
    cols = ["h3_index", "tso", "date"] + config.WEATHER_VARS
    for year, gy in df.groupby(df["date"].dt.year):
        path = config.DATA_HIST / f"{h3_index}_{year}.csv"
        if path.exists():
            old = pd.read_csv(path)
            old["date"] = pd.to_datetime(old["date"], utc=True)
            gy = pd.concat([old, gy], ignore_index=True)
        gy = gy.drop_duplicates("date", keep="last").sort_values("date")
        gy[cols].to_csv(path, index=False)


def refresh_recent(past_days: int = 7, forecast_days: int = 2, chunk: int = 100) -> None:
    """Punkt-Wetter rollierend aktualisieren: fuellt Luecken der letzten `past_days`
    UND verlaengert um `forecast_days` (Standard 2) — je Zelle in die Jahres-CSVs
    gemergt. Multi-Location (Forecast-Endpoint), daher ~3 Requests fuer alle Zellen."""
    g = grid.download_cells()               # 224 dt. Zellen + LU-Solo-Zelle
    client = _point_client()
    config.DATA_HIST.mkdir(parents=True, exist_ok=True)
    print(f"Punkt-Wetter-Refresh: {len(g)} Zellen, past_days={past_days} "
          f"+ forecast_days={forecast_days}", file=sys.stderr)
    updated = 0
    for i in range(0, len(g), chunk):
        sub = g.iloc[i:i + chunk]
        params = {"latitude": sub.lat.tolist(), "longitude": sub.lng.tolist(),
                  "minutely_15": config.WEATHER_VARS,
                  "past_days": past_days, "forecast_days": forecast_days}
        try:
            resp = client.weather_api(FCAST_URL, params=params)
        except Exception as exc:
            print(f"  FEHLER Chunk {i}: {exc}", file=sys.stderr)
            time.sleep(2.0)
            continue
        if not isinstance(resp, list):
            resp = [resp]
        for cell, r in zip(sub.itertuples(index=False), resp):
            _merge_point_csv(cell.h3_index, config.ZONE_LABEL.get(cell.zone, ""),
                             _point_response_to_df(r))
            updated += 1
        time.sleep(SLEEP)
    print(f"  {updated} Zellen aktualisiert -> {config.DATA_HIST}", file=sys.stderr)


# --- Ensemble ---------------------------------------------------------------

def _ens_fetch(lats, lngs, forecast_days, start=None, end=None):
    """Roh-Request an die Ensemble-API (Liste von Location-Dicts)."""
    q = {"latitude": ",".join(f"{x:.4f}" for x in lats),
         "longitude": ",".join(f"{x:.4f}" for x in lngs),
         "minutely_15": ",".join(ENS_VARS), "models": ENS_MODEL}
    if start:
        q["start_date"], q["end_date"] = start, end
    else:
        q["forecast_days"] = forecast_days
    url = ENS_URL + "?" + urllib.parse.urlencode(q)
    for i in range(6):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                out = json.load(r)
            return out if isinstance(out, list) else [out]
        except Exception:
            if i == 5:
                raise
            time.sleep(2.5 * (i + 1))


def download_ensemble(forecast_days: int = 2, chunk: int = 30,
                      start: str | None = None, end: str | None = None) -> None:
    """ICON-D2-EPS je Zelle -> data/historical_ens/{hex}_{datum}.csv.

    Speichert die Member ROH je Zelle (member, date, <ENS_VARS>, is_day); die
    Aggregation auf Zonen (pop / centroid) uebernimmt aggregate.py. Dateiname
    nutzt das erste Prognosedatum (bei --start/--end den Starttag)."""
    g = grid.download_cells()               # 224 dt. Zellen + LU-Solo-Zelle
    config.DATA_HIST_ENS.mkdir(parents=True, exist_ok=True)
    day = start or (date.today() + timedelta(days=1)).isoformat()

    print(f"Ensemble-Wetter ({ENS_MODEL}, {day}): {len(g)} Zellen in Chunks a {chunk}",
          file=sys.stderr)
    written = 0
    for i in range(0, len(g), chunk):
        sub = g.iloc[i:i + chunk]
        resp = _ens_fetch(sub.lat.tolist(), sub.lng.tolist(), forecast_days, start, end)
        for cell, loc in zip(sub.itertuples(index=False), resp):
            h = loc["minutely_15"]
            times = pd.to_datetime(h["time"], utc=True)
            # member00 = Kontroll-Lauf, danach _memberNN
            frames = []
            n_mem = None
            member_cols = {v: sorted(k for k in h if k.startswith(v + "_member"))
                           for v in ENS_VARS}
            n_mem = 1 + len(member_cols[ENS_VARS[0]])
            for m in range(n_mem):
                d = {"member": m, "date": times}
                for v in ENS_VARS:
                    col = h[v] if m == 0 else h[member_cols[v][m - 1]]
                    d[v] = np.nan_to_num(np.asarray(col, "float32"))
                df = pd.DataFrame(d)
                df["is_day"] = (df["shortwave_radiation"] > 0).astype("float32")
                frames.append(df)
            out_df = pd.concat(frames, ignore_index=True)
            out_df.insert(0, "tso", config.ZONE_LABEL.get(cell.zone, ""))
            out_df.insert(0, "h3_index", cell.h3_index)
            out_df.to_csv(config.DATA_HIST_ENS / f"{cell.h3_index}_{day}.csv", index=False)
            written += 1
        print(f"  Zellen {i + 1}-{min(i + chunk, len(g))}/{len(g)}", file=sys.stderr, flush=True)
        time.sleep(1.0)
    print(f"{written} Zellen -> {config.DATA_HIST_ENS} ({day})", file=sys.stderr)


def main() -> None:
    config.load_dotenv()
    p = argparse.ArgumentParser(description="Wetter je H3-Zelle (Punkt + Ensemble)")
    sub = p.add_subparsers(dest="mode", required=True)
    pp = sub.add_parser("point", help="Punktprognose (historisch, je Zelle/Jahr)")
    pp.add_argument("--years", type=int, nargs="+", default=config.DEFAULT_YEARS)
    pr = sub.add_parser("refresh", help="Punkt-Wetter rollierend: Luecken + naechste Tage")
    pr.add_argument("--past-days", type=int, default=7)
    pr.add_argument("--forecast-days", type=int, default=2)
    pe = sub.add_parser("ensemble", help="ICON-D2-EPS (je Zelle/Tag)")
    pe.add_argument("--forecast-days", type=int, default=2)
    pe.add_argument("--start", default=None)
    pe.add_argument("--end", default=None)
    a = p.parse_args()
    if a.mode == "point":
        download_point(sorted(set(a.years)))
    elif a.mode == "refresh":
        refresh_recent(past_days=a.past_days, forecast_days=a.forecast_days)
    else:
        download_ensemble(a.forecast_days, start=a.start, end=a.end)


if __name__ == "__main__":
    main()
