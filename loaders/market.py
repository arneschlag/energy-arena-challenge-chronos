"""QUELLE 2/3 — Markt-/Auxiliary-Reihen je Zone + Gesamt DE-LU.

  aux_price     DE-LU Day-ahead-Preis (EUR/MWh), ENTSO-E — eine Gebotszone, fuer
                alle Gebiete identisch.
  aux_solar     realisierte PV-Erzeugung (MW), SMARD, je Zone; de_lu = Summe der 4.
  aux_wind      realisierte Wind-Erzeugung On+Offshore (MW), SMARD, je Zone;
                de_lu = Summe der 4.
  aux_residual  Residuallast = Ist-Last - Solar - Wind (MW), stuendlich, je Gebiet.

Ausgabe: data/features/aux_<name>/<gebiet>.csv  (Spalten: date, aux_<name>)

Aufruf:
    export ENTSOE_API_KEY=xxxx-...
    python -m loaders.market
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

import pandas as pd

from . import config

# SMARD chart_data
BASE_URL = "https://www.smard.de/app/chart_data"
SOLAR_FILTER = 4068                      # Photovoltaik
WIND_ONSHORE_FILTER = 4067
WIND_OFFSHORE_FILTER = 1225
RES = "hour"

START = "2024-01-01"                      # aux-Reihen ab 2024


def _end() -> str:
    return (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _feat_path(name: str, area: str):
    d = config.DATA_FEATURES / name
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{area}.csv"


# --- SMARD-Helfer -----------------------------------------------------------

def _get_json(url, retries=8):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                out = json.load(r)
            time.sleep(0.15)
            return out
        except Exception as e:
            if "404" in str(e):
                return None
            if i == retries - 1:
                raise
            time.sleep(2.0 * (i + 1))
    return None


def _fetch_filter_series(fid, region, since_ms, end_ms):
    idx = _get_json(f"{BASE_URL}/{fid}/{region}/index_{RES}.json")
    if not idx:
        return {}
    out = {}
    for t in [t for t in idx["timestamps"] if t >= since_ms - 8 * 24 * 3600 * 1000]:
        j = _get_json(f"{BASE_URL}/{fid}/{region}/{fid}_{region}_{RES}_{t}.json")
        if not j:
            continue
        for ms, v in j["series"]:
            if v is not None and since_ms <= ms <= end_ms:
                out[ms] = v
    return out


def _fetch_smard(zone, filter_ids, start, end) -> pd.Series:
    """Stuendliche Summe ueber filter_ids fuer eine Zone -> Series (date-indexiert)."""
    region = config.SMARD_REGION[zone]
    since_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    total = {}
    for fid in filter_ids:
        for ms, v in _fetch_filter_series(fid, region, since_ms, end_ms).items():
            total[ms] = total.get(ms, 0.0) + v
    if not total:
        return pd.Series(dtype="float64")
    s = pd.Series(total).sort_index()
    s.index = pd.to_datetime(s.index, unit="ms", utc=True)
    return s.resample("1h").mean()


def _save(series: pd.Series, name: str, area: str) -> None:
    out = series.dropna().reset_index()
    out.columns = ["date", name]
    out.to_csv(_feat_path(name, area), index=False)
    print(f"  [{area}] {name}: {len(out)} Zeilen", file=sys.stderr)


# --- 1. Preis (global) ------------------------------------------------------

def build_price(client) -> None:
    print("=== aux_price (DE-LU Day-ahead, ENTSO-E) ===", file=sys.stderr)
    start = pd.Timestamp(START, tz="Europe/Berlin")
    end = pd.Timestamp(_end(), tz="Europe/Berlin")
    prices = client.query_day_ahead_prices("DE_LU", start=start, end=end).tz_convert("UTC")
    # identische Reihe fuer alle Gebiete inkl. lu (Luxemburg teilt den DE-LU-Preis)
    for area in config.ALL_AREAS + [config.LU]:
        _save(prices.rename("aux_price"), "aux_price", area)


# --- 2./3. Solar & Wind je Zone, de_lu = Summe der 4 ------------------------

def _build_gen(name: str, filter_ids: list[int]) -> None:
    print(f"=== {name} (SMARD, je Zone) ===", file=sys.stderr)
    end = _end()
    zone_series = {}
    for zone in config.ZONES:
        s = _fetch_smard(zone, filter_ids, START, end)
        if s.empty:
            print(f"  [{zone}] WARN: keine {name}-Daten", file=sys.stderr)
            continue
        zone_series[zone] = s
        _save(s, name, zone)
    if zone_series:                          # de_lu = Summe der Zonen
        delu = pd.concat(zone_series.values(), axis=1).sum(axis=1, min_count=1)
        _save(delu, name, config.DELU)


def build_solar() -> None:
    _build_gen("aux_solar", [SOLAR_FILTER])


def build_wind() -> None:
    _build_gen("aux_wind", [WIND_ONSHORE_FILTER, WIND_OFFSHORE_FILTER])


# --- 4. Residuallast je Gebiet ----------------------------------------------

def _read_actual_hourly(area: str) -> pd.Series:
    parts = [pd.read_csv(f, usecols=["date", "actual_load_mw"])
             for f in sorted(config.DATA_LOADS.glob(f"{area}_*.csv"))]
    if not parts:
        return pd.Series(dtype="float64")
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    s = df.drop_duplicates("date").set_index("date")["actual_load_mw"].sort_index()
    return s.resample("1h").mean()


def _read_feature(name: str, area: str) -> pd.Series:
    p = _feat_path(name, area)
    if not p.exists():
        return pd.Series(dtype="float64")
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.set_index("date")[name]


def build_residual() -> None:
    print("=== aux_residual (Ist-Last - Solar - Wind) ===", file=sys.stderr)
    for area in config.ALL_AREAS:
        load = _read_actual_hourly(area)
        solar = _read_feature("aux_solar", area)
        wind = _read_feature("aux_wind", area)
        if load.empty or solar.empty or wind.empty:
            print(f"  [{area}] WARN: Last/Solar/Wind fehlt -> uebersprungen", file=sys.stderr)
            continue
        comb = pd.DataFrame({"load": load, "solar": solar, "wind": wind}).dropna()
        _save((comb["load"] - comb["solar"] - comb["wind"]).rename("aux_residual"),
              "aux_residual", area)


def main() -> None:
    import argparse
    argparse.ArgumentParser(description="Markt-/Aux-Reihen je Zone + de_lu").parse_args()
    token = config.entsoe_key()
    if not token:
        sys.exit("Kein ENTSO-E-Token. Setze ENTSOE_API_KEY (.env).")
    from entsoe import EntsoePandasClient
    client = EntsoePandasClient(api_key=token)
    build_price(client)
    build_solar()
    build_wind()
    build_residual()
    print("Fertig.", file=sys.stderr)


if __name__ == "__main__":
    main()
