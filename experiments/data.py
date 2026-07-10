"""Laden der sauberen loader-Ausgaben und Zusammenbau aligned 15-min-Reihen.

Alle Reihen UTC, 15-minuetlich. Aux (Preis/Solar/Wind) ist stuendlich und wird per
forward-fill auf das 15-min-Raster gebracht. Leckage-sichere Fenster erzeugt
walkforward.py; hier nur das Laden/Alignen.

Datenquellen (code/data/):
  loads/{area}_{jahr}.csv            target (actual_load_mw)
  load_forecast/{area}_{jahr}.csv    ENTSO-E-Benchmark (forecast_load_mw)
  weather_zone/{pop,centroid}/{area}.csv   aggregiertes Zonen-Wetter (7 Vars)
  historical/{h3}_{jahr}.csv         je-Zell-Wetter (fuer C2.3 even4)
  features/aux_{price,solar,wind}/{area}.csv   Aux (stuendlich)
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from loaders import config, grid

WEATHER_VARS = config.WEATHER_VARS
AUX_NAMES = ("price", "solar", "wind")


def _read_concat(paths, valcol, rename):
    parts = []
    for p in paths:
        if p.exists():
            parts.append(pd.read_csv(p, usecols=["date", valcol]))
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    s = df.drop_duplicates("date").set_index("date")[valcol].sort_index()
    return s.rename(rename)


@lru_cache(maxsize=None)
def target(area: str) -> pd.Series:
    """Ist-Last (15-min) eines Gebiets."""
    s = _read_concat(sorted(config.DATA_LOADS.glob(f"{area}_2*.csv")),
                     "actual_load_mw", "target")
    if s is None:
        raise FileNotFoundError(f"keine Last-Daten fuer {area}")
    return s


@lru_cache(maxsize=None)
def benchmark(area: str) -> pd.Series | None:
    """ENTSO-E Day-ahead-Prognose (15-min) oder None."""
    return _read_concat(sorted(config.DATA_FCAST.glob(f"{area}_2*.csv")),
                        "forecast_load_mw", "benchmark")


@lru_cache(maxsize=None)
def aux(area: str, name: str) -> pd.Series | None:
    """Aux-Reihe (stuendlich) oder None, wenn fuer das Gebiet nicht vorhanden."""
    p = config.DATA_FEATURES / f"aux_{name}" / f"{area}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.drop_duplicates("date").set_index("date")[f"aux_{name}"].sort_index()


def has_aux(area: str, name: str) -> bool:
    return aux(area, name) is not None


@lru_cache(maxsize=None)
def weather(area: str, source: str) -> pd.DataFrame:
    """Wetter-Covariaten (15-min) je Quelle:
       'centroid' / 'pop' -> weather_zone/{source}/{area}.csv (7 Spalten)
       'even4'            -> 4 gleichmaessig verteilte Zellen, je 7 Spalten
                             (Spalten <var>__c0..c3; lu -> 1 Zelle)."""
    if source in ("pop", "centroid"):
        p = config.DATA_WEATHER_ZONE / source / f"{area}.csv"
        df = pd.read_csv(p)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df.set_index("date")[list(WEATHER_VARS)].sort_index()
    if source == "even4":
        cols = {}
        for i, h in enumerate(grid.even_cells(area, 4)):
            cs = _read_cell(h)
            for v in WEATHER_VARS:
                cols[f"{v}__c{i}"] = cs[v]
        df = pd.DataFrame(cols).sort_index()
        return df
    raise ValueError(f"unbekannte Wetterquelle {source}")


@lru_cache(maxsize=None)
def _read_cell(h3_index: str) -> pd.DataFrame:
    parts = [pd.read_csv(f, usecols=["date"] + list(WEATHER_VARS))
             for f in sorted(config.DATA_HIST.glob(f"{h3_index}_2*.csv"))]
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.drop_duplicates("date").set_index("date").sort_index()


def frame(area: str, weather_source: str | None = None,
          aux_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    """Aligned 15-min-DataFrame: 'target' + optional Wetter + optional aux_<name>.
    Aux wird per ffill aufs 15-min-Raster gebracht. Fehlende aux -> Spalte entfaellt."""
    df = target(area).to_frame()
    if weather_source:
        df = df.join(weather(area, weather_source), how="inner")
    for name in aux_cols:
        s = aux(area, name)
        if s is not None:
            df[f"aux_{name}"] = s.reindex(df.index, method="ffill")
    return df.dropna().sort_index()
