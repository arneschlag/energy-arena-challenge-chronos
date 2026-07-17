"""QUELLE 1 (Weiterverarbeitung) — Zell-Wetter -> Zonen-Wetter, ZWEI Methoden.

Aus den je-Zelle gespeicherten Reihen (weather.py) wird pro Gebiet (4 Zonen +
de_lu) EINE Wetterreihe erzeugt — auf zwei Arten:

  pop       bevoelkerungsgewichtetes Mittel ueber alle Zellen des Gebiets (aktuell)
  centroid  nur die Repraesentativ-Zelle (Bevoelkerungs-Schwerpunkt, grid.centroid_cell)

Ausgabe:
  data/weather_zone/pop/{gebiet}.csv        (date, <WEATHER_VARS>)
  data/weather_zone/centroid/{gebiet}.csv
  data/weather_zone/{pop,centroid}/ens_{gebiet}.csv   (member, date, <WEATHER_VARS>)

Zusaetzlich beantwortet analyse_diff() die Frage "wie stark unterscheiden sich die
beiden Aggregationen": MAE, RMSE, Korrelation je Gebiet & Variable
  -> data/analysis/weather_agg_diff.csv

Aufruf:
    python -m loaders.aggregate            # beide Methoden (point) + Diff-Analyse
    python -m loaders.aggregate --ensemble # zusaetzlich Ensemble aggregieren
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from . import config, grid

METHODS = ("pop", "centroid")


# --- Einlesen ---------------------------------------------------------------

def _read_cell_point(h3_index: str) -> pd.DataFrame | None:
    """Alle Jahre der Punktprognose einer Zelle -> date-indexiert, nur WEATHER_VARS."""
    parts = []
    for f in sorted(config.DATA_HIST.glob(f"{h3_index}_*.csv")):
        d = pd.read_csv(f, usecols=["date"] + config.WEATHER_VARS)
        parts.append(d)
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.drop_duplicates("date").set_index("date").sort_index()


def _read_cell_ens(h3_index: str) -> pd.DataFrame | None:
    """Alle Ensemble-Dateien einer Zelle -> [member, date, <WEATHER_VARS>]."""
    parts = []
    for f in sorted(config.DATA_HIST_ENS.glob(f"{h3_index}_*.csv")):
        parts.append(pd.read_csv(f, usecols=["member", "date"] + config.WEATHER_VARS))
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.drop_duplicates(["member", "date"])


# --- Punkt-Aggregation ------------------------------------------------------

def _aggregate_point_area(area: str, method: str, g: pd.DataFrame) -> pd.DataFrame | None:
    zc = grid.zone_cells(area, g)
    if method == "centroid":
        h = grid.centroid_cell(area, g)
        df = _read_cell_point(h)
        return None if df is None else df[config.WEATHER_VARS]
    # pop: gewichtete Summe ueber Zellen, inkrementell (spart Speicher)
    acc = None
    for cell in zc.itertuples(index=False):
        df = _read_cell_point(cell.h3_index)
        if df is None:
            continue
        w = df[config.WEATHER_VARS] * cell.w
        acc = w if acc is None else acc.add(w, fill_value=0.0)
    return acc


def aggregate_point(methods=METHODS) -> None:
    g = grid.cells()
    for method in methods:
        out_dir = config.DATA_WEATHER_ZONE / method
        out_dir.mkdir(parents=True, exist_ok=True)
        for area in config.AGG_AREAS:
            df = _aggregate_point_area(area, method, g)
            if df is None:
                print(f"  [{method}/{area}] keine Punktdaten", file=sys.stderr)
                continue
            config.atomic_to_csv(df.reset_index(), out_dir / f"{area}.csv", index=False)
            print(f"  [{method}/{area}] {len(df)} Zeilen -> {out_dir.name}/{area}.csv",
                  file=sys.stderr)


# --- Ensemble-Aggregation ---------------------------------------------------

def _aggregate_ens_area(area: str, method: str, g: pd.DataFrame) -> pd.DataFrame | None:
    zc = grid.zone_cells(area, g)
    if method == "centroid":
        h = grid.centroid_cell(area, g)
        df = _read_cell_ens(h)
        return None if df is None else df[["member", "date"] + config.WEATHER_VARS]
    # pop: je (member, date) gewichtete Summe ueber die Zellen des Gebiets
    frames = []
    for cell in zc.itertuples(index=False):
        df = _read_cell_ens(cell.h3_index)
        if df is None:
            continue
        for v in config.WEATHER_VARS:
            df[v] = df[v] * cell.w
        frames.append(df)
    if not frames:
        return None
    allc = pd.concat(frames, ignore_index=True)
    agg = allc.groupby(["member", "date"], as_index=False)[config.WEATHER_VARS].sum()
    # is_day ist ein Anteil nach der Gewichtung -> wieder binaer machen
    agg["is_day"] = (agg["is_day"] > 0.5).astype("float32")
    return agg


def aggregate_ensemble(methods=METHODS) -> None:
    if not any(config.DATA_HIST_ENS.glob("*.csv")):
        print("  keine Ensemble-Rohdaten (weather.py ensemble zuerst laufen lassen)",
              file=sys.stderr)
        return
    g = grid.cells()
    for method in methods:
        out_dir = config.DATA_WEATHER_ZONE / method
        out_dir.mkdir(parents=True, exist_ok=True)
        for area in config.AGG_AREAS:
            df = _aggregate_ens_area(area, method, g)
            if df is None:
                continue
            config.atomic_to_csv(df, out_dir / f"ens_{area}.csv", index=False)
            print(f"  [{method}/{area}] Ensemble {len(df)} Zeilen "
                  f"({df.member.nunique()} Member) -> ens_{area}.csv", file=sys.stderr)


# --- Diff-Analyse: pop vs. centroid -----------------------------------------

def analyse_diff() -> pd.DataFrame:
    """Vergleich der beiden Aggregationen je Gebiet & Variable: MAE, RMSE, Korr.
    -> data/analysis/weather_agg_diff.csv (+ gedruckte Zusammenfassung)."""
    config.DATA_ANALYSIS.mkdir(parents=True, exist_ok=True)
    rows = []
    for area in config.AGG_AREAS:
        pop_f = config.DATA_WEATHER_ZONE / "pop" / f"{area}.csv"
        cen_f = config.DATA_WEATHER_ZONE / "centroid" / f"{area}.csv"
        if not (pop_f.exists() and cen_f.exists()):
            continue
        a = pd.read_csv(pop_f, parse_dates=["date"])
        b = pd.read_csv(cen_f, parse_dates=["date"])
        m = a.merge(b, on="date", suffixes=("_pop", "_cen"))
        for v in config.WEATHER_VARS:
            d = m[f"{v}_pop"] - m[f"{v}_cen"]
            rows.append({
                "area": area, "variable": v, "n": len(m),
                "mae": float(d.abs().mean()),
                "rmse": float(np.sqrt((d ** 2).mean())),
                "bias": float(d.mean()),
                "corr": float(m[f"{v}_pop"].corr(m[f"{v}_cen"])),
                "std_pop": float(m[f"{v}_pop"].std()),
            })
    res = pd.DataFrame(rows)
    if res.empty:
        print("  keine aggregierten Reihen fuer die Diff-Analyse gefunden", file=sys.stderr)
        return res
    out = config.DATA_ANALYSIS / "weather_agg_diff.csv"
    config.atomic_to_csv(res, out, index=False)
    print(f"\nDiff pop vs. centroid -> {out}")
    print(res.to_string(index=False,
                        formatters={c: "{:.4f}".format for c in
                                    ["mae", "rmse", "bias", "corr", "std_pop"]}))
    return res


def main() -> None:
    p = argparse.ArgumentParser(description="Zonen-Wetter aggregieren (pop + centroid)")
    p.add_argument("--ensemble", action="store_true", help="zusaetzlich Ensemble aggregieren")
    a = p.parse_args()
    aggregate_point()
    if a.ensemble:
        aggregate_ensemble()
    analyse_diff()


if __name__ == "__main__":
    main()
