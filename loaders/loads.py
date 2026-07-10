"""QUELLE 2/3 — ENTSO-E Last: Ist-Last (Ziel) + Day-ahead-Prognose (Benchmark).

Fuer die 4 ÜNB-Zonen UND die kombinierte Gebotszone de_lu:
  Ist-Last          query_load                      -> data/loads/{zone}_{jahr}.csv
  Day-ahead-Prognose query_load_forecast(A01)        -> data/load_forecast/{zone}_{jahr}.csv

de_lu ist die Ground-Truth des Gesamtmarkts (inkl. Luxemburg — Luxemburg hat keine
eigene Gebotszone). reconcile_delu() prueft empirisch, wie gut sich die 4 Zonen zur
DE-LU-Last summieren (Luxemburg-Strategie, siehe README).

Aufruf:
    export ENTSOE_API_KEY=xxxx-...
    python -m loaders.loads
    python -m loaders.loads --years 2026
    python -m loaders.loads --reconcile          # nur Reconciliation-Diagnose
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from . import config

SLEEP = 0.5


def _write_query(df: pd.DataFrame, valcol: str, tso: str, out_dir, year: int) -> int:
    """entsoe-py-Ergebnis normalisieren (UTC) und als {tso}_{jahr}.csv ablegen."""
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    df = df.reset_index()
    df.columns = ["date" if i == 0 else valcol for i in range(len(df.columns))]
    df.insert(0, "tso", tso)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{tso}_{year}.csv", index=False)
    return len(df)


def download(years: list[int], token: str, actual=True, forecast=True) -> None:
    from entsoe import EntsoePandasClient
    client = EntsoePandasClient(api_key=token)
    for tso, area in config.ENTSOE_AREA.items():
        for year in years:
            start, end = config.year_bounds_ts(year)
            if actual:
                out = config.DATA_LOADS / f"{tso}_{year}.csv"
                if not out.exists():
                    try:
                        df = client.query_load(area, start=start, end=end)
                        n = _write_query(df, "actual_load_mw", tso, config.DATA_LOADS, year)
                        print(f"  Ist  {tso} {year}: {n} Zeilen", file=sys.stderr)
                    except Exception as exc:
                        print(f"  FEHLER Ist {tso} {year}: {exc}", file=sys.stderr)
                        time.sleep(2.0)
                    time.sleep(SLEEP)
            if forecast:
                out = config.DATA_FCAST / f"{tso}_{year}.csv"
                if not out.exists():
                    try:
                        df = client.query_load_forecast(area, start=start, end=end,
                                                        process_type="A01")
                        n = _write_query(df, "forecast_load_mw", tso, config.DATA_FCAST, year)
                        print(f"  Prog {tso} {year}: {n} Zeilen", file=sys.stderr)
                    except Exception as exc:
                        print(f"  FEHLER Prog {tso} {year}: {exc}", file=sys.stderr)
                        time.sleep(2.0)
                    time.sleep(SLEEP)


# --- Inkrementelle Aktualisierung (fuer den Scheduler) ----------------------

def _merge_csv(path, df: pd.DataFrame) -> None:
    """Neue Zeilen in eine bestehende {tso}_{jahr}.csv mergen (dedup nach date)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    if path.exists():
        old = pd.read_csv(path)
        old["date"] = pd.to_datetime(old["date"], utc=True)
        df = pd.concat([old, df], ignore_index=True)
    df.drop_duplicates("date", keep="last").sort_values("date").to_csv(path, index=False)


def refresh_recent(days: int, token: str) -> None:
    """Letzte `days` Tage Ist-Last + Day-ahead-Prognose fuer alle Gebiete holen
    und in die jeweiligen Jahres-CSVs mergen (idempotent)."""
    from entsoe import EntsoePandasClient
    client = EntsoePandasClient(api_key=token)
    now = pd.Timestamp.now(tz="UTC")
    start = (now - pd.Timedelta(days=days)).tz_convert("Europe/Brussels")
    end_a = now.tz_convert("Europe/Brussels")
    end_f = (now + pd.Timedelta(days=2)).tz_convert("Europe/Brussels")
    for tso, area in config.ENTSOE_AREA.items():
        try:
            a = client.query_load(area, start=start, end=end_a)
            a.index = pd.DatetimeIndex(a.index).tz_convert("UTC")
            da = a.reset_index(); da.columns = ["date", "actual_load_mw"]; da.insert(0, "tso", tso)
            for y, gy in da.groupby(da.date.dt.year):
                config.DATA_LOADS.mkdir(parents=True, exist_ok=True)
                _merge_csv(config.DATA_LOADS / f"{tso}_{y}.csv", gy)
            f = client.query_load_forecast(area, start=start, end=end_f, process_type="A01")
            f.index = pd.DatetimeIndex(f.index).tz_convert("UTC")
            df = f.reset_index(); df.columns = ["date", "forecast_load_mw"]; df.insert(0, "tso", tso)
            for y, gy in df.groupby(df.date.dt.year):
                config.DATA_FCAST.mkdir(parents=True, exist_ok=True)
                _merge_csv(config.DATA_FCAST / f"{tso}_{y}.csv", gy)
            print(f"  refresh {tso}: {len(a)} Ist / {len(f)} Prognose", file=sys.stderr)
        except Exception as exc:
            print(f"  FEHLER refresh {tso}: {exc}", file=sys.stderr)
            time.sleep(2.0)


# --- Luecken erkennen und gezielt nachfordern -------------------------------

def _read_actual(tso: str) -> pd.Series:
    # {tso}_2*.csv trifft nur Jahres-Dateien (schliesst z.B. lu_derived_* aus)
    parts = [pd.read_csv(f, usecols=["date", "actual_load_mw"])
             for f in sorted(config.DATA_LOADS.glob(f"{tso}_2*.csv"))]
    if not parts:
        return pd.Series(dtype="float64")
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.drop_duplicates("date").set_index("date")["actual_load_mw"].sort_index()


def _missing_blocks(idx: pd.DatetimeIndex, until: pd.Timestamp,
                    pad: pd.Timedelta) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Fehlende 15-Min-Slots zwischen idx.min() und `until` als (start,end)-Bloecke
    (je mit Rand `pad` fuer die Requery)."""
    if len(idx) == 0:
        return []
    full = pd.date_range(idx.min(), until, freq="15min", tz="UTC")
    missing = full.difference(idx)
    if len(missing) == 0:
        return []
    ms = missing.to_series()
    blk = (ms.diff() != pd.Timedelta("15min")).cumsum()
    return [(g.min() - pad, g.max() + pad) for _, g in ms.groupby(blk)]


def fill_gaps(token: str, until: pd.Timestamp | None = None,
              pad: str = "3h", forecast=True) -> None:
    """Fehlende Ist-Last-Slots (und optional Prognose) bis `until` (Default jetzt)
    je Gebiet erkennen und NUR die Luecken-Bereiche bei ENTSO-E neu ziehen + mergen.
    Schliesst sowohl kleine interne Loecher als auch den fehlenden aktuellen Rand."""
    from entsoe import EntsoePandasClient
    client = EntsoePandasClient(api_key=token)
    now = until or pd.Timestamp.now(tz="UTC")
    padt = pd.Timedelta(pad)
    for tso, area in config.ENTSOE_AREA.items():
        s = _read_actual(tso)
        blocks = _missing_blocks(s.index, now, padt)
        if not blocks:
            print(f"  {tso}: keine Luecken", file=sys.stderr)
            continue
        slots = sum(int((e - st) / pd.Timedelta("15min")) for st, e in blocks)
        print(f"  {tso}: {len(blocks)} Luecken-Bloecke (~{slots} Slots) -> nachfordern",
              file=sys.stderr)
        for st, en in blocks:
            st_b = st.tz_convert("Europe/Brussels")
            en_b = en.tz_convert("Europe/Brussels")
            try:
                a = client.query_load(area, start=st_b, end=en_b)
                a.index = pd.DatetimeIndex(a.index).tz_convert("UTC")
                da = a.reset_index(); da.columns = ["date", "actual_load_mw"]
                da.insert(0, "tso", tso)
                for y, gy in da.groupby(da.date.dt.year):
                    _merge_csv(config.DATA_LOADS / f"{tso}_{y}.csv", gy)
                if forecast:
                    f = client.query_load_forecast(area, start=st_b, end=en_b, process_type="A01")
                    f.index = pd.DatetimeIndex(f.index).tz_convert("UTC")
                    fd = f.reset_index(); fd.columns = ["date", "forecast_load_mw"]
                    fd.insert(0, "tso", tso)
                    for y, gy in fd.groupby(fd.date.dt.year):
                        _merge_csv(config.DATA_FCAST / f"{tso}_{y}.csv", gy)
            except Exception as exc:
                print(f"    FEHLER {tso} {st.date()}..{en.date()}: {exc}", file=sys.stderr)
                time.sleep(2.0)


def reconcile_delu() -> pd.DataFrame:
    """r(t) = de_lu_ist - Summe(4 Zonen) ueber die gemeinsame Historie.
    Zeigt, ob die Zonensumme die DE-LU-Gebotszone reproduziert (Luxemburg-Frage).
    -> data/analysis/delu_reconciliation.csv"""
    delu = _read_actual("de_lu")
    if delu.empty:
        print("  de_lu-Last fehlt — zuerst laden", file=sys.stderr)
        return pd.DataFrame()
    zone_sum = None
    for z in config.ZONES:
        s = _read_actual(z)
        zone_sum = s if zone_sum is None else zone_sum.add(s, fill_value=np.nan)
    both = pd.DataFrame({"de_lu": delu, "zone_sum": zone_sum}).dropna()
    both["residual_mw"] = both["de_lu"] - both["zone_sum"]
    config.DATA_ANALYSIS.mkdir(parents=True, exist_ok=True)
    out = config.DATA_ANALYSIS / "delu_reconciliation.csv"
    both.reset_index().to_csv(out, index=False)

    r = both["residual_mw"]
    pct = 100 * r.mean() / both["de_lu"].mean()
    print(f"\nDE-LU Reconciliation (de_lu - Summe 4 Zonen) -> {out}")
    print(f"  n={len(both)}  mean={r.mean():.1f} MW  median={r.median():.1f} MW  "
          f"std={r.std():.1f} MW  ({pct:+.2f}% der DE-LU-Last)")
    if abs(pct) < 0.5:
        print("  => Summe reproduziert DE-LU (Luxemburg in Amprion enthalten): keine Korrektur noetig.")
    elif r.std() < abs(r.mean()):
        print(f"  => kleiner ~konstanter Offset: als feste Korrektur (+{r.median():.0f} MW) in die DE-LU-Kombi.")
    else:
        print("  => grosser/zeitvariabler Rest: de_lu direkt als eigenes Ziel nutzen (nicht ueber Summe).")
    return both


def derive_lu() -> None:
    """Abgeleitete Luxemburg-Last als Kontrolle zur direkten ENTSO-E-LU-Reihe:
    lu ~= de_lu - Summe(4 dt. Zonen) (Reconciliation-Rest), je Jahr abgelegt
    -> data/loads/lu_derived_{jahr}.csv (tso,date,actual_load_mw)."""
    both = reconcile_delu()
    if both.empty:
        return
    df = both["residual_mw"].reset_index()
    df.columns = ["date", "actual_load_mw"]
    df.insert(0, "tso", "lu_derived")
    for y, gy in df.groupby(df.date.dt.year):
        if y not in config.DEFAULT_YEARS:      # Rand-Zeitstempel (z.B. 31.12. 23:00) ueberspringen
            continue
        config.DATA_LOADS.mkdir(parents=True, exist_ok=True)
        gy.to_csv(config.DATA_LOADS / f"lu_derived_{y}.csv", index=False)
    print(f"  lu_derived: {len(df)} Zeilen -> data/loads/lu_derived_*.csv", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description="ENTSO-E Ist-Last + Day-ahead-Prognose")
    p.add_argument("--years", type=int, nargs="+", default=config.DEFAULT_YEARS)
    p.add_argument("--token", default=None)
    p.add_argument("--reconcile", action="store_true",
                   help="nur die DE-LU-Reconciliation-Diagnose (kein Download)")
    p.add_argument("--fill-gaps", action="store_true",
                   help="fehlende Slots bis jetzt erkennen und gezielt nachfordern")
    a = p.parse_args()

    if a.reconcile:
        reconcile_delu()
        derive_lu()
        return

    token = a.token or config.entsoe_key()
    if not token:
        sys.exit("Kein ENTSO-E-Token. Setze ENTSOE_API_KEY (.env) oder --token.\n"
                 "Kostenlos: https://transparency.entsoe.eu/ -> Web Api Security Token.")
    if a.fill_gaps:
        fill_gaps(token)
    else:
        download(sorted(set(a.years)), token)
    derive_lu()


if __name__ == "__main__":
    main()
