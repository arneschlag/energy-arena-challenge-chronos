"""Metrik-Aggregation aus den gespeicherten Samples (+ DE-LU-Zusammenbau + Benchmark).

Liest out/{config}.parquet (Long-Format je Zone), setzt fuer indep/joint DE-LU aus den
Zonen zusammen (sampleweise Summe; bei joint korrekt gekoppelt, bei indep komonoton
approximiert — Hinweis im Output), bewertet Punkt/Quantil/Ensemble je Zeitraum und
schreibt eine tidy-Tabelle results.csv. Zusaetzlich der ENTSO-E-Benchmark (Median-only).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from loaders import config as lc
from . import configs, data, metrics

OUT = Path(__file__).resolve().parent / "out"
SCOLS = [f"s{i}" for i in range(21)]


def _period(ts: pd.Series) -> pd.Series:
    y = ts.dt.year
    p = y.astype(str)
    p = p.where(y != 2026, "2026H1")
    return p


def _delu_rows(df: pd.DataFrame) -> pd.DataFrame:
    """indep/joint: DE-LU = sampleweise Zonen-Summe je Zeitpunkt (alle Zonen noetig)."""
    zones = [z for z in lc.ZONES + [lc.LU] if z in set(df.area)]
    sub = df[df.area.isin(zones)]
    n_zone = sub.groupby("timestamp")["area"].transform("nunique")
    sub = sub[n_zone == len(zones)]
    if sub.empty:
        return pd.DataFrame()
    agg = sub.groupby("timestamp")[SCOLS].sum().reset_index()
    agg["area"] = lc.DELU
    agg["config"] = df.config.iloc[0]
    at = data.target(lc.DELU).reindex(agg.timestamp).to_numpy()
    bt = data.benchmark(lc.DELU)
    agg["actual"] = at
    agg["benchmark"] = bt.reindex(agg.timestamp).to_numpy() if bt is not None else np.nan
    return agg


def _group_arrays(g: pd.DataFrame):
    """y (n,), samples (21,n), days [(y_day, samp_day)] fuer den Energy Score."""
    g = g.sort_values("timestamp")
    y = g["actual"].to_numpy(float)
    samp = g[SCOLS].to_numpy(float).T                    # (21, n)
    days = []
    for _, d in g.groupby(g.timestamp.dt.normalize()):
        days.append((d["actual"].to_numpy(float), d[SCOLS].to_numpy(float).T))
    return y, samp, days


def _score_group(g: pd.DataFrame) -> dict:
    y, samp, days = _group_arrays(g)
    out = {}
    for k, v in metrics.score_point(y, samp).items():
        out[("point", k)] = v
    for k, v in metrics.score_quantile(y, samp).items():
        out[("quantile", k)] = v
    for k, v in metrics.score_ensemble(y, samp, days).items():
        out[("ensemble", k)] = v
    return out


def score_config(name: str, out_dir: Path = OUT) -> pd.DataFrame:
    path = Path(out_dir) / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    cfg = configs.get(name)
    if cfg.gran in ("indep", "joint"):
        delu = _delu_rows(df)
        if not delu.empty:
            df = pd.concat([df, delu], ignore_index=True)
    df = df.dropna(subset=["actual"])
    df["period"] = _period(df["timestamp"])
    recs = []
    for (area, period), g in df.groupby(["area", "period"]):
        for (otype, metric), val in _score_group(g).items():
            recs.append(dict(config=name, area=area, period=period,
                             output=otype, metric=metric, value=val))
        # overall zusaetzlich
    for area, g in df.groupby("area"):
        for (otype, metric), val in _score_group(g).items():
            recs.append(dict(config=name, area=area, period="overall",
                             output=otype, metric=metric, value=val))
    return pd.DataFrame(recs)


def score_benchmark() -> pd.DataFrame:
    """ENTSO-E Day-ahead-Prognose (Median-only) je Gebiet & Zeitraum."""
    recs = []
    for area in lc.AGG_AREAS:
        b = data.benchmark(area)
        y = data.target(area)
        if b is None:
            continue
        idx = b.index.intersection(y.index)
        d = pd.DataFrame({"actual": y.reindex(idx), "pred": b.reindex(idx)}, index=idx).dropna()
        d = d[d.index >= pd.Timestamp("2024-01-01", tz="UTC")]
        d["period"] = _period(d.index.to_series())
        for period, g in list(d.groupby("period")) + [("overall", d)]:
            yy, pp = g["actual"].to_numpy(), g["pred"].to_numpy()
            for otype in ("point", "quantile"):
                vals = {"rmse": metrics.rmse(yy, pp), "r2": metrics.r2(yy, pp),
                        "mae": metrics.mae(yy, pp)} if otype == "point" else \
                       {"wis": metrics.wis_median_only(yy, pp), "mae_median": metrics.mae(yy, pp)}
                for m, v in vals.items():
                    recs.append(dict(config="ENTSOE", area=area, period=period,
                                     output=otype, metric=m, value=v))
    return pd.DataFrame(recs)


def score_all(out_csv=None, out_dir: Path = OUT) -> pd.DataFrame:
    out_dir = Path(out_dir)
    parts = []
    for p in sorted(out_dir.glob("*.parquet")):
        if "__" in p.stem:                   # Zonen-Partial (ft_run_config) -> ueberspringen
            continue
        r = score_config(p.stem, out_dir)
        if not r.empty:
            parts.append(r)
            print(f"  scored {p.stem}: {len(r)} rows", file=sys.stderr)
    parts.append(score_benchmark())
    res = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out_csv = Path(out_csv) if out_csv else out_dir / "results.csv"
    res.to_csv(out_csv, index=False)
    print(f"-> {out_csv} ({len(res)} rows)", file=sys.stderr)
    return res
