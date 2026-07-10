"""Rollierende Tages-Schleife (Walk-Forward), zero-shot.

Fuer jeden Liefertag D:
  cutoff  = D-1 09:00 UTC (Gate)
  Kontext = letzte ctx_days*96 Steps bis cutoff (leckage-sicher)
  Horizont= cutoff+15min .. D 23:45 (155 Steps); bewertet werden die 96 Steps von D.
Je (Config, Gebiet) werden die 21 Chronos-2-Samples der 96 Liefer-Steps + Ist-Last +
ENTSO-E-Benchmark gespeichert -> out/{config}.parquet (Long-Format, s0..s20).
DE-LU wird nicht hier, sondern in score.py aus den Zonen zusammengesetzt.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import configs, data, model

FREQ = "15min"
STEPS_PER_DAY = 96


def _utc(x):
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def eval_days(start, end, cadence=1):
    return pd.date_range(_utc(start).normalize(), _utc(end).normalize(), freq=f"{cadence}D")


def _day_geometry(D: pd.Timestamp, gate_hour: int):
    cutoff = D - pd.Timedelta(days=1) + pd.Timedelta(hours=gate_hour)
    end = D + pd.Timedelta(hours=23, minutes=45)
    fut_idx = pd.date_range(cutoff + pd.Timedelta(FREQ), end, freq=FREQ)
    deliv = fut_idx.normalize() == D
    return cutoff, fut_idx, deliv


def _rows(cfg, area, ts, samp, actual, bench):
    """samp: (21, 96) -> Long-Format-Zeilen."""
    n = samp.shape[1]
    d = {"config": cfg.name, "area": area, "timestamp": ts[:n],
         "actual": actual, "benchmark": bench}
    for s in range(samp.shape[0]):
        d[f"s{s}"] = samp[s]
    return pd.DataFrame(d)


def run_config(cfg: configs.Config, start, end, cadence=1, ctx_days=63,
               gate_hour=9, out_dir=None):
    from pathlib import Path
    out_dir = Path(out_dir) if out_dir else Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx_steps = ctx_days * STEPS_PER_DAY

    areas = cfg.areas
    frames = {}                                    # area -> (df, tcols, known, past)
    for a in areas:
        try:
            frames[a] = model.area_frame(cfg, a)
        except FileNotFoundError as e:
            print(f"  [skip] {cfg.name}/{a}: {e}", file=sys.stderr)
    if not frames:
        print(f"  [skip] {cfg.name}: keine Daten", file=sys.stderr)
        return None
    actuals = {a: data.target(a) for a in frames}
    benches = {a: data.benchmark(a) for a in frames}
    # Ground-Truth-Bezug fuer joint: DE-LU aus Zonen -> in score.py
    load_pos = {a: 0 for a in frames}              # Load-Variate steht je Zonenblock vorn

    rows = []
    days = eval_days(start, end, cadence)
    n_ok = 0
    for D in days:
        cutoff, fut_idx, deliv = _day_geometry(D, gate_hour)
        ts = fut_idx[deliv]
        if cfg.gran in ("whole", "indep"):
            tasks, tas = [], []
            for a, (df, tcols, known, past) in frames.items():
                t = model.build_task(df, cutoff, tcols, known, past, ctx_steps, fut_idx)
                if t is not None:
                    tasks.append(t); tas.append(a)
            if not tasks:
                continue
            outs = model.predict(tasks, len(fut_idx))
            for a, o in zip(tas, outs):
                samp = np.clip(o[0][:, deliv], 0, None)          # variate0=load, (21,96)
                rows.append(_row_for(cfg, a, ts, samp, actuals[a], benches[a]))
            n_ok += 1
        else:                                                    # joint
            task, zorder = model.build_joint_task(frames, cutoff, ctx_steps, fut_idx)
            if task is None:
                continue
            arr = model.predict([task], len(fut_idx))[0]         # (n_var,21,hor)
            pos = 0
            for a in zorder:
                nvar = len(frames[a][1])                         # Zonenblock-Groesse
                samp = np.clip(arr[pos][:, deliv], 0, None)      # Load = erste Variate
                rows.append(_row_for(cfg, a, ts, samp, actuals[a], benches[a]))
                pos += nvar
            n_ok += 1

    if not rows:
        print(f"  {cfg.name}: 0 Tage (Daten decken den Bereich nicht)", file=sys.stderr)
        return None
    df_out = pd.concat(rows, ignore_index=True)
    path = out_dir / f"{cfg.name}.parquet"
    df_out.to_parquet(path, index=False)
    print(f"  {cfg.name}: {n_ok} Tage, {len(df_out)} Zeilen -> {path.name}", file=sys.stderr)
    return path


def _row_for(cfg, area, ts, samp, actual_s, bench_s):
    a = actual_s.reindex(ts).to_numpy()
    b = bench_s.reindex(ts).to_numpy() if bench_s is not None else np.full(len(ts), np.nan)
    return _rows(cfg, area, ts, samp, a, b)


# --- Finetuned Walk-Forward (LoRA auf 2023, dann Eval 2024-2026) -------------

def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _refit_periods(start, end, refit_months):
    """Rolling-Refit-Fenster: Liste (train_end, eval_start, eval_end).
    Bei train_end wird auf allen Daten < train_end (expanding window) gefittet."""
    start = pd.Timestamp(start, tz="UTC").normalize()
    end = pd.Timestamp(end, tz="UTC").normalize()
    bounds = list(pd.date_range(start.replace(day=1), end, freq=f"{refit_months}MS"))
    if not bounds or bounds[0] > start:
        bounds = [start] + bounds
    out = []
    for i, b in enumerate(bounds):
        nb = bounds[i + 1] if i + 1 < len(bounds) else end + pd.Timedelta(days=1)
        es = max(b, start)
        if es < nb:
            out.append((max(b, start), es, nb))          # train_end=b (>=start), eval [es,nb)
    return out


def run_config_ft(cfg: configs.Config, start, end, cadence=1, ctx_days=63,
                  gate_hour=9, num_steps=300, refit_months=3, chunk=64, out_dir=None):
    """Expanding-window continual LoRA-Finetuning: alle `refit_months` neu fitten (auf
    Daten < Fensterstart, also 2023 -> +2024 -> +2025 ...), dann das Fenster
    out-of-sample walk-forward. Leckage-frei. -> out_ft/{config}.parquet."""
    from pathlib import Path
    out_dir = Path(out_dir) if out_dir else Path(__file__).resolve().parent / "out_ft"
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx_steps = ctx_days * STEPS_PER_DAY
    HOR = 155

    frames = {}
    for a in cfg.areas:
        try:
            frames[a] = model.area_frame(cfg, a)
        except FileNotFoundError:
            pass
    if not frames:
        return None
    actuals = {a: data.target(a) for a in frames}
    benches = {a: data.benchmark(a) for a in frames}
    periods = _refit_periods(start, end, refit_months)
    rows, n_fit = [], 0

    for train_end, es, ee in periods:
        days = eval_days(es, ee - pd.Timedelta(days=1), cadence)
        geoms = [_day_geometry(D, gate_hour) for D in days]
        if cfg.gran in ("whole", "indep"):
            for a, (df, tcols, known, past) in frames.items():
                fit_task = model.build_fit_task(df, train_end, tcols, known, past)
                if fit_task is None:
                    continue
                ft = model.fit_lora([fit_task], HOR, ctx_steps, num_steps=num_steps)
                n_fit += 1
                dtasks, metas = [], []
                for (cutoff, fut_idx, deliv) in geoms:
                    t = model.build_task(df, cutoff, tcols, known, past, ctx_steps, fut_idx)
                    if t is not None and len(fut_idx) == HOR:
                        dtasks.append(t); metas.append((fut_idx[deliv], deliv))
                for tk, mt in zip(_chunks(dtasks, chunk), _chunks(metas, chunk)):
                    for o, (ts, deliv) in zip(model.predict(tk, HOR, batch_size=chunk, model_pipe=ft), mt):
                        rows.append(_row_for(cfg, a, ts, np.clip(o[0][:, deliv], 0, None),
                                             actuals[a], benches[a]))
                del ft; _free_gpu()
        else:                                            # joint
            fit_task = model.build_joint_fit_task(frames, train_end)
            if fit_task is None:
                continue
            ft = model.fit_lora([fit_task], HOR, ctx_steps, num_steps=num_steps)
            n_fit += 1
            for (cutoff, fut_idx, deliv) in geoms:
                task, zo = model.build_joint_task(frames, cutoff, ctx_steps, fut_idx)
                if task is None or len(fut_idx) != HOR:
                    continue
                arr = model.predict([task], HOR, model_pipe=ft)[0]
                pos, ts = 0, fut_idx[deliv]
                for a in zo:
                    nvar = len(frames[a][1])
                    rows.append(_row_for(cfg, a, ts, np.clip(arr[pos][:, deliv], 0, None),
                                         actuals[a], benches[a]))
                    pos += nvar
            del ft; _free_gpu()

    if not rows:
        print(f"  {cfg.name}(ft): 0 Zeilen", file=sys.stderr)
        return None
    path = out_dir / f"{cfg.name}.parquet"
    dfo = pd.concat(rows, ignore_index=True)
    dfo.to_parquet(path, index=False)
    print(f"  {cfg.name}(ft): {n_fit} fits, {len(dfo)} rows -> {path.name}", file=sys.stderr)
    return path


def _free_gpu():
    try:
        import gc
        import torch
        gc.collect(); torch.cuda.empty_cache()
    except Exception:
        pass
