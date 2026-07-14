"""Chronos-2-Pipeline, Task-Bau und Vorhersage.

Braucht `chronos` + GPU (nur im chronos-Container). Der chronos-Import ist lazy,
damit configs/data/metrics ohne chronos importierbar/testbar bleiben.

Task-Schema (Chronos-2):
  {"target": (hist,) oder (n_var, hist),
   "past_covariates":   {name: (hist,)},        # known-future (Kontextteil) + past-only
   "future_covariates": {name: (horizon,)}}     # nur known-future (Teilmenge von past)
predict([task], prediction_length=H) -> Liste von (n_var, 21 samples, H).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import configs, data

FREQ = "15min"
MODEL = "amazon/chronos-2"
_PIPE = None

# Bundesweite DE-Feiertage 2023--2026 + de-facto lastreduzierte Tage (24./31.12).
# Als known-future-Covariate (Feiertage sind im Voraus bekannt).
_DE_HOLIDAYS = frozenset({
    "2023-01-01", "2023-04-07", "2023-04-10", "2023-05-01", "2023-05-18", "2023-05-29",
    "2023-10-03", "2023-12-24", "2023-12-25", "2023-12-26", "2023-12-31",
    "2024-01-01", "2024-03-29", "2024-04-01", "2024-05-01", "2024-05-09", "2024-05-20",
    "2024-10-03", "2024-12-24", "2024-12-25", "2024-12-26", "2024-12-31",
    "2025-01-01", "2025-04-18", "2025-04-21", "2025-05-01", "2025-05-29", "2025-06-09",
    "2025-10-03", "2025-12-24", "2025-12-25", "2025-12-26", "2025-12-31",
    "2026-01-01", "2026-04-03", "2026-04-06", "2026-05-01", "2026-05-14", "2026-05-25",
    "2026-10-03", "2026-12-24", "2026-12-25", "2026-12-26", "2026-12-31",
})


def _calendar_feature(index, name):
    """Kalender-Covariate aus dem (UTC-)Zeitindex, aligned zur UTC-Liefertag-Logik."""
    idx = pd.DatetimeIndex(index)
    if name in ("holiday", "bridge", "pre_holiday", "post_holiday", "xmas"):
        days = idx.normalize()
        one = pd.Timedelta(days=1)
        hol = lambda d: d.strftime("%Y-%m-%d") in _DE_HOLIDAYS
        vals = {}
        for d in days.unique():
            prev, nxt = d - one, d + one
            if name == "holiday":
                v = hol(d)
            elif name == "pre_holiday":                # Tag vor Feiertag (selbst keiner)
                v = not hol(d) and hol(nxt)
            elif name == "post_holiday":               # Tag nach Feiertag (selbst keiner)
                v = not hol(d) and hol(prev)
            elif name == "bridge":                     # Brueckentag: Werktag zwischen
                v = (d.dayofweek < 5 and not hol(d)    # Feiertag und Wochenende/Feiertag
                     and ((hol(prev) and (nxt.dayofweek >= 5 or hol(nxt)))
                          or (hol(nxt) and (prev.dayofweek >= 5 or hol(prev)))))
            else:                                      # xmas: 24.12. bis 01.01.
                v = (d.month == 12 and d.day >= 24) or (d.month == 1 and d.day == 1)
            vals[d] = 1.0 if v else 0.0
        return np.array([vals[d] for d in days], dtype="float32")
    if name == "weekend":
        return (idx.dayofweek >= 5).astype("float32")
    if name in ("dow_sin", "dow_cos"):                 # Wochentag zyklisch (7-Tage-Periode)
        dow = idx.dayofweek.to_numpy()
        fn = np.sin if name == "dow_sin" else np.cos
        return fn(2 * np.pi * dow / 7).astype("float32")
    if name in ("doy_sin", "doy_cos"):                 # Tag im Jahr zyklisch (Jahres-Saison)
        doy = idx.dayofyear.to_numpy()
        fn = np.sin if name == "doy_sin" else np.cos
        return fn(2 * np.pi * doy / 365.25).astype("float32")
    raise ValueError(f"unbekanntes Kalender-Feature {name!r}")


def pipe():
    """Lazy-Singleton der Chronos-2-Pipeline (Device via env DEVICE, default cuda)."""
    global _PIPE
    if _PIPE is None:
        from chronos import Chronos2Pipeline
        _PIPE = Chronos2Pipeline.from_pretrained(
            MODEL, device_map=os.environ.get("DEVICE", "cuda"))
    return _PIPE


# --- Spalten je (Config, Gebiet) --------------------------------------------

def area_columns(cfg: configs.Config, area: str):
    """Verfuegbare Aux filtern und Spaltengruppen liefern:
    (weather_source, target_aux, known_aux, past_aux)."""
    ta = [a for a in cfg.aux_target if data.has_aux(area, a)]
    ka = [a for a in cfg.aux_known if data.has_aux(area, a)]
    pa = [a for a in cfg.aux_past if data.has_aux(area, a)]
    return cfg.weather, ta, ka, pa


def area_frame(cfg: configs.Config, area: str, include_future: bool = False):
    """DataFrame + Spaltenlisten fuer ein Gebiet.
    Returns (df, target_cols, known_cov, past_cov)."""
    w, ta, ka, pa = area_columns(cfg, area)
    df = data.frame(area, w, tuple(ta + ka + pa), include_future=include_future)
    target_cols = ["target"] + [f"aux_{a}" for a in ta]
    weather_cols = [c for c in df.columns if c != "target" and not c.startswith("aux_")]
    known_cov = weather_cols + [f"aux_{a}" for a in ka]
    past_cov = [f"aux_{a}" for a in pa]
    for name in cfg.calendar:                          # known-future Kalender-Features
        col = f"cal_{name}"
        df[col] = _calendar_feature(df.index, name)
        known_cov.append(col)
    # FT_PAD_COVARIATES (env): n konstante Null-Covariaten anhaengen. Informationsfrei;
    # einziger Zweck: die Covariaten-ANZAHL verschieben, wenn eine Config einen kaputten
    # ROCm-Kernel-Bucket trifft (deterministischer HIP-Crash im LoRA-Backward).
    for i in range(int(os.environ.get("FT_PAD_COVARIATES", "0") or 0)):
        col = f"cal_pad{i}"
        df[col] = np.float32(0.0)
        known_cov.append(col)
    return df, target_cols, known_cov, past_cov


# --- Task-Bau ----------------------------------------------------------------

def build_task(df: pd.DataFrame, cutoff: pd.Timestamp, target_cols, known_cov, past_cov,
               ctx_steps: int, fut_idx: pd.DatetimeIndex):
    """Ein Chronos-2-Task fuer einen Liefertag. None, wenn Kontext/Zukunft fehlt.
    Leckage-sicher: target nur bis cutoff; Zukunft nur Covariaten."""
    ctx = df.loc[:cutoff].tail(ctx_steps)
    if len(ctx) < 96:                                   # zu wenig Kontext
        return None
    need_fut = known_cov                                # future_covariates muessen existieren
    if need_fut and not fut_idx.isin(df.index).all():
        return None
    fut = df.reindex(fut_idx)
    tgt = ctx[target_cols].to_numpy("float32")
    target = tgt[:, 0] if tgt.shape[1] == 1 else tgt.T   # (hist,) | (n_var, hist)
    task = {"target": target}
    past = {c: ctx[c].to_numpy("float32") for c in known_cov + past_cov}
    if past:
        task["past_covariates"] = past
    if known_cov:
        task["future_covariates"] = {c: fut[c].to_numpy("float32") for c in known_cov}
    return task


def build_joint_task(frames: dict, cutoff, ctx_steps, fut_idx):
    """Ein multivariates Task ueber mehrere Zonen (G3). frames: {area: (df, tcols, known,
    past)}. Ziel = alle Zonen-Lasten (+ evtl. Aux-Ziele je Zone) gestapelt; Covariaten
    zonen-praefixiert. Returns (task, zone_order) oder (None, []).
    Alle Zonen werden auf den gemeinsamen 15-min-Index innergejoint."""
    # gemeinsamer Kontext-Index
    common = None
    for area, (df, *_ ) in frames.items():
        idx = df.loc[:cutoff].tail(ctx_steps).index
        common = idx if common is None else common.intersection(idx)
    if common is None or len(common) < 96:
        return None, []
    zone_order = list(frames.keys())
    tgt_stack, past, future = [], {}, {}
    for area in zone_order:
        df, tcols, known, pastc = frames[area]
        if not fut_idx.isin(df.index).all() and known:
            return None, []
        ctx = df.loc[common]
        for tc in tcols:                                # load + aux-Ziele je Zone
            tgt_stack.append(ctx[tc].to_numpy("float32"))
        fut = df.reindex(fut_idx)
        for c in known:
            past[f"{area}__{c}"] = ctx[c].to_numpy("float32")
            future[f"{area}__{c}"] = fut[c].to_numpy("float32")
        for c in pastc:
            past[f"{area}__{c}"] = ctx[c].to_numpy("float32")
    task = {"target": np.stack(tgt_stack, axis=0)}      # (n_var, hist)
    if past:
        task["past_covariates"] = past
    if future:
        task["future_covariates"] = future
    return task, zone_order


# --- Vorhersage --------------------------------------------------------------

def predict(tasks, horizon, batch_size=256, model_pipe=None):
    """Liste von Tasks -> Liste von (n_var, 21, horizon)-Arrays.
    model_pipe: optional eine finetunte Pipeline (sonst die Zero-Shot-Basis)."""
    p = model_pipe if model_pipe is not None else pipe()
    outs = p.predict(tasks, prediction_length=horizon, batch_size=batch_size)
    return [o.float().cpu().numpy() for o in outs]


# --- Finetuning (LoRA auf 2023) ---------------------------------------------

def build_fit_task(df: pd.DataFrame, train_end: pd.Timestamp, target_cols, known_cov, past_cov,
                   train_start=None):
    """Fit-Task aus dem Trainingszeitraum (< train_end). Ganze Reihe; Chronos-2 sampelt
    intern Fenster. future_covariates=None markiert known-future.
    train_start: optionaler Fensterbeginn (Rolling Window statt Expanding)."""
    tr = df.loc[train_start:train_end] if train_start is not None else df.loc[:train_end]
    if len(tr) < 96 * 70:                               # < ~70 Tage -> zu wenig
        return None
    tgt = tr[target_cols].to_numpy("float32")
    task = {"target": tgt[:, 0] if tgt.shape[1] == 1 else tgt.T}
    past = {c: tr[c].to_numpy("float32") for c in known_cov + past_cov}
    if past:
        task["past_covariates"] = past
    if known_cov:
        task["future_covariates"] = {c: None for c in known_cov}
    return task


def build_joint_fit_task(frames: dict, train_end, train_start=None):
    common = None
    for _, (df, *_ ) in frames.items():
        idx = (df.loc[train_start:train_end] if train_start is not None
               else df.loc[:train_end]).index
        common = idx if common is None else common.intersection(idx)
    if common is None or len(common) < 96 * 70:
        return None
    tgt_stack, past, future = [], {}, {}
    for area, (df, tcols, known, pastc) in frames.items():
        tr = df.loc[common]
        for tc in tcols:
            tgt_stack.append(tr[tc].to_numpy("float32"))
        for c in known:
            past[f"{area}__{c}"] = tr[c].to_numpy("float32")
            future[f"{area}__{c}"] = None
        for c in pastc:
            past[f"{area}__{c}"] = tr[c].to_numpy("float32")
    task = {"target": np.stack(tgt_stack, axis=0)}
    if past:
        task["past_covariates"] = past
    if future:
        task["future_covariates"] = future
    return task


def fit_lora(fit_tasks, horizon, ctx_steps, num_steps=500, lr=1e-5,
             batch_size=8, out_dir=None):
    """LoRA-Finetuning; gibt die finetunte Pipeline zurueck (direkt nutzbar).
    tf32=False fuer ROCm/AMD-Kompatibilitaet.
    FT_BATCH_SIZE (env) uebersteuert batch_size (Auto-Retry bei HIP/OOM: 8->4).
    FT_SEED (env) setzt Trainings-Seed (Adapter-Init, Sampling, Trainer) fuer
    reproduzierbare, aber zwischen Wiederholungslaeufen unterschiedliche Ergebnisse."""
    batch_size = int(os.environ.get("FT_BATCH_SIZE", batch_size))
    lr = float(os.environ.get("FT_LR", lr))              # Stabilitaets-Fallback bei NaN-Loss
    extra = {}
    seed_env = os.environ.get("FT_SEED")
    if seed_env:                                         # leer/None -> Default-Seed
        import torch
        seed = int(seed_env)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        extra["seed"] = seed                             # -> HF TrainingArguments.seed
    return pipe().fit(fit_tasks, prediction_length=horizon, finetune_mode="lora",
                      context_length=ctx_steps, learning_rate=lr, num_steps=num_steps,
                      batch_size=batch_size, output_dir=out_dir,
                      remove_printer_callback=True, tf32=False, **extra)
