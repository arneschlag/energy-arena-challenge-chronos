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


def area_frame(cfg: configs.Config, area: str):
    """DataFrame + Spaltenlisten fuer ein Gebiet.
    Returns (df, target_cols, known_cov, past_cov)."""
    w, ta, ka, pa = area_columns(cfg, area)
    df = data.frame(area, w, tuple(ta + ka + pa))
    target_cols = ["target"] + [f"aux_{a}" for a in ta]
    weather_cols = [c for c in df.columns if c != "target" and not c.startswith("aux_")]
    known_cov = weather_cols + [f"aux_{a}" for a in ka]
    past_cov = [f"aux_{a}" for a in pa]
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

def build_fit_task(df: pd.DataFrame, train_end: pd.Timestamp, target_cols, known_cov, past_cov):
    """Fit-Task aus dem Trainingszeitraum (< train_end, d.h. nur 2023). Ganze Reihe;
    Chronos-2 sampelt intern Fenster. future_covariates=None markiert known-future."""
    tr = df.loc[:train_end]
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


def build_joint_fit_task(frames: dict, train_end):
    common = None
    for _, (df, *_ ) in frames.items():
        idx = df.loc[:train_end].index
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
    tf32=False fuer ROCm/AMD-Kompatibilitaet."""
    return pipe().fit(fit_tasks, prediction_length=horizon, finetune_mode="lora",
                      context_length=ctx_steps, learning_rate=lr, num_steps=num_steps,
                      batch_size=batch_size, output_dir=out_dir,
                      remove_printer_callback=True, tf32=False)
