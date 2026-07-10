"""Live-Forecast des naechsten Liefertags mit dem Produktions-Modell.

Nutzt dieselbe Chronos-2-Task-Logik wie die Experimente (experiments.model), aber
fuer EIN zukuenftiges Liefertag-Fenster. Liefert je Gebiet die 21 Samples der 96
Liefer-Steps; DE-LU wird daraus kombiniert (Quantil-Summe bzw. Bootstrap-Ensemble).

Produktions-Config via env PROD_CONFIG (Default G2_C4 — 5 Regionen + Wetter/Aux-Input).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from experiments import configs, data, model

FREQ = "15min"
STEPS = 96
CTX_DAYS = int(os.environ.get("CTX_DAYS", "63"))
# Produktions-Config: Default robust (Wetter-only, keine Abhaengigkeit vom noch
# unveroeffentlichten Day-ahead-Preis). G2_C4 ist im Backtest bestes, braucht aber
# den Liefertag-Preis als Covariate -> nur setzen, wenn der zur Abgabezeit vorliegt.
PROD_CONFIG = os.environ.get("PROD_CONFIG", "G2_C2.2")
QUANTILE_LEVELS = [0.025, 0.25, 0.5, 0.75, 0.975]


def _delivery_window(delivery_date=None, gate_hour: int = 9):
    """(cutoff, fut_idx, deliv-mask) fuer den Liefertag (Default: morgen UTC).
    Cutoff = D-1 09:00 UTC (Day-ahead-Gate) — identisch zum Backtest, damit
    Live-Prognose und Evaluation deckungsgleich sind."""
    now = pd.Timestamp.now(tz="UTC")
    D = pd.Timestamp(delivery_date) if delivery_date is not None else (now.normalize() + pd.Timedelta(days=1))
    D = (D.tz_convert("UTC") if D.tz else D.tz_localize("UTC")).normalize()
    cutoff = D - pd.Timedelta(days=1) + pd.Timedelta(hours=gate_hour)
    end = D + pd.Timedelta(hours=23, minutes=45)
    fut_idx = pd.date_range(cutoff + pd.Timedelta(FREQ), end, freq=FREQ)
    return cutoff, fut_idx, (fut_idx.normalize() == D)


def zone_samples(cfg_name: str = PROD_CONFIG, delivery_date=None):
    """{gebiet: (21, 96)} Last-Samples des Liefertags + die 96 Liefer-Zeitstempel."""
    cfg = configs.get(cfg_name)
    cutoff, fut_idx, deliv = _delivery_window(delivery_date)
    ctx_steps = CTX_DAYS * STEPS
    frames = {}
    for a in cfg.areas:
        try:
            frames[a] = model.area_frame(cfg, a)
        except FileNotFoundError:
            pass
    out = {}
    ts = fut_idx[deliv]
    if cfg.gran in ("whole", "indep"):
        tasks, tas = [], []
        for a, (df, tcols, known, past) in frames.items():
            t = model.build_task(df, cutoff, tcols, known, past, ctx_steps, fut_idx)
            if t is not None:
                tasks.append(t); tas.append(a)
        for a, o in zip(tas, model.predict(tasks, len(fut_idx))):
            out[a] = np.clip(o[0][:, deliv], 0, None)
    else:                                           # joint
        task, zo = model.build_joint_task(frames, cutoff, ctx_steps, fut_idx)
        if task is not None:
            arr = model.predict([task], len(fut_idx))[0]
            pos = 0
            for a in zo:
                nvar = len(frames[a][1])
                out[a] = np.clip(arr[pos][:, deliv], 0, None)
                pos += nvar
    if not out:
        raise RuntimeError("Forecast leer — Daten decken das Liefertag-Fenster nicht "
                           "(Wetter/Aux aktuell genug?).")
    return out, ts


# --- DE-LU-Kombination der Zonen --------------------------------------------

def delu_quantiles(zsamp: dict, ql=QUANTILE_LEVELS) -> np.ndarray:
    """Summe der per-Zone-Quantile -> (nq, 96). (whole: nur de_lu -> dessen Quantile)"""
    return sum(np.quantile(s, ql, axis=0) for s in zsamp.values())


def delu_ensemble(zsamp: dict, n: int = 100, seed: int = 0) -> np.ndarray:
    """Bootstrap-Ensemble: je Zone n zufaellige Samples ziehen und summieren -> (n, 96)."""
    rng = np.random.default_rng(seed)
    total = None
    for s in zsamp.values():
        pick = s[rng.integers(0, s.shape[0], size=n)]     # (n, 96)
        total = pick if total is None else total + pick
    return total


def delu_point(zsamp: dict) -> np.ndarray:
    """Punktprognose = Summe der Zonen-Mediane -> (96,)."""
    q = delu_quantiles(zsamp)
    return q[QUANTILE_LEVELS.index(0.5)]
