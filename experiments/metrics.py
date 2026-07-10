"""Bewertungsmetriken (Energy-Arena-Definitionen).

Konventionen:
  y       (n,)            realisierte Werte
  p       (n,)            Punktprognose
  qmat    (n, K)          Quantilprognosen, Spalten in QLEVELS-Reihenfolge
  samples (S, n)          S Ensemble-Member je Zeitpunkt
  Fuer den Energy Score: pro Liefertag (y_day (H,), samp_day (S,H)).

Quantil-Level (Challenge 22): [0.025, 0.25, 0.5, 0.75, 0.975]
  Cov50 = [q0.25, q0.75] (idx 1,3);  Cov95 = [q0.025, q0.975] (idx 0,4);  Median = idx 2.
Alle Scores: niedriger = besser (ausser R2 und Coverage).
"""
from __future__ import annotations

import numpy as np

QLEVELS = [0.025, 0.25, 0.5, 0.75, 0.975]
MID = 2            # Index des Medians (q0.5)
COV = {"cov50": (1, 3), "cov95": (0, 4)}


# --- Punkt --------------------------------------------------------------------

def rmse(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(p, float) - np.asarray(y, float))))


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def smape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    denom = np.abs(y) + np.abs(p)
    return float(200 * np.mean(np.where(denom == 0, 0.0, np.abs(p - y) / denom)))


# --- Quantile -----------------------------------------------------------------

def pinball(y, q_pred, level):
    """Pinball-/Quantil-Verlust (LQS) auf einem Level, gemittelt ueber Zeitpunkte."""
    y, q_pred = np.asarray(y, float), np.asarray(q_pred, float)
    d = y - q_pred
    return float(np.mean(np.maximum(level * d, (level - 1) * d)))


def lqs(y, qmat, levels=QLEVELS):
    """Dict {level: Pinball} + Mittel (== WIS in dieser WQL-Formulierung)."""
    qmat = np.asarray(qmat, float)
    per = {lv: pinball(y, qmat[:, j], lv) for j, lv in enumerate(levels)}
    per["mean"] = float(np.mean(list(per.values())))
    return per


# symmetrische Intervalle aus QLEVELS: (lo_idx, hi_idx, alpha=1-Nominalabdeckung)
_WIS_PAIRS = [(0, 4, 0.05), (1, 3, 0.5)]      # 95%-Intervall, 50%-Intervall


def wis(y, qmat, pairs=_WIS_PAIRS, mid=MID):
    """Weighted Interval Score (Energy-Arena-Definition):
        WIS = [1/2*|y-med| + sum_k (alpha_k/2)*IS_k] / (K + 1/2)
    mit IS_k = (u-l) + (2/alpha)(l-y)1{y<l} + (2/alpha)(y-u)1{y>u}.
    Reduziert sich auf MAE, wenn nur der Median vorliegt (pairs=[])."""
    y = np.asarray(y, float)
    qmat = np.asarray(qmat, float)
    total = 0.5 * np.abs(y - qmat[:, mid])
    for lo, hi, alpha in pairs:
        l, u = qmat[:, lo], qmat[:, hi]
        IS = (u - l) + (2 / alpha) * (l - y) * (y < l) + (2 / alpha) * (y - u) * (y > u)
        total = total + (alpha / 2) * IS
    return float(np.mean(total / (len(pairs) + 0.5)))


def wis_median_only(y, med):
    """WIS fuer eine Median-only-Prognose (z.B. ENTSO-E-Benchmark) == MAE."""
    return mae(y, med)


def mae_median(y, qmat):
    return mae(y, np.asarray(qmat, float)[:, MID])


def coverage(y, qmat, lo_idx, hi_idx):
    """Anteil (%) der y innerhalb [q_lo, q_hi]."""
    y = np.asarray(y, float)
    qmat = np.asarray(qmat, float)
    return float(np.mean((qmat[:, lo_idx] <= y) & (y <= qmat[:, hi_idx])) * 100)


def coverages(y, qmat):
    return {k: coverage(y, qmat, lo, hi) for k, (lo, hi) in COV.items()}


# --- Ensemble -----------------------------------------------------------------

def crps(y, samples):
    """Mittlerer empirischer CRPS ueber Zeitpunkte.
    CRPS(t) = E|X-y| - 1/2 E|X-X'|, aus S Membern (samples: (S, n)).
    Nutzt die sortierte O(S log S)-Form fuer den Paar-Term (speicherschonend)."""
    y = np.asarray(y, float)
    x = np.asarray(samples, float)                  # (S, n)
    S = x.shape[0]
    t1 = np.abs(x - y[None, :]).mean(0)             # (n,)
    xs = np.sort(x, axis=0)
    i = np.arange(1, S + 1)[:, None]
    ediff = 2.0 * ((2 * i - S - 1) * xs).sum(0) / (S * S)   # E|X-X'| je Zeitpunkt
    return float((t1 - 0.5 * ediff).mean())


def energy_score_day(y_day, samp_day):
    """Energy Score einer Tages-Trajektorie: ES = E||X-y|| - 1/2 E||X-X'||
    (euklidische Norm ueber die H Zeitschritte). y_day (H,), samp_day (S,H)."""
    y = np.asarray(y_day, float)
    x = np.asarray(samp_day, float)                 # (S, H)
    t1 = np.linalg.norm(x - y[None, :], axis=1).mean()
    diff = x[:, None, :] - x[None, :, :]            # (S,S,H) — S klein (21)
    t2 = np.linalg.norm(diff, axis=2).mean()
    return float(t1 - 0.5 * t2)


def energy_score(days):
    """Mittlerer Energy Score ueber Liefertage. days: Iterable von (y_day, samp_day)."""
    vals = [energy_score_day(y, s) for y, s in days]
    return float(np.mean(vals)) if vals else float("nan")


# --- Sammel-Scorer ------------------------------------------------------------

def score_point(y, samples):
    """Punkt-Metriken (Punkt = Sample-Mittel, RMSE-optimal)."""
    p = np.asarray(samples, float).mean(0)
    return {"rmse": rmse(y, p), "r2": r2(y, p), "mae": mae(y, p), "smape": smape(y, p)}


def score_quantile(y, samples, levels=QLEVELS):
    """Quantil-Metriken aus Ensemble-Samples (via np.quantile)."""
    qmat = np.quantile(np.asarray(samples, float), levels, axis=0).T   # (n, K)
    out = {"wis": wis(y, qmat), "lqs": lqs(y, qmat, levels)["mean"],
           "mae_median": mae_median(y, qmat)}
    out.update(coverages(y, qmat))
    return out


def score_ensemble(y, samples, days):
    """Ensemble-Metriken: Quantil-Set + CRPS + Energy Score."""
    out = score_quantile(y, samples)
    out["crps"] = crps(y, samples)
    out["energy_score"] = energy_score(days)
    return out
