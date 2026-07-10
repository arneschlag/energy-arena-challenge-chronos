"""Ergebnis-Plots der Zero-Shot Chronos-2 Walk-Forward-Experimente.

Liest experiments/out/results.csv und erzeugt (in diesen Ordner):
  results_overview.png   WIS + RMSE je Config-Typ x Granularitaet (DE-LU, overall) + ENTSO-E
  results_ablation.png   Wetter-Aggregation (C1/C2.1/2.2/2.3) und Aux (C1/C3.*) als WIS
  results_periods.png    WIS je Zeitraum (2024/2025/2026H1) fuer ausgewaehlte Configs

Run:  python scripts/plot_results/plot_results.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "out" / "results.csv"
OUT = Path(__file__).resolve().parent

# validierte Palette, nach Granularitaet
GRAN_COLOR = {"G1": "#2a78d6", "G2": "#1baf7a", "G3": "#4a3aa7"}
GRAN_LABEL = {"G1": "G1 whole", "G2": "G2 region-indep", "G3": "G3 region-joint"}
ENTSOE_C = "#e34948"
CTYPES = ["C1", "C2.1", "C2.2", "C2.3", "C3.1", "C3.2", "C3.3", "C3.4", "C4"]


def load():
    r = pd.read_csv(RESULTS)
    return r


def val(r, config, output, metric, area="de_lu", period="overall"):
    x = r[(r.config == config) & (r.area == area) & (r.period == period)
          & (r.output == output) & (r.metric == metric)]
    return float(x.value.iloc[0]) if len(x) else np.nan


def _grouped(ax, r, output, metric, title, ylabel, entsoe=None):
    x = np.arange(len(CTYPES)); w = 0.26
    for i, g in enumerate(["G1", "G2", "G3"]):
        vals = [val(r, f"{g}_{c}", output, metric) for c in CTYPES]
        ax.bar(x + (i - 1) * w, vals, w, label=GRAN_LABEL[g], color=GRAN_COLOR[g])
    if entsoe is not None and np.isfinite(entsoe):
        ax.axhline(entsoe, color=ENTSOE_C, ls="--", lw=1.6, label=f"ENTSO-E ({entsoe:.0f})")
    ax.set_xticks(x); ax.set_xticklabels(CTYPES, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_overview(r):
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    _grouped(axes[0], r, "quantile", "wis",
             "WIS (probabilistisch, niedriger = besser) — DE-LU, 2024–2026H1",
             "WIS [MW]", entsoe=val(r, "ENTSOE", "quantile", "wis"))
    _grouped(axes[1], r, "point", "rmse",
             "RMSE (Punkt) — DE-LU, 2024–2026H1", "RMSE [MW]",
             entsoe=val(r, "ENTSOE", "point", "rmse"))
    axes[0].legend(frameon=False, fontsize=9, ncol=4, loc="upper right")
    fig.suptitle("Zero-Shot Chronos-2 vs. ENTSO-E — je Config-Typ und Granularitaet",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "results_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(r):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Wetter-Aggregation (WIS je Granularitaet)
    wcfgs = ["C1", "C2.1", "C2.2", "C2.3"]
    x = np.arange(len(wcfgs)); w = 0.26
    for i, g in enumerate(["G1", "G2", "G3"]):
        axes[0].bar(x + (i - 1) * w, [val(r, f"{g}_{c}", "quantile", "wis") for c in wcfgs],
                    w, color=GRAN_COLOR[g], label=GRAN_LABEL[g])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["none", "1 cell", "pop-avg", "4 cells"], fontsize=9)
    axes[0].set_title("Weather source (C2) — WIS", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("WIS [MW]")
    # Aux als Ziel (C3)
    acfgs = ["C1", "C3.1", "C3.2", "C3.3", "C3.4"]
    x2 = np.arange(len(acfgs))
    for i, g in enumerate(["G1", "G2", "G3"]):
        axes[1].bar(x2 + (i - 1) * w, [val(r, f"{g}_{c}", "quantile", "wis") for c in acfgs],
                    w, color=GRAN_COLOR[g])
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(["none", "+price", "+solar", "+wind", "+all"], fontsize=9)
    axes[1].set_title("Aux as auxiliary target (C3) — WIS", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("WIS [MW]")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Ablation (DE-LU, 2024–2026H1): more raw inputs / aux targets do NOT help zero-shot",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "results_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_periods(r):
    periods = ["2024", "2025", "2026H1"]
    sel = [("G2_C4", "#1baf7a"), ("G1_C2.2", "#2a78d6"), ("G1_C1", "#eda100")]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(periods)); w = 0.2
    for i, (cfg, col) in enumerate(sel):
        vals = [val(r, cfg, "quantile", "wis", period=p) for p in periods]
        ax.bar(x + (i - 1.5) * w, vals, w, label=cfg, color=col)
    ent = [val(r, "ENTSOE", "quantile", "wis", period=p) for p in periods]
    ax.bar(x + 1.5 * w, ent, w, label="ENTSO-E", color=ENTSOE_C)
    ax.set_xticks(x); ax.set_xticklabels(periods)
    ax.set_ylabel("WIS [MW]"); ax.set_title("WIS je Zeitraum (DE-LU)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "results_periods.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    if not RESULTS.exists():
        sys.exit(f"fehlt: {RESULTS}")
    r = load()
    plot_overview(r)
    plot_ablation(r)
    plot_periods(r)
    print(f"3 Plots -> {OUT}")


if __name__ == "__main__":
    main()
