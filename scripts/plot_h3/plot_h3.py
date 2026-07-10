"""Maps of the H3 zone partition (+ population-centroid cells).

Draws the 224 German H3 cells (resolution 4) plus the LU solo cell as hexagons,
coloured by TSO zone, and marks each area's population-centroid cell
(loaders.grid.centroid_cell).

Writes three figures into this folder (scripts/plot_h3/):
  zones_overview.png    all zones + LU, centroids as stars
  zones_population.png  cells shaded by population, centroids marked
  zones_facets.png      small multiples: each area highlighted + its centroid

Run:
    python scripts/plot_h3/plot_h3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import h3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

# Paket 'loaders' aus dem Projekt-Root (code/) importierbar machen
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from loaders import config, grid  # noqa: E402

# Categorical palette (validated, CVD-safe) — fixed order, never cycled
ZONE_COLOR = {"tennet": "#2a78d6", "50hertz": "#1baf7a", "amprion": "#eda100",
              "transnetbw": "#008300", "lu": "#4a3aa7"}
ZONE_ORDER = ["tennet", "50hertz", "amprion", "transnetbw", "lu"]
OUT_DIR = Path(__file__).resolve().parent


def _boundary_xy(h3_index: str):
    """H3 cell boundary -> (x=lng, y=lat) polygon vertices."""
    ring = h3.cell_to_boundary(h3_index)          # [(lat, lng), ...]
    return [(lng, lat) for lat, lng in ring]


def _aspect(lat_mean: float) -> float:
    """Longitude degrees are narrower at ~51N — un-distort the map."""
    return 1.0 / np.cos(np.radians(lat_mean))


def _draw_cells(ax, g: pd.DataFrame, facecolors, edgecolor="white", lw=0.4, alpha=1.0):
    patches = [Polygon(_boundary_xy(h), closed=True) for h in g.h3_index]
    pc = PatchCollection(patches, facecolor=facecolors, edgecolor=edgecolor,
                         linewidths=lw, alpha=alpha)
    ax.add_collection(pc)


def _mark_centroid(ax, g, area, color, star=True, label=True):
    h = grid.centroid_cell(area)
    row = g[g.h3_index == h].iloc[0]
    ax.add_patch(Polygon(_boundary_xy(h), closed=True, fill=False,
                         edgecolor="#111111", linewidth=2.2, zorder=5))
    if star:
        ax.plot(row.lng, row.lat, marker="*", markersize=17, color=color,
                markeredgecolor="#111111", markeredgewidth=0.8, zorder=6)
    if label:
        ax.annotate(config.ZONE_LABEL.get(area, area),
                    (row.lng, row.lat), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8.5, fontweight="bold", color="#111111",
                    zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    return row


def _frame(ax, g):
    ax.set_aspect(_aspect(g.lat.mean()))
    ax.set_xlim(g.lng.min() - 0.4, g.lng.max() + 0.4)
    ax.set_ylim(g.lat.min() - 0.3, g.lat.max() + 0.3)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# --- 1) Overview ------------------------------------------------------------

def plot_overview(g: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 10.5))
    colors = [ZONE_COLOR.get(z, "#cccccc") for z in g.zone]
    _draw_cells(ax, g, colors)
    for area in ZONE_ORDER:
        _mark_centroid(ax, g, area, ZONE_COLOR[area])
    hd = grid.centroid_cell(config.DELU)          # whole-DE-LU centroid
    rd = g[g.h3_index == hd].iloc[0]
    ax.plot(rd.lng, rd.lat, marker="D", markersize=9, color="#111111", zorder=6)
    _frame(ax, g)
    handles = [Line2D([0], [0], marker="s", linestyle="", markersize=10,
                      markerfacecolor=ZONE_COLOR[z], markeredgecolor="white",
                      label=config.ZONE_LABEL.get(z, z)) for z in ZONE_ORDER]
    handles += [Line2D([0], [0], marker="*", linestyle="", markersize=13,
                       markerfacecolor="#777", markeredgecolor="#111", label="Zone centroid cell"),
                Line2D([0], [0], marker="D", linestyle="", markersize=9,
                       markerfacecolor="#111", markeredgecolor="#111", label="DE-LU centroid (overall)")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)
    ax.set_title("H3 zone partition (resolution 4) with population centroids",
                 fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "zones_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- 2) Population ----------------------------------------------------------

def plot_population(g: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 10.5))
    pop = g["population"].to_numpy(dtype=float)
    norm = matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=np.nanmax(pop))
    cmap = matplotlib.colormaps["Blues"]
    colors = cmap(norm(pop))
    _draw_cells(ax, g, colors, edgecolor="#ffffff", lw=0.3)
    for area in ZONE_ORDER:
        _mark_centroid(ax, g, area, "#e34948", label=False)
    _frame(ax, g)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Population per cell", fontsize=9)
    ax.set_title("Population per H3 cell — centroid cells (red) sit at the density center",
                 fontsize=11.5, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "zones_population.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- 3) Small multiples -----------------------------------------------------

def plot_facets(g: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(17, 7.5))
    for ax, area in zip(axes, ZONE_ORDER):
        _draw_cells(ax, g, ["#eceae4"] * len(g), edgecolor="white", lw=0.3)   # all grey
        sub = g[g.zone == area]
        _draw_cells(ax, sub, [ZONE_COLOR[area]] * len(sub), edgecolor="white", lw=0.4)
        _mark_centroid(ax, g, area, ZONE_COLOR[area], label=False)
        _frame(ax, g)
        share = 100 * sub["population"].sum() / g["population"].sum()
        word = "cell" if len(sub) == 1 else "cells"
        ax.set_title(f"{config.ZONE_LABEL.get(area, area)}\n{len(sub)} {word} · "
                     f"{share:.0f}% pop.", fontsize=10, fontweight="bold")
    fig.suptitle("Zones individually — cells of the zone coloured, centroid as star",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "zones_facets.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    g = grid.download_cells()                     # 224 German cells + LU solo cell
    plot_overview(g)
    plot_population(g)
    plot_facets(g)
    print(f"3 figures -> {OUT_DIR}")


if __name__ == "__main__":
    main()
