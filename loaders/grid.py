"""H3-Grid: Zellen, Bevoelkerungsgewichte und Repraesentativ-Zelle je Gebiet.

Ersetzt die frueher hartkodierten Zell-Listen (tso_hexagons.py) — alle 224 Zellen
und ihre Zuordnung werden aus reference/germany_h3_res4.csv + cell_population.csv
abgeleitet.

Zwei Aggregations-Grundlagen fuer das Zonen-Wetter:
  - population-gewichtet  -> Spalte w (Anteil an der Zonen-Bevoelkerung)
  - eine Zelle je Zone    -> centroid_cell(): Zelle am bevoelkerungsgewichteten
                             Schwerpunkt des Gebiets.

Aufruf zur Kontrolle:
    python -m loaders.grid
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def cells() -> pd.DataFrame:
    """224 Zellen mit Geo + Zone-Key + Bevoelkerung + Zonen-Gewicht w.

    Spalten: h3_index, lat, lng, zone, population, w
    (w = population / Summe(population) je Zone; Summe je Zone == 1.0).
    """
    c = pd.read_csv(config.CELLS_CSV)
    p = pd.read_csv(config.POP_CSV)[["h3_index", "population"]]
    c = c.merge(p, on="h3_index", how="left")
    c["zone"] = c["tso"].map(config.LABEL_TO_ZONE)
    c["population"] = c["population"].fillna(0.0)
    c["w"] = c.groupby("zone")["population"].transform(lambda s: s / s.sum())
    return c[["h3_index", "lat", "lng", "zone", "population", "w"]]


def lu_cell() -> pd.DataFrame:
    """Luxemburgs eigene Solo-Zelle (nicht Teil des deutschen Grids)."""
    c = config.LU_CELL
    return pd.DataFrame([{"h3_index": c["h3_index"], "lat": c["lat"], "lng": c["lng"],
                          "zone": config.LU, "population": c["population"], "w": 1.0}])


def download_cells() -> pd.DataFrame:
    """Alle Zellen fuer den Wetter-Download: 224 deutsche + die LU-Solo-Zelle."""
    return pd.concat([cells(), lu_cell()], ignore_index=True)


def zone_cells(area: str, grid: pd.DataFrame | None = None) -> pd.DataFrame:
    """Zellen eines Gebiets. area in ZONES -> nur diese Zone; area == de_lu ->
    alle (deutschen) Zellen, neu ueber Deutschland normiert; area == lu -> Solo-Zelle."""
    if area == config.LU:
        return lu_cell()
    g = cells() if grid is None else grid
    if area == config.DELU:
        g = g.copy()
        g["w"] = g["population"] / g["population"].sum()
        return g
    return g[g.zone == area].copy()


def centroid_cell(area: str, grid: pd.DataFrame | None = None) -> str:
    """h3_index der Zelle am bevoelkerungsgewichteten Schwerpunkt des Gebiets.

    Schwerpunkt = mit population gewichtetes Mittel aus (lat, lng); zurueckgegeben
    wird die Zelle mit minimalem Abstand dazu. area == de_lu -> ueber alle Zellen.
    """
    g = zone_cells(area, grid)
    w = g["population"].to_numpy()
    if w.sum() == 0:
        w = np.ones(len(g))
    lat0 = np.average(g["lat"], weights=w)
    lng0 = np.average(g["lng"], weights=w)
    d2 = (g["lat"] - lat0) ** 2 + (g["lng"] - lng0) ** 2
    return g.loc[d2.idxmin(), "h3_index"]


def centroid_cells(grid: pd.DataFrame | None = None) -> dict[str, str]:
    """Repraesentativ-Zelle je Gebiet (4 Zonen + de_lu + lu)."""
    g = cells() if grid is None else grid
    return {a: centroid_cell(a, g) for a in config.AGG_AREAS}


def even_cells(area: str, k: int = 4, grid: pd.DataFrame | None = None,
               seed: int = 0) -> list[str]:
    """k geografisch gleichmaessig verteilte Zellen eines Gebiets (fuer C2.3 —
    raeumliches Wetter ohne Aggregation). k-means auf (lat,lng), je Cluster die
    naechstgelegene reale Zelle. Gebiete mit <= k Zellen (z.B. lu) -> alle Zellen.
    Deterministisch ueber seed."""
    g = zone_cells(area, grid).reset_index(drop=True)
    pts = g[["lat", "lng"]].to_numpy(float)
    if len(g) <= k:
        return g["h3_index"].tolist()
    rng = np.random.default_rng(seed)
    centers = [pts[rng.integers(len(pts))]]                 # k-means++ Init
    for _ in range(1, k):
        d2 = np.min([((pts - c) ** 2).sum(1) for c in centers], axis=0)
        centers.append(pts[rng.choice(len(pts), p=d2 / d2.sum())])
    centers = np.array(centers)
    for _ in range(50):                                     # Lloyd-Iterationen
        assign = np.argmin(((pts[:, None] - centers[None]) ** 2).sum(2), axis=1)
        new = np.array([pts[assign == j].mean(0) if (assign == j).any() else centers[j]
                        for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    chosen: list[str] = []                                  # je Cluster naechste freie Zelle
    for j in range(k):
        for idx in np.argsort(((pts - centers[j]) ** 2).sum(1)):
            h = g.loc[idx, "h3_index"]
            if h not in chosen:
                chosen.append(h)
                break
    return chosen


def main() -> None:
    g = cells()
    print(f"{len(g)} Zellen, Zonen: {sorted(g.zone.dropna().unique())}")
    print("\nGewichtssumme je Zone (soll 1.0):")
    print(g.groupby("zone")["w"].sum())
    print("\nRepraesentativ-Zelle (Bevoelkerungs-Schwerpunkt) je Gebiet:")
    disp = download_cells().set_index("h3_index")   # inkl. LU-Solo-Zelle
    for area, h in centroid_cells(g).items():
        row = disp.loc[h]
        print(f"  {area:11s} -> {h}  ({row.lat:.3f}, {row.lng:.3f})  pop={row.population:,.0f}")
    print("\n4 gleichmaessig verteilte Zellen je Gebiet (C2.3):")
    for area in config.AGG_AREAS:
        cs = even_cells(area, 4, g)
        pts = [f"({disp.loc[h].lat:.2f},{disp.loc[h].lng:.2f})" for h in cs]
        print(f"  {area:11s} -> {len(cs)} Zellen  {' '.join(pts)}")


if __name__ == "__main__":
    main()
