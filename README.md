# Daten-Pipeline — Day-Ahead-Lastprognose (DE-LU)

Schlanke Lade-/Aufbereitungsskripte für die Forschungsfragen (Chronos-2 / Chronos-X
vs. ENTSO-E-Benchmark). Nur **Daten** — kein Modelltraining, kein Submit.

## Drei Datenquellen

| # | Quelle | Modul(e) | Ausgabe unter `data/` |
|---|--------|----------|-----------------------|
| 1 | **Wetter je H3-Zelle** (224 Zellen), Punkt + Ensemble | `weather.py`, `aggregate.py` | `historical/`, `historical_ens/`, `weather_zone/{pop,centroid}/` |
| 2 | **Markt je ÜNB-Zone** (Preis/Solar/Wind/Residual/Last) | `loads.py`, `market.py` | `loads/`, `load_forecast/`, `features/aux_*/` |
| 3 | **Gesamt DE-LU** (aggregiert + eine Zelle) | `loads.py`, `market.py`, `aggregate.py` | dieselben Ordner, Gebiet `de_lu` |

Gebiete überall: die 4 Regelzonen `50hertz, tennet, amprion, transnetbw` + die
kombinierte Gebotszone `de_lu` + **Luxemburg `lu` als eigene Reihe** (siehe unten).

## Wetter-Aggregation: zwei Methoden

Zell-Wetter → Zonen-Wetter auf zwei Arten, damit ihr Effekt vergleichbar ist:

- **pop** — bevölkerungsgewichtetes Mittel über alle Zellen des Gebiets (aktuell).
- **centroid** — nur die Repräsentativ-Zelle (Zelle am bevölkerungsgewichteten
  Schwerpunkt, `grid.centroid_cell`).

`aggregate.analyse_diff()` quantifiziert den Unterschied (MAE/RMSE/Korrelation je
Gebiet & Variable) → `data/analysis/weather_agg_diff.csv`. (Beispiel: T2m-Korr
≈ 0,98–0,99, Wind deutlich niedriger ≈ 0,80.)

## Luxemburg

`DE_LU` ist **eine** ENTSO-E-Gebotszone; Luxemburg hat keine eigene Zone. `de_lu`
Ist-Last und Preis (aus ENTSO-E `DE_LU`) enthalten Luxemburg also implizit — die
Ground-Truth des Gesamtmarkts ist vollständig vorhanden.

`loads.reconcile_delu()` misst empirisch `de_lu − Σ(4 Zonen)`
(→ `data/analysis/delu_reconciliation.csv`): in der aktuellen Historie ein
**stabiler Offset von ~544 MW (Median, +1,0 %)** — das ist gerade die
Luxemburg-Last. Die Feature-Seite der Zonen (Wetterzellen + Solar/Wind = Summe der
4 dt. ÜNB) ist bewusst Deutschland-only.

### Luxemburg als eigene Reihe (`lu`)

Luxemburg wird zusätzlich als **eigenständiges Gebiet** geführt:

- **Wetter**: eigene H3-Zelle `841fa3dffffffff` (Landes-Schwerpunkt, 49,82 N / 6,13 O),
  konfiguriert in `config.LU_CELL`. `weather.py` lädt sie (Punkt + Ensemble) mit;
  `aggregate.py` erzeugt `weather_zone/{pop,centroid}/lu.csv` (eine Zelle ⇒ pop == centroid).
- **Last (Ziel)**: ENTSO-E `LU` (Creos), viertelstündlich ab 2015 — direkt via
  `loads.py` → `data/loads/lu_*.csv` + `load_forecast/lu_*.csv`. Zusätzlich als
  Kontrolle die **abgeleitete** Reihe `de_lu − Σ(4 Zonen)` → `data/loads/lu_derived_*.csv`.
- **Preis**: identisch zum DE-LU-Day-ahead → `features/aux_price/lu.csv`.
- **Solar/Wind/Residual**: SMARD kennt keine LU-Region ⇒ bewusst **weggelassen**.

## Setup & Nutzung

```bash
pip install -r requirements.txt
cp .env.example .env      # ENTSOE_API_KEY eintragen (SMARD/Open-Meteo: kein Key)
```

Einzelmodule (alle `python -m loaders.<modul>`):

```bash
python -m loaders.grid                 # Kontrolle: Zellen, Gewichte, Centroid-Zellen
python -m loaders.weather point        # Punkt-Wetter je Zelle/Jahr (Einmal-Backfill)
python -m loaders.weather ensemble     # ICON-D2-EPS, nächster Tag
python -m loaders.aggregate --ensemble # Zonen-Wetter (pop+centroid) + Diff-Analyse
python -m loaders.loads                # Ist-Last + Day-ahead-Prognose (4 Zonen + de_lu)
python -m loaders.loads --reconcile    # nur DE-LU-Reconciliation-Diagnose
python -m loaders.market               # Preis/Solar/Wind/Residual (+ de_lu)
```

Orchestrierung (ein einziger Scheduler):

```bash
python -m loaders.schedule --once      # Voll-Backfill (Erststart)
python -m loaders.schedule             # Daemon: stündl. Last-Refresh, tägl. Wetter/Markt
```

## Daten-Snapshot (Backup / Weitergabe)

`data/` ist git-ignoriert (~2,8 GB). Statt es zu versionieren, wird es als
komprimiertes Archiv gesichert/weitergegeben (zstd, ~600 MB, behält die
Ordnerstruktur).

**Erstellen** (außerhalb des Repos ablegen, damit es nicht mitgetrackt wird):
```bash
cd code
tar --zstd -cf ../data-snapshot-$(date +%F).tar.zst data/
```

**Entpacken** (stellt `data/` wieder her):
```bash
cd code
tar --zstd -xf ../data-snapshot-2026-07-09.tar.zst      # Dateinamen anpassen
```
Fehlt `zstd`, vorher installieren (`sudo dnf install zstd` / `apt install zstd`);
alternativ `tar -I zstd -xf …`. Nach dem Entpacken liegen alle Reihen wieder unter
`code/data/` — die Loader nutzen sie direkt, ein erneuter Download entfällt.

## Struktur

```
code/
├── loaders/        config.py grid.py weather.py aggregate.py loads.py market.py schedule.py
├── reference/      germany_h3_res4.csv  cell_population.csv   (statische Eingaben)
├── scripts/plot_h3/  H3-Zonenkarten (plot_h3.py + PNGs)
└── data/           erzeugte Ausgaben (git-ignoriert; via Snapshot gesichert)
```

Alle Konstanten/Pfade/Codes zentral in `loaders/config.py`. Zeiten durchgehend UTC,
15-min bzw. (aux) stündlich.
