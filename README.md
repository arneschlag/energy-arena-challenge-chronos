# Day-Ahead-Lastprognose DE-LU - Chronos-2-Pipeline

Vollautomatische Day-ahead-Lastprognose für die **DE-LU-Gebotszone** mit Amazon
**Chronos-2**: holt selbst alle Daten (Wetter, ENTSO-E-Last, SMARD-Markt), erstellt
täglich die Prognose und reicht sie bei der **Energy Arena** ein. Man braucht nur
einen **GPU-Docker-Container (AMD oder NVIDIA)** und **API-Keys** — der Rest läuft
von allein.

---

## Schnellstart - alles automatisch (Windows & Linux)

### Voraussetzungen
- **Docker** inkl. Docker Compose. Unter **Windows**: *Docker Desktop* mit WSL2-Backend.
- **Eine GPU** (empfohlen):
  - **NVIDIA** - aktueller Treiber; Linux: *NVIDIA Container Toolkit*, Windows: Docker Desktop + WSL2 genuegt.
  - **AMD** - ROCm-faehige GPU unter **Linux** (AMD-GPU-Passthrough unter Windows wird nicht unterstuetzt).
  - **Ohne GPU** läuft es auf CPU (`DEVICE=cpu` in `.env`), nur deutlich langsamer.
- **ENTSO-E-API-Key** (kostenlos): https://transparency.entsoe.eu → *Account Settings → Web Api Security Token*.

### In 3 Schritten

**1) Repo holen und `.env` anlegen**

Linux / macOS:
```bash
git clone <REPO-URL> && cd code
cp .env.example .env
```
Windows (PowerShell):
```powershell
git clone <REPO-URL>; cd code
copy .env.example .env
```
Dann `.env` öffnen und **`ENTSOE_API_KEY`** eintragen. Für die **echte Abgabe** zusätzlich
`ARENA_API_KEY` setzen und `SUBMIT_ENABLED=true`. (Ohne das läuft alles als **Dry-Run** —
es wird gerechnet, aber nichts gesendet.)

**2) Container starten** (Profil je nach GPU)
```bash
# NVIDIA
docker compose -f docker/docker-compose.yml --profile nvidia up -d --build
# AMD (Linux)
docker compose -f docker/docker-compose.yml --profile amd    up -d --build
```

**3) Fertig.** Logs mitverfolgen:
```bash
docker compose -f docker/docker-compose.yml logs -f
```

Beim **ersten Start** lädt der Container automatisch die historischen Daten (einmalig,
dauert je nach Verbindung). Danach hält er alles stündlich frisch und reicht täglich ein.
Daten (`data/`) und das Chronos-2-Modell (HuggingFace-Cache) werden in Docker-Volumes
persistiert, überstehen also Neustarts.

### Was dann automatisch passiert
| Zeit (UTC) | Job |
|---|---|
| stündlich `:05` | **Daten-Refresh** - Punktwetter-Forecasts, ENTSO-E-Last, SMARD-Markt, Aggregation |
| stündlich `:20` | **Prognose** des nächsten Liefertags + **Abgabe** an die Energy Arena, sobald ein Fenster offen ist (Dedup, DST-korrekt) |

### Modell waehlen (`PROD_CONFIG` in `.env`)
- **`G1_C5`** *(Default / Submission-Champion)* - ein DE-LU-Gesamtmodell mit
  Punktwetter-Forecasts, past-only Preis/Solar/Wind und Kalenderfeatures.
- **`G1_C1`** - univariat, nur historische Last. Robustester Fallback, wenn Wetter- oder
  Aux-Daten nicht aktuell sind.
- Alle Varianten: `experiments/configs.py`.

---

## Ohne Docker (lokale Entwicklung)
```bash
pip install -r requirements.txt        # + torch und chronos-forecasting passend zur GPU
cp .env.example .env                   # Keys eintragen
python -m pipeline.orchestrator        # dieselbe Automatik wie im Container
```

---

## Aufbau des Projekts

Drei Ebenen — versioniert und reproduzierbar:

1. **`loaders/`** — Daten holen & aufbereiten.
2. **`experiments/`** — Backtests (Zero-Shot & Finetuning) über 33 Configs mit den Energy-Arena-Metriken.
3. **`pipeline/`** — Live-Betrieb: `forecast` → `arena` (Submit) → `orchestrator` (Scheduler).

```
code/
├── pipeline/       forecast.py  arena.py  orchestrator.py   (Live: Prognose + Abgabe)
├── loaders/        config grid weather aggregate loads market schedule   (Daten)
├── experiments/    configs data model metrics walkforward score run      (Backtests)
├── results/        gespeicherte Experiment-Ergebnisse (Parquet/CSV, nicht im Docker-Image)
├── reference/      germany_h3_res4.csv  cell_population.csv               (statische Eingaben)
├── docker/         Dockerfile (AMD/NVIDIA)  docker-compose.yml  entrypoint.sh
└── data/           erzeugte Ausgaben (git-ignoriert; via Snapshot sicherbar)
```
Alles UTC, 15-minütig (Aux stündlich). Konstanten/Pfade zentral in `loaders/config.py`.

---

## Datenquellen (`loaders/`)

| # | Quelle | Modul(e) | Ausgabe unter `data/` |
|---|--------|----------|-----------------------|
| 1 | **Wetter je H3-Zelle** (224), Punkt + Ensemble | `weather.py`, `aggregate.py` | `historical/`, `historical_ens/`, `weather_zone/{pop,centroid}/` |
| 2 | **Markt je ÜNB-Zone** (Preis/Solar/Wind/Residual/Last) | `loads.py`, `market.py` | `loads/`, `load_forecast/`, `features/aux_*/` |
| 3 | **Gesamt DE-LU** + **Luxemburg `lu`** | alle | dieselben Ordner, Gebiete `de_lu`/`lu` |

Gebiete: die 4 Regelzonen `50hertz, tennet, amprion, transnetbw`, die Gebotszone
`de_lu` und **Luxemburg `lu`** als eigene Reihe (eigene H3-Zelle + ENTSO-E-`LU`-Last;
Solar/Wind fehlen mangels SMARD-LU-Region). `de_lu` = Σ(4 Zonen) + `lu` (exakt).

Manuelle Loader-Aufrufe (falls nötig, sonst macht der Orchestrator das):
```bash
python -m loaders.schedule --once      # Voll-Backfill
python -m loaders.schedule --refresh   # ein Refresh-Tick (Wetter/Last/Markt/Aggregation)
python -m loaders.grid                 # Kontrolle: Zellen, Gewichte, Centroids
```

---

## Backtests & Ergebnisse (`experiments/`)

33 Config-Familien = **3 Granularitaeten** (whole / region-indep / region-joint) x
**11 Eingabe-Varianten** (C1 nur Last, C2.1/2.2/2.3 Wetter aus 1 Zelle / pop-Mittel /
4 Zellen, C3.1-3.4 Aux als Multitask-Ziel, C4 Wetter + past-only Marktinput,
C5 C4 + Kalender, C6 nur Kalender). Zero-shot, Walk-Forward
2024–2026, Metriken: **RMSE, R², WIS, LQS, MAE-Median, Cov50/95, CRPS, Energy Score**;
Benchmark = ENTSO-E-Day-ahead-Prognose.

```bash
# im GPU-Container:
python -m experiments.run run --config all --start 2024-01-01 --end 2026-06-01 --score
python -m experiments.run finetune --config G1 --start 2024-01-01 --end 2026-06-01 \
  --num-steps 300 --refit-months 3 --train-window-months 12 --out-dir out_ft
python -m experiments.run score --dir out_ft --out experiments/out_ft/results.csv
```

Gespeicherte Ergebnisartefakte fuer die oeffentliche Version liegen getrennt vom Code unter
`results/branch_b/`. Die Trainings-/Scoring-CLI schreibt standardmaessig weiter nach
`experiments/out*`; fuer neue Veroeffentlichungsartefakte kann `--out-dir` auf einen
separaten Ordner gesetzt und das Ergebnis anschliessend nach `results/` kopiert werden.

Die empfohlene Submission nutzt bewusst **Punktwetter-Forecasts als Chronos-Input**,
nicht die ICON-D2-Ensemble-Wetterdaten. Als Output reichen wir weiterhin alle drei
Energy-Arena-Formate ein: `point`, `quantile` und `ensemble`.

Finetuning nutzt Chronos-2-LoRA. Auf ROCm kann `FT_PAD_COVARIATES=2` fuer einzelne
Covariate-Anzahlen noetig sein; das fuegt informationsfreie Null-Covariaten hinzu und
umgeht bekannte HIP-Backward-Crashes.

**Kernbefund:** Chronos-2 schlaegt ENTSO-E probabilistisch klar. Fuer Submission ist
`G1_C5` der Default-Champion (DE-LU-Gesamtmodell mit Punktwetter, past-only
Preis/Solar/Wind und Kalenderfeatures). `G1_C1` bleibt der robuste univariate Fallback,
falls Wetter- oder Aux-Daten nicht aktuell sind.

---

## Lizenz
Apache License 2.0 (siehe `LICENSE`) — dieselbe Lizenz wie das verwendete
Chronos-2-Modell (`amazon/chronos-2`, chronos-forecasting).

---

## Daten-Snapshot (Backup / Weitergabe)
`data/` (~2,8 GB) ist git-ignoriert. Als komprimiertes Archiv sichern (zstd, ~600 MB):
```bash
tar --zstd -cf ../data-snapshot-$(date +%F).tar.zst data/    # erstellen
tar --zstd -xf ../data-snapshot-YYYY-MM-DD.tar.zst           # entpacken -> data/
```
Fehlt `zstd`: `apt/dnf install zstd` bzw. `tar -I zstd -xf …`.
