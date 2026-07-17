# Day-Ahead-Lastprognose DE-LU - Chronos-2-Pipeline

Vollautomatische Day-ahead-Lastprognose für die **DE-LU-Gebotszone** mit
**`amazon/chronos-2` (120 Mio. Parameter)**. Die AMD-Produktionsumgebung besteht
aus einem CPU-Datendienst und
zwei getrennten GPU-Workern: dem strikt univariaten **G1_C1** sowie **G3_C5** mit
Group Attention, Wetter-, Auxiliary- und Kalenderdaten. Beide Worker verwenden
eigene Energy-Arena-Accounts, Zustände und LoRA-Adapter.

---

## Schnellstart - drei AMD-Services

### Voraussetzungen
- **Docker** mit Docker Compose unter Linux.
- Eine ROCm-faehige **AMD-GPU** mit Zugriff auf `/dev/kfd` und `/dev/dri`.
- **ENTSO-E-API-Key** (kostenlos): https://transparency.entsoe.eu → *Account Settings → Web Api Security Token*.
- Je ein **Energy-Arena-API-Key** fuer G1_C1 und G3_C5.

### In 3 Schritten

**1) Repo holen und lokale Konfigurationsdateien anlegen**

Linux / macOS:
```bash
git clone https://github.com/arneschlag/energy-arena-challenge-chronos.git
cd energy-arena-challenge-chronos
cp docker/env/data.local.env.example docker/env/data.local.env
cp docker/env/g1-c1.local.env.example docker/env/g1-c1.local.env
cp docker/env/g3-c5.local.env.example docker/env/g3-c5.local.env
```

In `data.local.env` kommt nur der ENTSO-E-Key sowie gegebenenfalls die
SMTP-Konfiguration. Die beiden Arena-Keys kommen getrennt nach
`g1-c1.local.env` beziehungsweise `g3-c5.local.env`. Echte Secrets werden weder
committet noch in das Image kopiert.

`SUBMIT_ENABLED` bleibt fuer den Erststart in beiden Worker-Dateien auf `false`.
So laufen Forecast und API-Validierung als **Dry-Run**, ohne eine Live-Abgabe.

**2) Die drei Container starten**

```bash
REPO_COMMIT="$(git rev-parse HEAD)" \
  docker compose -f docker/docker-compose.yml up -d --build
docker compose -f docker/docker-compose.yml ps
docker compose -f docker/docker-compose.yml logs -f \
  ea-data-summary ea-g1-c1 ea-g3-c5
```

`ea-data-summary` ist der einzige Prozess mit Schreibzugriff auf `data/`. Die
beiden Worker mounten dieselben Daten read-only. Getrennte State-/Adapter-Volumes,
ein gemeinsamer HuggingFace-Cache und ein gemeinsamer GPU-Lock bleiben ueber
Neustarts erhalten.

**3) Initial trainieren, Dry-Run pruefen und Live-Abgabe freigeben**

Nach dem erfolgreichen Daten-Backfill werden die beiden Adapter einmal manuell
und nacheinander erzeugt:

```bash
docker compose -f docker/docker-compose.yml exec ea-g1-c1 python -m pipeline.finetune
docker compose -f docker/docker-compose.yml exec ea-g3-c5 python -m pipeline.finetune
```

`REQUIRE_ADAPTER=true` verhindert, dass ein fehlender oder ungueltiger Adapter
unbemerkt als Zero-Shot-Modell weiterlaeuft. Erst wenn beide Fine-Tunings und ein
Dry-Run erfolgreich waren, wird in der jeweiligen `*.local.env` separat
`SUBMIT_ENABLED=true` gesetzt und nur der betreffende Worker neu erstellt:

```bash
docker compose -f docker/docker-compose.yml up -d --force-recreate ea-g1-c1 ea-g3-c5
```

Auch nach einem Neustart erfolgt keine sofortige Abgabe; der erste Versuch wartet
auf den naechsten konfigurierten Scheduler-Termin.

### Was dann automatisch passiert
| Zeit | Job |
|---|---|
| stündlich `:45` UTC | `ea-data-summary`: ENTSO-E-Last aktualisieren |
| `06:00`, `08:00`, `09:00` UTC | Wetter aktualisieren und regional aggregieren |
| `08:20`, `09:20` UTC | Preis/Solar/Wind aktualisieren (Sommer- und Winterzeitfenster) |
| `:00`, `:20`, `:40` UTC | `ea-g1-c1`: G1_C1 pruefen und bei Faelligkeit prognostizieren/einreichen |
| `:10`, `:30`, `:50` UTC | `ea-g3-c5`: G3_C5 pruefen und bei Faelligkeit prognostizieren/einreichen |
| täglich `11:55` Europe/Berlin | Gemeinsame Statusmail mit getrennten Modellabschnitten |
| 1. Jan./Apr./Jul./Okt. `07:10` | Quartalsweises G1_C1-LoRA-Fine-Tuning (UTC) |
| 1. Jan./Apr./Jul./Okt. `07:30` | Quartalsweises G3_C5-LoRA-Fine-Tuning (UTC) |

Die Zeiten sind in den nicht geheimen Dateien unter `docker/env/` konfigurierbar.
Der GPU-Lock serialisiert Forecasts und Fine-Tunings beider Worker; ein
Data-Read/Write-Lock verhindert Prognosen auf halb aktualisierten Dateien.

### Fest zugeordnete Modelle

Beide Rollen verwenden exakt `amazon/chronos-2`. Dieses Chronos-2-Modell hat
120 Mio. Parameter; das groessere `chronos-t5-large` gehoert zur vorherigen
Chronos-Generation und unterstuetzt den nativen G3-/Group-Attention-Pfad nicht.

- **`ea-g1-c1` / `G1_C1`**: eine DE-LU-Lastreihe; keine Wetter- oder
  Auxiliary-Loader im Worker.
- **`ea-g3-c5` / `G3_C5`**: gemeinsames multivariates Modell der Regionen mit
  Group Attention sowie den C5-Variablen.
- Weitere Ablationsvarianten bleiben fuer reproduzierbare Experimente in
  `experiments/configs.py`, werden aber nicht produktiv geplant.

Chronos-2 liefert 21 native Quantilkurven und keine 21 Ensemble-Samples. Die
Arena-Quantile werden deshalb direkt aus der Quantilfunktion interpoliert; die
100 Ensemblepfade entstehen deterministisch an 100 gleichmaessigen
inverse-CDF-Raengen. Fuer G3 wird derselbe Rang ueber Zeit und Regionen verwendet.
Diese komonotone Kopplung ist eine explizite Abhaengigkeitsannahme, nicht ein vom
Modell ausgegebenes gemeinsames Sample.

---

## Ohne Docker (lokale Entwicklung)

Die Produktionsrollen bleiben auch lokal getrennte Prozesse. Beispielsweise in
drei Terminals (mit jeweils passender Umgebung):

```bash
pip install -r requirements.txt        # + torch und chronos-forecasting passend zur GPU
python -m pipeline.data_orchestrator
PROD_CONFIG=G1_C1 python -m pipeline.worker_orchestrator
PROD_CONFIG=G3_C5 python -m pipeline.worker_orchestrator
```

Der alte gemischte `pipeline.orchestrator` ist nicht fuer die Drei-Service-
Produktion vorgesehen.

---

## Aufbau des Projekts

Drei Ebenen — versioniert und reproduzierbar:

1. **`loaders/`** — Daten holen & aufbereiten.
2. **`experiments/`** — Backtests (Zero-Shot & Finetuning) über 33 Configs mit den Energy-Arena-Metriken.
3. **`pipeline/`** — Live-Betrieb: Forecast/Arena, getrennte Worker-/Daten-Scheduler,
   LoRA-Fine-Tuning und Statusmail.

```
code/
├── pipeline/       forecast, arena, finetune, worker/data orchestrator, status
├── loaders/        config grid weather aggregate loads market schedule   (Daten)
├── experiments/    configs data model metrics walkforward score run      (Backtests)
├── results/        gespeicherte Experiment-Ergebnisse (Parquet/CSV, nicht im Docker-Image)
├── reference/      germany_h3_res4.csv  cell_population.csv               (statische Eingaben)
├── docker/         AMD-Dockerfile  Drei-Service-Compose  rollenbasierter Entrypoint
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

G3_C5 nutzt bewusst **Punktwetter-Forecasts als Chronos-Input**, nicht die
ICON-D2-Ensemble-Wetterdaten. G1_C1 arbeitet ausschliesslich mit der Lastreihe.
Beide Worker reichen die drei Energy-Arena-Formate `point`, `quantile` und
`ensemble` aus derselben nativen Chronos-Quantilverteilung ein.

Finetuning nutzt Chronos-2-LoRA. Auf ROCm kann `FT_PAD_COVARIATES=2` fuer einzelne
Covariate-Anzahlen noetig sein; das fuegt informationsfreie Null-Covariaten hinzu und
umgeht bekannte HIP-Backward-Crashes.

**Produktionsauswahl:** `G3_C5` ist die multivariate Paper-Konfiguration mit dem
groessten beobachteten Fine-Tuning-Nutzen. `G1_C1` laeuft getrennt als strikt
univariate Referenz, die keine Wetter- oder Auxiliary-Daten benoetigt.

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
