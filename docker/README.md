# Drei AMD-Produktionscontainer

Die Compose-Datei trennt Datenzugriff und beide Energy-Arena-Accounts:

| Service | Rolle | Datenzugriff | persistenter Zustand |
|---|---|---|---|
| `ea-data-summary` | Loader und optionales Summary-Modul | read/write | `data-state` |
| `ea-g1-c1` | `amazon/chronos-2` (120M), `G1_C1` strikt univariat | read-only | `g1-c1-state`, `g1-c1-adapters` |
| `ea-g3-c5` | `amazon/chronos-2` (120M), `G3_C5` | read-only | `g3-c5-state`, `g3-c5-adapters` |

Die Worker teilen den HuggingFace-Cache und eine dateibasierte GPU-Sperre. Der
Datenservice nimmt fuer Updates einen exklusiven Daten-Lock; Forecasts halten
waehrend des Laufs einen gemeinsamen Leselock. Ein Worker fuehrt beim Start
absichtlich keine Submission aus.

## Konfiguration

```bash
cp docker/env/data.local.env.example docker/env/data.local.env
cp docker/env/g1-c1.local.env.example docker/env/g1-c1.local.env
cp docker/env/g3-c5.local.env.example docker/env/g3-c5.local.env
```

Danach werden nur in den ignorierten `*.local.env`-Dateien die jeweiligen Tokens
gesetzt. Fuer einen kontrollierten Erststart bleibt `SUBMIT_ENABLED=false`. Nach
Dry-Run und Logpruefung kann es je Worker separat auf `true` gesetzt werden.

Die Scheduler-Minuten sind getrennt in `g1-c1.env` und `g3-c5.env` ueber
`SUBMIT_MINUTE` konfigurierbar. Sie gelten in UTC und enthalten mehrere versetzte
Prueftermine, damit ein partiell fehlgeschlagenes Format noch vor der Deadline
wiederholt werden kann. Der Arena-State verhindert eine zweite erfolgreiche
Abgabe desselben Formats und Liefertags.

Der Datendienst aktualisiert Last stuendlich, Wetter mehrmals vor dem Gate und
Markt-/Auxiliary-Reihen in zwei UTC-Fenstern fuer Sommer- und Winterzeit. Die
gemeinsame Mail wird um 11:55 Uhr `Europe/Berlin` nach den regulaeren
Submissionversuchen erstellt.

Beide Worker besitzen ausserdem getrennte quartalsweise LoRA-Jobs. Standard sind
die Monate 1, 4, 7 und 10, jeweils der erste Tag; Monat, Tag, Stunde und die je
Worker versetzte Minute sind ueber `FINETUNE_*` konfigurierbar.
`REQUIRE_ADAPTER=true` verhindert einen stillen Zero-Shot-Fallback. Vor der ersten
Live-Aktivierung muss deshalb je Worker einmal `python -m pipeline.finetune`
erfolgreich gelaufen sein; der gueltige Adapter liegt danach atomar unter
`/app/adapters/current`.

Das Summary-/Mailmodul wird im Datenservice als Modulreferenz gestartet:

```dotenv
SUMMARY_MODULE=pipeline.status
SUMMARY_ARGS=--send
```

Es liest die getrennten, read-only gemounteten Worker-States aus
`/state/g1/status.json` und `/state/g3/status.json`. Zusaetzlich stehen ueber
`SUMMARY_MODEL_LABELS` eindeutige Bezeichnungen beider Modelle bereit.

## Start und Kontrolle

```bash
REPO_COMMIT="$(git rev-parse HEAD)" \
  docker compose -f docker/docker-compose.yml up -d --build
docker compose -f docker/docker-compose.yml ps
docker compose -f docker/docker-compose.yml logs -f ea-data-summary ea-g1-c1 ea-g3-c5
```

Vor der ersten Live-Aktivierung sollte je Worker ein manueller Dry-Run im
Container erfolgen. Die beiden `ARENA_API_KEY`s duerfen nicht in eingecheckte
Dateien, Compose-Ausgaben oder Logs kopiert werden.
