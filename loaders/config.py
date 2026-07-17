"""Zentrale Konfiguration der Daten-Lade-Pipeline.

Einzige Quelle der Wahrheit fuer Pfade, Zonen, ENTSO-E-/SMARD-Codes und
Wettervariablen. Ersetzt die frueher ueber mehrere Skripte kopierten Konstanten
(load_dotenv, TSO_AREAS, year_bounds, WEATHER_VARS).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# --- Pfade ------------------------------------------------------------------
# code/ (Paket liegt unter code/loaders/, also eine Ebene hoch)
BASE = Path(__file__).resolve().parents[1]
REFERENCE = BASE / "reference"                 # statische Eingaben (Grid, Bevoelkerung)
DATA = BASE / "data"                           # erzeugte Ausgaben

CELLS_CSV = REFERENCE / "germany_h3_res4.csv"  # h3_index, landkreis, bundesland, tso, lat, lng
POP_CSV = REFERENCE / "cell_population.csv"     # h3_index, bundesland, tso, population

DATA_HIST = DATA / "historical"                # Wetter-Punktprognose je Zelle/Jahr
DATA_HIST_ENS = DATA / "historical_ens"        # Wetter-Ensemble je Zelle/Datum
DATA_WEATHER_ZONE = DATA / "weather_zone"      # aggregiertes Zonen-Wetter (pop/ + centroid/)
DATA_LOADS = DATA / "loads"                    # ENTSO-E Ist-Last (Zonen + de_lu)
DATA_FCAST = DATA / "load_forecast"            # ENTSO-E Day-ahead-Prognose (Benchmark)
DATA_FEATURES = DATA / "features"              # aux_price/solar/wind/residual je Zone
DATA_ANALYSIS = DATA / "analysis"              # Diagnose-Ausgaben (Diffs, Reconciliation)

# --- Zonen ------------------------------------------------------------------
# Die 4 deutschen ÜNB-Regelzonen (regionale Modelle) ...
ZONES = ["50hertz", "tennet", "amprion", "transnetbw"]
# ... plus die kombinierte Gebotszone DE-LU (Gesamtmarkt, inkl. Luxemburg).
DELU = "de_lu"
# ... plus Luxemburg als eigene Reihe (eigene Wetterzelle + ENTSO-E LU-Last).
LU = "lu"
ALL_AREAS = ZONES + [DELU]
# Gebiete mit Wetter (inkl. LU als Solo-Zelle) — fuer weather/aggregate.
AGG_AREAS = ZONES + [DELU, LU]

# Luxemburg-eigene H3-Zelle (Auflösung 4, Landes-Schwerpunkt; nicht im dt. Grid).
LU_CELL = {"h3_index": "841fa3dffffffff", "lat": 49.815, "lng": 6.13,
           "population": 672050}

# ÜNB-Key -> Anzeigename (wie in germany_h3_res4.csv "tso"-Spalte)
ZONE_LABEL = {"50hertz": "50Hertz", "tennet": "TenneT",
              "amprion": "Amprion", "transnetbw": "TransnetBW", "lu": "LU"}
# Anzeigename -> ÜNB-Key (Umkehrung, fuer CSV-Joins)
LABEL_TO_ZONE = {v: k for k, v in ZONE_LABEL.items()}

# ÜNB-Key -> entsoe-py Area-Code (de_lu = kombinierte Gebotszone, lu = Creos/LU)
ENTSOE_AREA = {"50hertz": "DE_50HZ", "tennet": "DE_TENNET",
               "amprion": "DE_AMPRION", "transnetbw": "DE_TRANSNET",
               "de_lu": "DE_LU", "lu": "LU"}
# ÜNB-Key -> SMARD-Regionsname
SMARD_REGION = {"50hertz": "50Hertz", "amprion": "Amprion",
                "tennet": "TenneT", "transnetbw": "TransnetBW"}

# --- Wetter -----------------------------------------------------------------
WEATHER_VARS = ["temperature_2m", "apparent_temperature", "is_day",
                "shortwave_radiation", "wind_speed_10m", "relative_humidity_2m",
                "dew_point_2m"]

# --- Zeit -------------------------------------------------------------------
TZ = "UTC"                            # GMT+0 fuer alle Reihen (Wetter == Last)
DEFAULT_YEARS = [2023, 2024, 2025, 2026]
HISTORICAL_2026_END = "2026-06-01"    # 2026 historisch nur bis zu diesem Stichtag


def year_bounds(year: int) -> tuple[str, str]:
    """Start-/Enddatum (Strings) fuer ein Jahr; 2026 nur bis Stichtag.
    Ende ist inklusiv (Open-Meteo-Konvention: end_date einschliessend)."""
    start = f"{year}-01-01"
    end = HISTORICAL_2026_END if year == 2026 else f"{year}-12-31"
    return start, end


def year_bounds_ts(year: int):
    """tz-aware Start/Ende (UTC, Ende exklusiv) fuer ENTSO-E-Requests."""
    import pandas as pd
    start = pd.Timestamp(f"{year}-01-01", tz=TZ)
    end_str = HISTORICAL_2026_END if year == 2026 else f"{year + 1}-01-01"
    return start, pd.Timestamp(end_str, tz=TZ)


def load_dotenv(path: str | os.PathLike | None = None) -> None:
    """Minimaler .env-Loader (KEY=VALUE-Zeilen), ohne Abhaengigkeit.
    Setzt nur, was noch nicht in der Umgebung steht (setdefault)."""
    p = Path(path) if path else (BASE / ".env")
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def entsoe_key() -> str:
    """ENTSO-E-Token aus Umgebung/.env holen (leer, wenn nicht gesetzt)."""
    load_dotenv()
    return os.environ.get("ENTSOE_API_KEY", "")


def atomic_to_csv(frame, path: str | os.PathLike, **kwargs) -> None:
    """CSV im selben Verzeichnis schreiben und erst vollstaendig ersetzen.

    Damit bleibt bei Containerabbruch oder vollem Dateisystem immer die letzte
    vollstaendige Version sichtbar. Der zusaetzliche Data-Lock koordiniert
    weiterhin mehrere Dateien eines Refresh-Laufs.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, **kwargs)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
