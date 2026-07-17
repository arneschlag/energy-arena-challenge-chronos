"""Live-Forecast des naechsten Liefertags mit dem Produktions-Modell.

Nutzt dieselbe Chronos-2-Task-Logik wie die Experimente (experiments.model), aber
fuer EIN zukuenftiges Liefertag-Fenster. Liefert je Gebiet die 21 nativen
Quantilkurven der 96 Arena-Liefer-Steps. DE-LU wird komonoton ueber die Gebiete
aggregiert; Arena-Quantile und deterministische Ensemblepfade werden durch direkte
Auswertung der resultierenden Quantilfunktion erzeugt.

Produktions-Config via env ``PROD_CONFIG``. Die beiden Worker setzen explizit
``G1_C1`` bzw. ``G3_C5``; der sichere lokale Default ist die univariate Variante.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from experiments import configs, data, model
from loaders import config as lc

FREQ = "15min"
STEPS = 96
ARENA_TZ = "Europe/Berlin"
lc.load_dotenv()
CTX_DAYS = int(os.environ.get("CTX_DAYS", "63"))
# Produktions-Config: Branch-B-Champion fuer Submission. Wetter darf als known-future
# Covariate in die Zukunft schauen, aber als einfacher Punktforecast-Input; die drei
# Arena-Ausgabeformate (point/quantile/ensemble) werden daraus erst nach Chronos gebaut.
PROD_CONFIG = os.environ.get("PROD_CONFIG", "G1_C1")
CHRONOS_QUANTILE_LEVELS = np.asarray(model.CHRONOS_QUANTILE_LEVELS, dtype=float)
ARENA_QUANTILE_LEVELS = np.asarray([0.025, 0.25, 0.5, 0.75, 0.975], dtype=float)
# Rueckwaertskompatibler Name fuer bestehende Aufrufer. Intern werden die beiden
# Level-Mengen bewusst getrennt benannt.
QUANTILE_LEVELS = ARENA_QUANTILE_LEVELS.tolist()


def _delivery_window(delivery_date=None, gate_hour: int = 9):
    """(cutoff, fut_idx, delivery-mask) fuer einen Arena-Liefertag.

    `delivery_date` darf ein Arena-`target_start` mit Zeitzone sein, z.B.
    ``2026-07-16T00:00:00+02:00``. Der Cutoff liegt um 09:00 Uhr Ortszeit am
    lokalen Vortag. Bewertet/gesendet werden gemaess Arena-Vertrag stets 96 reale,
    aufeinanderfolgende Viertelstunden ab ``target_start``.

    Dadurch umfasst der Forecast an regulaeren Tagen 155 Schritte. Auch an der
    Sommer-/Winterzeitumstellung bleiben es 96 Arena-Schritte; der letzte
    Zeitstempel kann dann in Europe/Berlin vom civilen 23:45-Uhr-Tagesende
    abweichen. Intern sind alle Indizes UTC, damit doppelte oder nicht existente
    lokale Uhrzeiten nicht entstehen.
    """
    if not 0 <= gate_hour <= 23:
        raise ValueError(f"gate_hour muss zwischen 0 und 23 liegen, nicht {gate_hour}")

    if delivery_date is None:
        now_local = pd.Timestamp.now(tz=ARENA_TZ)
        target_local = (now_local + pd.DateOffset(days=1)).normalize()
    else:
        target = pd.Timestamp(delivery_date)
        if target.tzinfo is None:
            raise ValueError("Arena target_start muss eine Zeitzone bzw. einen UTC-Offset enthalten")
        target_local = target.tz_convert(ARENA_TZ)

    if target_local != target_local.normalize():
        raise ValueError(
            "Arena target_start muss lokaler Tagesbeginn in Europe/Berlin sein; "
            f"erhalten: {target_local.isoformat()}"
        )

    previous_local_date = target_local.date() - pd.Timedelta(days=1)
    cutoff_local = pd.Timestamp(previous_local_date, tz=ARENA_TZ) + pd.Timedelta(
        hours=gate_hour
    )
    cutoff = cutoff_local.tz_convert("UTC")
    target_start = target_local.tz_convert("UTC")

    # Der Arena-Horizont ist eine feste Folge von 96 Viertelstunden ab dem durch
    # target_start bezeichneten Instant, auch an DST-Uebergangstagen.
    delivery_idx = pd.date_range(target_start, periods=STEPS, freq=FREQ)
    fut_idx = pd.date_range(cutoff + pd.Timedelta(FREQ), delivery_idx[-1], freq=FREQ)
    delivery_mask = fut_idx.isin(delivery_idx)
    if int(delivery_mask.sum()) != STEPS:
        raise RuntimeError(
            f"ungueltige Arena-Geometrie: {int(delivery_mask.sum())} statt {STEPS} Liefer-Schritte"
        )
    return cutoff, fut_idx, delivery_mask


def zone_quantiles(cfg_name: str = PROD_CONFIG, delivery_date=None):
    """Native Lastquantile je Gebiet plus die 96 Liefer-Zeitstempel.

    Rueckgabe je Gebiet: ``(21, 96)`` in der Reihenfolge von
    :data:`CHRONOS_QUANTILE_LEVELS`. Die erste Achse ist keine Sample-Achse.
    """
    # Der Prozess ist langlebig, die Loader aktualisieren ihre Dateien jedoch
    # stuendlich. Jeder Forecast muss deshalb mit einem frischen Dateisnapshot
    # beginnen.
    data.clear_caches()
    cfg = configs.get(cfg_name)
    cutoff, fut_idx, deliv = _delivery_window(delivery_date)
    ctx_steps = CTX_DAYS * STEPS
    frames = {}
    for a in cfg.areas:
        try:
            _validate_live_inputs(cfg, a, cutoff, fut_idx)
            frames[a] = model.area_frame(cfg, a, include_future=True)
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(f"ungueltige Live-Daten fuer {cfg.name}/{a}: {exc}") from exc
    missing_areas = set(cfg.areas) - set(frames)
    if missing_areas:
        raise RuntimeError(
            f"{cfg.name}: erforderliche Gebiete fehlen: {sorted(missing_areas)}"
        )
    out = {}
    ts = fut_idx[deliv]
    if cfg.gran in ("whole", "indep"):
        tasks, tas = [], []
        for a, (df, tcols, known, past) in frames.items():
            t = model.build_task(df, cutoff, tcols, known, past, ctx_steps, fut_idx)
            if t is None:
                raise RuntimeError(
                    f"{cfg.name}/{a}: Kontext oder erforderliche Forecast-Covariates fehlen"
                )
            tasks.append(t); tas.append(a)
        predictions = model.predict(tasks, len(fut_idx))
        if len(predictions) != len(tasks):
            raise RuntimeError(
                f"Chronos lieferte {len(predictions)} Outputs fuer {len(tasks)} Tasks"
            )
        for a, o in zip(tas, predictions):
            expected_vars = len(frames[a][1])
            if o.shape[0] != expected_vars:
                raise RuntimeError(
                    f"Chronos lieferte fuer {a} {o.shape[0]} statt "
                    f"{expected_vars} Zielvariablen"
                )
            out[a] = np.clip(o[0][:, deliv], 0, None)
    else:                                           # joint
        task, zo = model.build_joint_task(frames, cutoff, ctx_steps, fut_idx)
        if task is None:
            raise RuntimeError(
                f"{cfg.name}: gemeinsamer Kontext oder erforderliche Forecast-Covariates fehlen"
            )
        if zo != list(cfg.areas):
            raise RuntimeError(
                f"{cfg.name}: Joint-Task hat Gebiete {zo}, erwartet {list(cfg.areas)}"
            )
        arr = model.predict([task], len(fut_idx))[0]
        expected_vars = sum(len(frames[a][1]) for a in zo)
        if arr.shape[0] != expected_vars:
            raise RuntimeError(
                f"Chronos lieferte im Joint-Task {arr.shape[0]} statt "
                f"{expected_vars} Zielvariablen"
            )
        pos = 0
        for a in zo:
            nvar = len(frames[a][1])
            out[a] = np.clip(arr[pos][:, deliv], 0, None)
            pos += nvar
    if set(out) != set(cfg.areas):
        raise RuntimeError(
            f"{cfg.name}: Forecast-Gebiete {sorted(out)} stimmen nicht mit den "
            f"erforderlichen Gebieten {sorted(cfg.areas)} ueberein"
        )
    return out, ts


def _validate_live_inputs(cfg: configs.Config, area: str, cutoff: pd.Timestamp,
                          fut_idx: pd.DatetimeIndex) -> None:
    """Harte Frische-/Vollstaendigkeitspruefung vor einem Live-Forecast.

    Targets und past-only Auxiliary Variables muessen den Cutoff erreichen;
    Wetter und sonstige known-future Inputs muessen das ganze Forecastfenster
    abdecken. Luxemburg besitzt in den Paper-Konfigurationen nur den gemeinsamen
    DE-LU-Preis, nicht die deutschen SMARD-Solar-/Windreihen.
    """
    target = data.target(area)
    if cutoff not in target.index or pd.isna(target.loc[cutoff]):
        last = target.dropna().index.max() if not target.dropna().empty else None
        raise ValueError(f"Istlast deckt Cutoff {cutoff} nicht ab (letzter Wert: {last})")

    declared_aux = set(cfg.aux_target) | set(cfg.aux_known) | set(cfg.aux_past)
    required_aux = declared_aux - ({"solar", "wind"} if area == lc.LU else set())
    for name in sorted(required_aux):
        series = data.aux(area, name)
        if series is None or series.dropna().empty:
            raise ValueError(f"Auxiliary Variable {name!r} fehlt")
        # Auxiliary Variables sind stuendlich; der letzte Stundenwert traegt per
        # Forward-Fill die vier zugehoerigen Viertelstunden.
        required_until = (fut_idx[-1].floor("h") if name in cfg.aux_known
                          else cutoff.floor("h"))
        last = series.dropna().index.max()
        if last < required_until:
            raise ValueError(
                f"Auxiliary Variable {name!r} ist veraltet: {last} < {required_until}"
            )

    if cfg.weather:
        weather = data.weather(area, cfg.weather)
        missing = fut_idx.difference(weather.dropna().index)
        if len(missing):
            raise ValueError(
                f"Wetter {cfg.weather!r} deckt Forecastfenster nicht ab; "
                f"erster fehlender Schritt: {missing[0]}"
            )


def zone_samples(cfg_name: str = PROD_CONFIG, delivery_date=None):
    """Kompatibilitaetswrapper; die Rueckgabe enthaelt Quantile, keine Samples."""
    return zone_quantiles(cfg_name=cfg_name, delivery_date=delivery_date)


# --- Quantilfunktion und DE-LU-Kombination der Zonen ------------------------

def _as_monotone_quantiles(values: np.ndarray) -> np.ndarray:
    """Validiert Quantilkurven und ordnet seltenes Quantil-Crossing monoton um."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != len(CHRONOS_QUANTILE_LEVELS):
        raise ValueError(
            "Quantilarray muss Shape "
            f"({len(CHRONOS_QUANTILE_LEVELS)}, horizon) haben, nicht {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise ValueError("Quantilarray enthaelt nicht-endliche Werte")
    # Eine inverse CDF muss monoton sein. Sortieren entlang der Quantilachse ist
    # die uebliche monotone Rearrangement-Korrektur: Anders als cumulative-max
    # behaelt sie die vorhergesagte empirische Verteilung je Zeitschritt bei.
    return np.sort(arr, axis=0)


def _interpolate_quantile_curves(values: np.ndarray,
                                 levels: Sequence[float]) -> np.ndarray:
    """Wertet native Chronos-Quantilkurven an beliebigen Levels aus.

    Ausserhalb des von Chronos gelieferten Bereichs [0.01, 0.99] wird die
    jeweilige Randkurve verwendet. Das betrifft beim 100er-Midpoint-Ensemble nur
    u=0.005 und u=0.995 und vermeidet unkontrollierte Tail-Extrapolation.
    """
    arr = _as_monotone_quantiles(values)
    requested = np.asarray(levels, dtype=float)
    if requested.ndim != 1 or not np.isfinite(requested).all():
        raise ValueError("Quantillevels muessen ein endlicher eindimensionaler Vektor sein")
    if ((requested < 0.0) | (requested > 1.0)).any():
        raise ValueError("Quantillevels muessen im Intervall [0, 1] liegen")
    # np.interp arbeitet eindimensional; die Schleife laeuft nur ueber 96 Steps
    # und macht die gemeinte Quantilachse unmissverstaendlich.
    return np.stack([
        np.interp(requested, CHRONOS_QUANTILE_LEVELS, arr[:, t])
        for t in range(arr.shape[1])
    ], axis=1)


def _validate_zone_quantiles(zquantiles: Mapping[str, np.ndarray]) -> None:
    if not zquantiles:
        raise ValueError("mindestens ein Gebiet mit Quantilkurven wird benoetigt")
    horizons = {np.asarray(q).shape[1] for q in zquantiles.values()
                if np.asarray(q).ndim == 2}
    if len(horizons) != 1 or len(horizons) == 0:
        raise ValueError("alle Gebiete muessen zweidimensionale Quantilarrays gleichen Horizonts haben")
    # Vollstaendige Shape-/Finite-Pruefung samt klarer Fehlermeldung.
    for q in zquantiles.values():
        _as_monotone_quantiles(q)


def delu_quantiles(zquantiles: Mapping[str, np.ndarray],
                   ql: Sequence[float] = ARENA_QUANTILE_LEVELS) -> np.ndarray:
    """DE-LU-Quantile durch direkte, komonotone Quantilaggregation.

    Fuer G1 ist dies direkt die Quantilfunktion des Gesamtmodells. Bei G3 wird
    dasselbe Level in allen gemeinsam modellierten Regionen ausgewertet und
    addiert. Rueckgabe: ``(n_levels, 96)``.
    """
    _validate_zone_quantiles(zquantiles)
    return np.sum(
        [_interpolate_quantile_curves(q, ql) for q in zquantiles.values()],
        axis=0,
    )


def delu_ensemble(zquantiles: Mapping[str, np.ndarray], n: int = 100) -> np.ndarray:
    """Deterministische DE-LU-Pfade via inverse Quantilfunktion.

    Die Midpoint-Levels ``(i + 0.5) / n`` approximieren eine Gleichverteilung ohne
    Zufallszahl oder Bootstrap. Dasselbe ``u`` gilt je Pfad fuer alle Zeitpunkte
    und, insbesondere fuer G3, fuer alle Regionen (komonotone Aggregation).
    Rueckgabe: ``(n, 96)``.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError(f"n muss eine positive Ganzzahl sein, nicht {n!r}")
    _validate_zone_quantiles(zquantiles)
    u = (np.arange(n, dtype=float) + 0.5) / n
    return np.sum(
        [_interpolate_quantile_curves(q, u) for q in zquantiles.values()],
        axis=0,
    )


def delu_point(zquantiles: Mapping[str, np.ndarray]) -> np.ndarray:
    """Punktprognose = Summe der Zonen-Mediane -> (96,)."""
    return delu_quantiles(zquantiles, [0.5])[0]
