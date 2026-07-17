"""Chronos-2-Pipeline, Task-Bau und Vorhersage.

Braucht `chronos` + GPU (nur im chronos-Container). Der chronos-Import ist lazy,
damit configs/data/metrics ohne chronos importierbar/testbar bleiben.

Task-Schema (Chronos-2):
  {"target": (hist,) oder (n_var, hist),
   "past_covariates":   {name: (hist,)},        # known-future (Kontextteil) + past-only
   "future_covariates": {name: (horizon,)}}     # nur known-future (Teilmenge von past)
predict([task], prediction_length=H) -> Liste von
(n_var, 21 native Quantile, H).
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import configs, data

FREQ = "15min"
MODEL = "amazon/chronos-2"
MODEL_PARAMETER_COUNT = 120_000_000
_PIPE = None
_PIPE_METADATA = None
FEATURE_TZ = "Europe/Berlin"

# Expliziter Produktionsvertrag fuer alle erzeugten und geladenen Adapter. Die
# Werte entsprechen dem LoRA-Default von chronos-forecasting 2.2.2, werden hier
# aber bewusst festgeschrieben, damit Training und Inferenz nicht auseinander
# driften koennen.
LORA_R = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = (
    "self_attention.q",
    "self_attention.v",
    "self_attention.k",
    "self_attention.o",
    "output_patch_embedding.output_layer",
)

# Chronos-2 gibt in der hier eingesetzten QUANTILES-Konfiguration genau diese
# Quantilkurven zurueck. Die mittlere Achse eines Predict-Outputs ist daher keine
# Sample-/Ensemble-Achse. Die explizite Zuordnung verhindert, dass die 21 Werte
# spaeter noch einmal mit np.quantile ausgewertet oder als Bootstrap-Population
# behandelt werden.
CHRONOS_QUANTILE_LEVELS = (
    0.01,
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
    0.99,
)

def _easter_sunday(year: int) -> date:
    """Gregorianisches Osterdatum nach Meeus/Jones/Butcher."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = (h + ell - 7 * m + 114) % 31 + 1
    return date(year, month, day)


@lru_cache(maxsize=None)
def _de_holidays(year: int) -> frozenset[date]:
    """Bundesweite Feiertage plus lastreduzierte Tage 24./31. Dezember."""
    easter = _easter_sunday(year)
    return frozenset({
        date(year, 1, 1),
        easter - timedelta(days=2),                 # Karfreitag
        easter + timedelta(days=1),                 # Ostermontag
        date(year, 5, 1),
        easter + timedelta(days=39),                # Christi Himmelfahrt
        easter + timedelta(days=50),                # Pfingstmontag
        date(year, 10, 3),
        date(year, 12, 24),
        date(year, 12, 25),
        date(year, 12, 26),
        date(year, 12, 31),
    })


def _calendar_feature(index, name):
    """Kalender-Covariate aus einem UTC-Index nach deutscher Ortszeit.

    Die Messreihen bleiben intern in UTC. Feiertag, Wochenende und Wochentag
    sind jedoch zivile Merkmale und muessen daher in ``Europe/Berlin``
    ausgewertet werden.
    """
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(FEATURE_TZ)
    if name in ("holiday", "bridge", "pre_holiday", "post_holiday", "xmas"):
        # Python-date statt Timestamp-Arithmetik: So bleibt der vorherige bzw.
        # folgende Kalendertag auch an DST-Wechseltagen eindeutig.
        days = list(idx.date)
        one = timedelta(days=1)

        def hol(day):
            return day in _de_holidays(day.year)

        vals = {}
        for d in set(days):
            prev, nxt = d - one, d + one
            if name == "holiday":
                v = hol(d)
            elif name == "pre_holiday":                # Tag vor Feiertag (selbst keiner)
                v = not hol(d) and hol(nxt)
            elif name == "post_holiday":               # Tag nach Feiertag (selbst keiner)
                v = not hol(d) and hol(prev)
            elif name == "bridge":                     # Brueckentag: Werktag zwischen
                v = (d.weekday() < 5 and not hol(d)    # Feiertag und Wochenende/Feiertag
                     and ((hol(prev) and (nxt.weekday() >= 5 or hol(nxt)))
                          or (hol(nxt) and (prev.weekday() >= 5 or hol(prev)))))
            else:                                      # xmas: 24.12. bis 01.01.
                v = (d.month == 12 and d.day >= 24) or (d.month == 1 and d.day == 1)
            vals[d] = 1.0 if v else 0.0
        return np.array([vals[d] for d in days], dtype="float32")
    if name == "weekend":
        return (idx.dayofweek >= 5).astype("float32")
    if name in ("dow_sin", "dow_cos"):                 # Wochentag zyklisch (7-Tage-Periode)
        dow = idx.dayofweek.to_numpy()
        fn = np.sin if name == "dow_sin" else np.cos
        return fn(2 * np.pi * dow / 7).astype("float32")
    if name in ("doy_sin", "doy_cos"):                 # Tag im Jahr zyklisch (Jahres-Saison)
        doy = idx.dayofyear.to_numpy()
        fn = np.sin if name == "doy_sin" else np.cos
        return fn(2 * np.pi * doy / 365.25).astype("float32")
    raise ValueError(f"unbekanntes Kalender-Feature {name!r}")


def _adapter_path() -> Path | None:
    raw = os.environ.get("ADAPTER_PATH", "").strip()
    if raw:
        return Path(raw)
    root = os.environ.get("ADAPTER_DIR", "").strip()
    return Path(root) / "current" if root else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def validate_lora_checkpoint(path: Path | str, *, expected_config: str | None = None,
                             require_manifest: bool = False) -> dict:
    """Validiere einen Chronos-2-LoRA-Checkpoint, bevor er geladen wird.

    Neben Dateien und LoRA-Hyperparametern werden Basismodell, Target Modules
    und -- sofern vorhanden -- das Produktionsmanifest geprueft. So kann ein
    Worker nicht unbemerkt den Adapter der anderen Produktionsvariante laden.
    """
    adapter = Path(path)
    cfg_path = adapter / "adapter_config.json"
    weights = adapter / "adapter_model.safetensors"
    manifest_path = adapter / "production_manifest.json"
    if not cfg_path.is_file() or not weights.is_file() or weights.stat().st_size == 0:
        raise RuntimeError(f"LoRA-Adapter unvollstaendig: {adapter}")

    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"ungueltige Adapter-Konfiguration: {adapter}") from exc

    if str(cfg.get("peft_type", "")).upper() != "LORA":
        raise RuntimeError(f"Adapter {adapter} ist kein LoRA-Adapter")
    if int(cfg.get("r", -1)) != LORA_R or int(cfg.get("lora_alpha", -1)) != LORA_ALPHA:
        raise RuntimeError(
            f"LoRA-Adapter {adapter} hat nicht die erwartete Konfiguration "
            f"r={LORA_R}/alpha={LORA_ALPHA}"
        )
    if set(cfg.get("target_modules") or ()) != set(LORA_TARGET_MODULES):
        raise RuntimeError(f"LoRA-Adapter {adapter} hat unerwartete Target Modules")
    if cfg.get("base_model_name_or_path") != MODEL:
        raise RuntimeError(
            f"LoRA-Adapter {adapter} gehoert nicht zum erwarteten Basismodell {MODEL}"
        )

    digest = _sha256(weights)
    metadata = {
        "base_model": MODEL,
        "parameter_count": MODEL_PARAMETER_COUNT,
        "training_mode": "LoRA",
        "adapter_digest": digest,
    }
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"ungueltiges Produktionsmanifest: {adapter}") from exc
        checks = {
            "base_model": MODEL,
            "training_mode": "LoRA",
            "rank": LORA_R,
            "alpha": LORA_ALPHA,
            "adapter_digest": digest,
        }
        for key, expected in checks.items():
            if manifest.get(key) != expected:
                raise RuntimeError(
                    f"Produktionsmanifest des Adapters stimmt bei {key!r} nicht ueberein"
                )
        manifest_config = manifest.get("config")
        if expected_config and manifest_config != expected_config:
            raise RuntimeError(
                f"Adapter ist fuer {manifest_config!r}, Worker erwartet {expected_config!r}"
            )
        metadata["config"] = manifest_config
    elif require_manifest:
        raise RuntimeError(f"Produktionsmanifest fehlt im LoRA-Adapter {adapter}")
    return metadata


def _validate_native_levels(pipeline) -> None:
    actual = tuple(float(x) for x in getattr(pipeline, "quantiles", ()))
    if (len(actual) != len(CHRONOS_QUANTILE_LEVELS)
            or not np.allclose(actual, CHRONOS_QUANTILE_LEVELS, rtol=0.0, atol=1e-12)):
        raise RuntimeError(
            "Chronos meldet unerwartete native Quantillevels; "
            f"erwartet {CHRONOS_QUANTILE_LEVELS}, erhalten {actual}"
        )


def pipe():
    """Lazy Chronos-2 pipeline with an optional, validated LoRA adapter.

    ``REQUIRE_ADAPTER=true`` turns a missing or invalid adapter into a hard
    startup/forecast error.  This prevents a nominally fine-tuned production
    worker from silently falling back to the zero-shot base model.
    """
    global _PIPE, _PIPE_METADATA
    if _PIPE is None:
        from chronos import Chronos2Pipeline
        device = os.environ.get("DEVICE", "cuda")
        # Erst nach vollstaendiger Adapter-/Quantilvalidierung global
        # veroeffentlichen. Andernfalls koennte ein fehlgeschlagener
        # REQUIRE_ADAPTER-Start bei einem spaeteren Aufruf unbemerkt die bereits
        # geladene Zero-Shot-Basis zurueckgeben.
        adapter = _adapter_path()
        require_adapter = os.environ.get("REQUIRE_ADAPTER", "false").lower() == "true"
        metadata = {
            "base_model": MODEL,
            "parameter_count": MODEL_PARAMETER_COUNT,
            "training_mode": "zero-shot",
            "adapter_digest": None,
        }
        if adapter is not None and adapter.is_dir():
            adapter_metadata = validate_lora_checkpoint(
                adapter,
                expected_config=os.environ.get("PROD_CONFIG", "").strip() or None,
                require_manifest=require_adapter,
            )
            # Chronos 2.2.2 erkennt PEFT-Adapter selbst, laedt das referenzierte
            # Basismodell und merged den Adapter. Ein manuelles PeftModel-Wrapping
            # erhaelt den erwarteten Chronos2Model-Vertrag nicht verlaesslich.
            candidate = Chronos2Pipeline.from_pretrained(str(adapter), device_map=device)
            metadata.update(adapter_metadata)
        elif require_adapter:
            raise RuntimeError(f"REQUIRE_ADAPTER=true, aber kein Adapter unter {adapter}")
        else:
            candidate = Chronos2Pipeline.from_pretrained(MODEL, device_map=device)
        _validate_native_levels(candidate)
        _PIPE = candidate
        _PIPE_METADATA = metadata
    return _PIPE


def active_model_metadata() -> dict:
    """Return non-secret metadata for status mail and deployment checks."""
    pipe()
    return dict(_PIPE_METADATA or {})


def reset_pipe() -> None:
    """Release the singleton, primarily before/after a standalone fine-tune."""
    global _PIPE, _PIPE_METADATA
    old = _PIPE
    _PIPE = None
    _PIPE_METADATA = None
    if old is not None:
        del old
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# --- Spalten je (Config, Gebiet) --------------------------------------------

def area_columns(cfg: configs.Config, area: str):
    """Verfuegbare Aux filtern und Spaltengruppen liefern:
    (weather_source, target_aux, known_aux, past_aux)."""
    ta = [a for a in cfg.aux_target if data.has_aux(area, a)]
    ka = [a for a in cfg.aux_known if data.has_aux(area, a)]
    pa = [a for a in cfg.aux_past if data.has_aux(area, a)]
    return cfg.weather, ta, ka, pa


def area_frame(cfg: configs.Config, area: str, include_future: bool = False):
    """DataFrame + Spaltenlisten fuer ein Gebiet.
    Returns (df, target_cols, known_cov, past_cov)."""
    w, ta, ka, pa = area_columns(cfg, area)
    df = data.frame(area, w, tuple(ta + ka + pa), include_future=include_future)
    target_cols = ["target"] + [f"aux_{a}" for a in ta]
    weather_cols = [c for c in df.columns if c != "target" and not c.startswith("aux_")]
    known_cov = weather_cols + [f"aux_{a}" for a in ka]
    past_cov = [f"aux_{a}" for a in pa]
    for name in cfg.calendar:                          # known-future Kalender-Features
        col = f"cal_{name}"
        df[col] = _calendar_feature(df.index, name)
        known_cov.append(col)
    # FT_PAD_COVARIATES (env): n konstante Null-Covariaten anhaengen. Informationsfrei;
    # einziger Zweck: die Covariaten-ANZAHL verschieben, wenn eine Config einen kaputten
    # ROCm-Kernel-Bucket trifft (deterministischer HIP-Crash im LoRA-Backward).
    for i in range(int(os.environ.get("FT_PAD_COVARIATES", "0") or 0)):
        col = f"cal_pad{i}"
        df[col] = np.float32(0.0)
        known_cov.append(col)
    return df, target_cols, known_cov, past_cov


# --- Task-Bau ----------------------------------------------------------------

def build_task(df: pd.DataFrame, cutoff: pd.Timestamp, target_cols, known_cov, past_cov,
               ctx_steps: int, fut_idx: pd.DatetimeIndex):
    """Ein Chronos-2-Task fuer einen Liefertag. None, wenn Kontext/Zukunft fehlt.
    Leckage-sicher: target nur bis cutoff; Zukunft nur Covariaten."""
    ctx = df.loc[:cutoff].tail(ctx_steps)
    if len(ctx) < 96:                                   # zu wenig Kontext
        return None
    need_fut = known_cov                                # future_covariates muessen existieren
    if need_fut and not fut_idx.isin(df.index).all():
        return None
    fut = df.reindex(fut_idx)
    tgt = ctx[target_cols].to_numpy("float32")
    target = tgt[:, 0] if tgt.shape[1] == 1 else tgt.T   # (hist,) | (n_var, hist)
    task = {"target": target}
    past = {c: ctx[c].to_numpy("float32") for c in known_cov + past_cov}
    if past:
        task["past_covariates"] = past
    if known_cov:
        task["future_covariates"] = {c: fut[c].to_numpy("float32") for c in known_cov}
    return task


def build_joint_task(frames: dict, cutoff, ctx_steps, fut_idx):
    """Ein multivariates Task ueber mehrere Zonen (G3). frames: {area: (df, tcols, known,
    past)}. Ziel = alle Zonen-Lasten (+ evtl. Aux-Ziele je Zone) gestapelt; Covariaten
    zonen-praefixiert. Returns (task, zone_order) oder (None, []).
    Alle Zonen werden auf den gemeinsamen 15-min-Index innergejoint."""
    # gemeinsamer Kontext-Index
    common = None
    for area, (df, *_ ) in frames.items():
        idx = df.loc[:cutoff].tail(ctx_steps).index
        common = idx if common is None else common.intersection(idx)
    if common is None or len(common) < 96:
        return None, []
    zone_order = list(frames.keys())
    tgt_stack, past, future = [], {}, {}
    for area in zone_order:
        df, tcols, known, pastc = frames[area]
        if not fut_idx.isin(df.index).all() and known:
            return None, []
        ctx = df.loc[common]
        for tc in tcols:                                # load + aux-Ziele je Zone
            tgt_stack.append(ctx[tc].to_numpy("float32"))
        fut = df.reindex(fut_idx)
        for c in known:
            past[f"{area}__{c}"] = ctx[c].to_numpy("float32")
            future[f"{area}__{c}"] = fut[c].to_numpy("float32")
        for c in pastc:
            past[f"{area}__{c}"] = ctx[c].to_numpy("float32")
    task = {"target": np.stack(tgt_stack, axis=0)}      # (n_var, hist)
    if past:
        task["past_covariates"] = past
    if future:
        task["future_covariates"] = future
    return task, zone_order


# --- Vorhersage --------------------------------------------------------------

def predict(tasks, horizon, batch_size=256, model_pipe=None):
    """Liste von Tasks -> Liste von ``(n_var, 21, horizon)``-Arrays.

    Achse 1 entspricht in dieser Reihenfolge den festen Levels aus
    :data:`CHRONOS_QUANTILE_LEVELS`; sie enthaelt keine Ensemble-Samples.
    model_pipe: optional eine finetunte Pipeline (sonst die Zero-Shot-Basis)."""
    p = model_pipe if model_pipe is not None else pipe()
    outs = p.predict(tasks, prediction_length=horizon, batch_size=batch_size)
    arrays = [o.float().cpu().numpy() for o in outs]
    for i, arr in enumerate(arrays):
        if arr.ndim != 3:
            raise ValueError(
                f"Chronos-Output {i} hat Shape {arr.shape}; erwartet wird "
                "(n_var, n_quantile, horizon)."
            )
        if arr.shape[1] != len(CHRONOS_QUANTILE_LEVELS):
            raise ValueError(
                f"Chronos-Output {i} enthaelt {arr.shape[1]} statt "
                f"{len(CHRONOS_QUANTILE_LEVELS)} nativen Quantilen. "
                "Quantillevels und Modellkonfiguration muessen gemeinsam "
                "aktualisiert werden."
            )
        if arr.shape[2] != horizon:
            raise ValueError(
                f"Chronos-Output {i} hat Horizont {arr.shape[2]} statt {horizon}."
            )
    return arrays


# --- Finetuning (LoRA auf 2023) ---------------------------------------------

def build_fit_task(df: pd.DataFrame, train_end: pd.Timestamp, target_cols, known_cov, past_cov,
                   train_start=None):
    """Fit-Task aus dem Trainingszeitraum (< train_end). Ganze Reihe; Chronos-2 sampelt
    intern Fenster. future_covariates=None markiert known-future.
    train_start: optionaler Fensterbeginn (Rolling Window statt Expanding)."""
    tr = df.loc[train_start:train_end] if train_start is not None else df.loc[:train_end]
    if len(tr) < 96 * 70:                               # < ~70 Tage -> zu wenig
        return None
    tgt = tr[target_cols].to_numpy("float32")
    task = {"target": tgt[:, 0] if tgt.shape[1] == 1 else tgt.T}
    past = {c: tr[c].to_numpy("float32") for c in known_cov + past_cov}
    if past:
        task["past_covariates"] = past
    if known_cov:
        task["future_covariates"] = {c: None for c in known_cov}
    return task


def build_joint_fit_task(frames: dict, train_end, train_start=None):
    common = None
    for _, (df, *_ ) in frames.items():
        idx = (df.loc[train_start:train_end] if train_start is not None
               else df.loc[:train_end]).index
        common = idx if common is None else common.intersection(idx)
    if common is None or len(common) < 96 * 70:
        return None
    tgt_stack, past, future = [], {}, {}
    for area, (df, tcols, known, pastc) in frames.items():
        tr = df.loc[common]
        for tc in tcols:
            tgt_stack.append(tr[tc].to_numpy("float32"))
        for c in known:
            past[f"{area}__{c}"] = tr[c].to_numpy("float32")
            future[f"{area}__{c}"] = None
        for c in pastc:
            past[f"{area}__{c}"] = tr[c].to_numpy("float32")
    task = {"target": np.stack(tgt_stack, axis=0)}
    if past:
        task["past_covariates"] = past
    if future:
        task["future_covariates"] = future
    return task


def fit_lora(fit_tasks, horizon, ctx_steps, num_steps=500, lr=1e-5,
             batch_size=8, out_dir=None):
    """LoRA-Finetuning; gibt die finetunte Pipeline zurueck (direkt nutzbar).
    tf32=False fuer ROCm/AMD-Kompatibilitaet.
    FT_BATCH_SIZE (env) uebersteuert batch_size (Auto-Retry bei HIP/OOM: 8->4).
    FT_SEED (env) setzt Trainings-Seed (Adapter-Init, Sampling, Trainer) fuer
    reproduzierbare, aber zwischen Wiederholungslaeufen unterschiedliche Ergebnisse."""
    batch_size = int(os.environ.get("FT_BATCH_SIZE", batch_size))
    lr = float(os.environ.get("FT_LR", lr))              # Stabilitaets-Fallback bei NaN-Loss
    extra = {}
    seed_env = os.environ.get("FT_SEED")
    if seed_env:                                         # leer/None -> Default-Seed
        import torch
        seed = int(seed_env)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        extra["seed"] = seed                             # -> HF TrainingArguments.seed
    lora_config = {
        "r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "target_modules": list(LORA_TARGET_MODULES),
    }
    return pipe().fit(fit_tasks, prediction_length=horizon, finetune_mode="lora",
                      lora_config=lora_config,
                      context_length=ctx_steps, learning_rate=lr, num_steps=num_steps,
                      batch_size=batch_size, output_dir=out_dir,
                      remove_printer_callback=True, tf32=False, **extra)
