"""Registry der Experiment-Konfigurationen (Granularitaet x C1..C6).

Granularitaeten:
  whole  (G1)  ein Ziel = de_lu-Last
  indep  (G2)  5 unabhaengige Modelle (50hertz,tennet,amprion,transnetbw,lu) -> DE-LU=Summe
  joint  (G3)  ein multivariates Modell ueber die 5 Zonen -> DE-LU=Quantil-Summe

Input/Output-Ablation (je Granularitaet):
  C1    univariat: nur Last
  C2.1  + Wetter (eine feste Zelle / centroid)      known-future Covariate
  C2.2  + Wetter (bevoelkerungsgewichtetes Mittel)  known-future Covariate
  C2.3  + Wetter (4 gleichmaessig verteilte Zellen) known-future Covariate (28 Spalten)
  C3.1  Multitask: Ziel = [Last, Preis]             (Aux als Ziel-Variate, kein Wetter)
  C3.2  Multitask: Ziel = [Last, Solar]
  C3.3  Multitask: Ziel = [Last, Wind]
  C3.4  Multitask: Ziel = [Last, Preis, Solar, Wind]
  C4    alles: Last-Output, Input = Wetter(pop) + Preis/Solar/Wind(past-only)
  C5    C4 + Kalender-Covariaten
  C6    univariat + Kalender-Covariaten
"""
from __future__ import annotations

from dataclasses import dataclass, field

from loaders import config

GRAN_TAG = {"whole": "G1", "indep": "G2", "joint": "G3"}


@dataclass(frozen=True)
class Config:
    name: str                       # z.B. "G1_C2.2"
    gran: str                       # whole | indep | joint
    weather: str | None = None      # None | centroid | pop | even4
    aux_target: tuple = ()          # Aux als zusaetzliche Ziel-Variaten (C3)
    aux_known: tuple = ()           # Aux als known-future Covariate (C4: price)
    aux_past: tuple = ()            # Aux als past-only Covariate  (C4: solar,wind)
    calendar: tuple = ()            # Kalender-Covariaten (known-future), z.B. ("holiday",)

    @property
    def areas(self) -> list[str]:
        return [config.DELU] if self.gran == "whole" else config.ZONES + [config.LU]

    @property
    def multivariate(self) -> bool:
        return self.gran == "joint"


# (tag, weather, aux_target, aux_known, aux_past, calendar)
# v2 (2026-07-11): Preis ist PAST-only (Day-ahead-Auktion schliesst ~12:00, Gate ist
# 09:00 -> der Preis des Liefertags ist am Gate noch unbekannt; Historie ist publiziert).
_ABLATION = [
    ("C1",   None,       (),                        (),          (),                 ()),
    ("C2.1", "centroid", (),                        (),          (),                 ()),
    ("C2.2", "pop",      (),                        (),          (),                 ()),
    ("C2.3", "even4",    (),                        (),          (),                 ()),
    ("C3.1", None,       ("price",),                (),          (),                 ()),
    ("C3.2", None,       ("solar",),                (),          (),                 ()),
    ("C3.3", None,       ("wind",),                 (),          (),                 ()),
    ("C3.4", None,       ("price", "solar", "wind"), (),         (),                 ()),
    ("C4",   "pop",      (),                        (),  ("price", "solar", "wind"), ()),
    # C5 = C4 + Kalender (alle known-future, da echter Kalender):
    # Feiertag, Brueckentag, Tag-vor/nach-Feiertag, Weihnachtsperiode (24.12.-01.01.),
    # Wochentag sin/cos, Tag-im-Jahr sin/cos, Wochenende.
    ("C5",   "pop",      (),                        (),  ("price", "solar", "wind"),
     ("holiday", "bridge", "pre_holiday", "post_holiday", "xmas",
      "dow_sin", "dow_cos", "doy_sin", "doy_cos", "weekend")),
    # C6 = univariat + NUR Kalender (kein Wetter, kein Aux): isoliert den Beitrag der
    # Kalender-Features. 10 known-Covariaten -> Bucket-Risiko (10 war bei C4-Shapes
    # kaputt); FT ggf. mit FT_PAD_COVARIATES ausweichen (s. model.area_frame).
    ("C6",   None,       (),                        (),  (),
     ("holiday", "bridge", "pre_holiday", "post_holiday", "xmas",
      "dow_sin", "dow_cos", "doy_sin", "doy_cos", "weekend")),
]


def all_configs() -> list[Config]:
    out = []
    for gran, gtag in GRAN_TAG.items():
        for tag, w, at, ak, ap, cal in _ABLATION:
            out.append(Config(f"{gtag}_{tag}", gran, w, at, ak, ap, cal))
    return out


def get(name: str) -> Config:
    for c in all_configs():
        if c.name == name:
            return c
    raise KeyError(f"unbekannte Config {name!r}")


def names() -> list[str]:
    return [c.name for c in all_configs()]
