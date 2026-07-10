"""Schlanke Daten-Lade-Pipeline fuer die Day-Ahead-Lastprognose (DE-LU).

Drei Datenquellen (siehe README):
  1. Wetter je H3-Zelle  (Punkt + Ensemble)      -> weather.py, aggregate.py
  2. Markt je ÜNB-Zone   (price/solar/wind/...)   -> loads.py, market.py
  3. Gesamt DE-LU        (aggregiert / eine Zelle) -> loads.py, market.py, aggregate.py

Alle Konstanten/Pfade liegen zentral in config.py.
"""
