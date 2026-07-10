"""Produktions-Pipeline (Live-Betrieb): Daten -> Forecast -> Submit an Energy Arena.

Anders als experiments/ (Backtesting) erzeugt dies TÄGLICH die Day-ahead-Prognose
fuer den naechsten Liefertag und reicht sie bei der Energy Arena ein — vollautomatisch.

  forecast.py      Live-Forecast des naechsten Liefertags (DE-LU aus Zonen)
  arena.py         Energy-Arena-Submission (Challenges 20/22/24)
  orchestrator.py  Scheduler: stuendlich Daten frisch halten, Submit wenn Fenster offen

Start (im Container):  python -m pipeline.orchestrator
"""
