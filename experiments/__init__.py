"""Zero-Shot Chronos-2 Walk-Forward-Experimente (Day-ahead-Lastprognose DE-LU).

Module:
  configs.py      Registry der Experiment-Konfigurationen (Granularitaet x C1..C4)
  data.py         Laden der sauberen loader-Ausgaben + leckage-sichere Fenster
  model.py        Chronos-2-Pipeline + Task-Builder + predict  (braucht chronos, GPU)
  metrics.py      RMSE, R2, WIS, LQS, MAE-Median, Cov50/95, CRPS, Energy Score
  walkforward.py  rollierende Tages-Schleife -> Samples je Tag
  score.py        Aggregation zu Metrik-Tabelle (+ ENTSO-E-Benchmark)
  run.py          CLI

Ausfuehrung im chronos-Docker-Container (torch-ROCm + chronos-2, GPU);
Code ist versioniert und reproduzierbar auch ausserhalb.
"""
