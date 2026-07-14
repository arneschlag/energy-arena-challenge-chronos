"""CLI der Experiment-Suite.

  # Walk-Forward einer/mehrerer Configs (Praefix-Filter: G1, G2, G1_C2, ...):
  python -m experiments.run run --config G1_C1 --start 2024-01-01 --end 2024-06-30
  python -m experiments.run run --config G1        --start 2024-01-01 --end 2026-06-01
  python -m experiments.run run --config all --cadence 1

  # Metriken aus den gespeicherten Samples aggregieren (+ ENTSO-E-Benchmark):
  python -m experiments.run score

Laeuft im chronos-Container: docker exec chronos python -m experiments.run run ...
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

from . import configs, score, walkforward

OUT = Path(__file__).resolve().parent / "out"


def _select(prefix: str):
    if prefix == "all":
        return configs.all_configs()
    sel = [c for c in configs.all_configs() if c.name == prefix or c.name.startswith(prefix + "_")
           or c.name.startswith(prefix)]
    if not sel:
        sys.exit(f"keine Config passt zu {prefix!r}. Verfuegbar: {configs.names()}")
    return sel


def main():
    p = argparse.ArgumentParser(description="Chronos-2 Walk-Forward Experimente")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Walk-Forward ausfuehren")
    r.add_argument("--config", required=True, help="Name/Praefix (z.B. G1_C1, G1, all)")
    r.add_argument("--start", default="2024-01-01")
    r.add_argument("--end", default="2026-06-01")
    r.add_argument("--cadence", type=int, default=1, help="Tage zwischen Origins")
    r.add_argument("--ctx-days", type=int, default=63)
    r.add_argument("--gate-hour", type=int, default=9)
    r.add_argument("--out-dir", default=None,
                   help="Zielverzeichnis (rel. zu experiments/ oder absolut; Default out)")
    r.add_argument("--force", action="store_true", help="vorhandene Ergebnisse ueberschreiben")
    r.add_argument("--score", action="store_true", help="am Ende automatisch scoren")

    f = sub.add_parser("finetune", help="Rolling-Window LoRA-Finetuning + Eval (-> out_ft/)")
    f.add_argument("--config", required=True)
    f.add_argument("--start", default="2024-01-01")
    f.add_argument("--end", default="2026-06-01")
    f.add_argument("--cadence", type=int, default=1)
    f.add_argument("--ctx-days", type=int, default=63)
    f.add_argument("--gate-hour", type=int, default=9)
    f.add_argument("--num-steps", type=int, default=200, help="LoRA-Steps je Refit (klein = wenig Overfit)")
    f.add_argument("--refit-months", type=int, default=3, help="Rolling-Refit-Cadence in Monaten")
    f.add_argument("--train-window-months", type=int, default=None,
                   help="Rolling-Trainingsfenster in Monaten (Default: Expanding = alle Historie)")
    f.add_argument("--out-dir", default=None,
                   help="Zielverzeichnis (rel. zu experiments/ oder absolut; Default out_ft)")
    f.add_argument("--area", default=None,
                   help="nur dieses Gebiet rechnen (indep/whole) -> Teil-Parquet "
                        "{config}__{area}.parquet; isoliert Fits gegen HIP-Fehler")
    f.add_argument("--force", action="store_true")

    s = sub.add_parser("score", help="Metriken aggregieren")
    s.add_argument("--out", default=None)
    s.add_argument("--dir", default=None, help="Verzeichnis der Parquets (Default out/; z.B. out_ft)")

    a = p.parse_args()
    if a.cmd == "finetune":
        from . import walkforward as wf
        _base = Path(__file__).resolve().parent
        ft_dir = (Path(a.out_dir) if a.out_dir and Path(a.out_dir).is_absolute()
                  else _base / (a.out_dir or "out_ft"))
        sel = _select(a.config)
        print(f"### FT START {len(sel)} Configs ({a.start}..{a.end}, refit {a.refit_months}M, "
              f"{a.num_steps} steps) {time.strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr, flush=True)
        for i, cfg in enumerate(sel, 1):
            if (ft_dir / f"{cfg.name}.parquet").exists() and not a.force:
                print(f"[{i}/{len(sel)}] skip {cfg.name} (vorhanden)", file=sys.stderr, flush=True)
                continue
            t0 = time.time()
            print(f"[{i}/{len(sel)}] == FT {cfg.name} == {time.strftime('%H:%M:%S')}",
                  file=sys.stderr, flush=True)
            try:
                wf.run_config_ft(cfg, a.start, a.end, cadence=a.cadence, ctx_days=a.ctx_days,
                                 gate_hour=a.gate_hour, num_steps=a.num_steps,
                                 refit_months=a.refit_months, out_dir=ft_dir, only_area=a.area,
                                 train_window_months=a.train_window_months)
            except Exception:
                print(f"[FEHLER] {cfg.name}:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            print(f"[{i}/{len(sel)}] {cfg.name} fertig in {time.time()-t0:.0f}s",
                  file=sys.stderr, flush=True)
        print(f"### FT DONE {time.strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr, flush=True)
    elif a.cmd == "run":
        zs_dir = (Path(a.out_dir) if a.out_dir and Path(a.out_dir).is_absolute()
                  else Path(__file__).resolve().parent / (a.out_dir or "out"))
        sel = _select(a.config)
        print(f"### START {len(sel)} Configs ({a.start}..{a.end}, cadence {a.cadence}) "
              f"{time.strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr, flush=True)
        for i, cfg in enumerate(sel, 1):
            pq = zs_dir / f"{cfg.name}.parquet"
            if pq.exists() and not a.force:
                print(f"[{i}/{len(sel)}] skip {cfg.name} (vorhanden)", file=sys.stderr, flush=True)
                continue
            t0 = time.time()
            print(f"[{i}/{len(sel)}] == {cfg.name} == {time.strftime('%H:%M:%S')}",
                  file=sys.stderr, flush=True)
            try:                                             # Fehler isolieren -> Batch laeuft weiter
                walkforward.run_config(cfg, a.start, a.end, cadence=a.cadence,
                                       ctx_days=a.ctx_days, gate_hour=a.gate_hour,
                                       out_dir=zs_dir)
            except Exception:
                print(f"[FEHLER] {cfg.name}:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            print(f"[{i}/{len(sel)}] {cfg.name} fertig in {time.time()-t0:.0f}s",
                  file=sys.stderr, flush=True)
        print(f"### DONE {time.strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr, flush=True)
        if a.score:
            score.score_all(out_dir=zs_dir)
    else:
        d = Path(__file__).resolve().parent / a.dir if a.dir else score.OUT
        score.score_all(a.out, out_dir=d)


if __name__ == "__main__":
    main()
