#!/usr/bin/env bash
# Container-Start: GPU pruefen, dann die vollautomatische Pipeline starten.
set -e
echo "=== Energy Day-ahead Pipeline (Chronos-2) ==="
python - <<'PY'
import torch
ok = torch.cuda.is_available()
name = torch.cuda.get_device_name(0) if ok else "CPU (keine GPU sichtbar!)"
print(f"torch {torch.__version__} | GPU verfuegbar: {ok} | {name}")
PY
echo "Modell: ${PROD_CONFIG:-G2_C2.2} | Submit aktiv: ${SUBMIT_ENABLED:-false}"
echo "---"
exec python -m pipeline.orchestrator
