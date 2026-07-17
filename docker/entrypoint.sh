#!/usr/bin/env bash
# Rollenbasierter Container-Start. Die Worker laden beim Start bewusst kein Modell
# und reichen nicht sofort ein; ihr erster Lauf erfolgt erst zum Scheduler-Termin.
set -euo pipefail

role="${1:-${SERVICE_ROLE:-}}"

echo "=== Energy Arena Chronos-2 ==="
echo "Rolle: ${role:-nicht gesetzt} | Modell: ${MODEL_LABEL:-${PROD_CONFIG:-n/a}} | Submit aktiv: ${SUBMIT_ENABLED:-false}"

case "$role" in
  data-summary)
    mkdir -p "${APP_BASE:-/app/state}" "${LOCK_DIR:-/locks}"
    exec python -m pipeline.data_orchestrator
    ;;
  worker)
    mkdir -p "${APP_BASE:-/app/state}" "${ADAPTER_DIR:-/app/adapters}"
    mkdir -p "${LOCK_DIR:-/locks}"
    python - <<'PY'
import torch
ok = torch.cuda.is_available()
name = torch.cuda.get_device_name(0) if ok else "CPU (keine GPU sichtbar)"
print(f"torch {torch.__version__} | GPU verfuegbar: {ok} | {name}", flush=True)
PY
    exec python -m pipeline.worker_orchestrator
    ;;
  *)
    echo "Unbekannte SERVICE_ROLE/command: ${role:-<leer>} (erwartet: data-summary|worker)" >&2
    exit 2
    ;;
esac
