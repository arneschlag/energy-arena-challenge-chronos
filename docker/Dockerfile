# GPU-Container fuer die Day-ahead-Pipeline — AMD ROCm ODER NVIDIA CUDA.
# torch kommt aus dem Base-Image und wird NICHT ueberschrieben; nur Chronos-2
# (chronos-forecasting) + Pipeline-Deps werden nachinstalliert.
#
#   NVIDIA (Default):  docker build -f docker/Dockerfile -t energy-pipeline .
#   AMD ROCm:          docker build -f docker/Dockerfile \
#                        --build-arg \
#                        BASE=rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.10.0 \
#                        -t energy-pipeline .
ARG BASE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE}

ARG REPO_COMMIT=unknown

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app APP_BASE=/app REPO_COMMIT=${REPO_COMMIT}

WORKDIR /app

# Chronos-2 + Datenquellen-/Scheduler-/Finetuning-Abhaengigkeiten.
RUN pip install \
      chronos-forecasting==2.2.2 \
      pandas numpy h3 pyarrow requests \
      entsoe-py openmeteo-requests requests-cache retry-requests \
      holidays apscheduler peft

COPY . /app

ENTRYPOINT ["/app/docker/entrypoint.sh"]
