# GPU-Container fuer die Day-ahead-Pipeline — AMD ROCm ODER NVIDIA CUDA.
# torch kommt aus dem Base-Image und wird NICHT ueberschrieben; nur Chronos-2
# (chronos-forecasting) + Pipeline-Deps werden nachinstalliert.
#
#   NVIDIA (Default):  docker build -f docker/Dockerfile -t energy-pipeline .
#   AMD ROCm:          docker build -f docker/Dockerfile \
#                        --build-arg BASE=rocm/pytorch:latest -t energy-pipeline .
ARG BASE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE}

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app APP_BASE=/app

WORKDIR /app

# Chronos-2 + Datenquellen-/Scheduler-/Finetuning-Abhaengigkeiten.
RUN pip install \
      chronos-forecasting==2.2.2 \
      pandas numpy h3 pyarrow requests \
      entsoe-py openmeteo-requests requests-cache retry-requests \
      holidays apscheduler peft

COPY . /app

ENTRYPOINT ["/app/docker/entrypoint.sh"]
