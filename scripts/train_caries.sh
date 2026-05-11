#!/usr/bin/env bash
# Train YOLOv8s 3-class caries detector against the Renielaz-derived
# data/prepared/yolo_caries/ split.
# Override DATA_YAML / WEIGHTS_OUT / EPOCHS via env.

. "$(dirname "$0")/_common.sh"
activate_venv

: "${DATA_YAML:=${REPO_ROOT}/data/prepared/yolo_caries/data.yaml}"
: "${WEIGHTS_OUT:=${WEIGHTS_DIR}/caries.pt}"
: "${EPOCHS:=200}"

python -m dental_rad_cli.training.caries \
  --data "${DATA_YAML}" \
  --out  "${WEIGHTS_OUT}" \
  --epochs "${EPOCHS}"
