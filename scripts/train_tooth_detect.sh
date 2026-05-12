#!/usr/bin/env bash
# Train YOLOv9e 3-class tooth detector.
# Override DATA_YAML / WEIGHTS_OUT / EPOCHS via env.

. "$(dirname "$0")/_common.sh"
activate_venv

: "${DATA_YAML:=${DATA_ROOT}/yolo_tooth_detect/dataset.yaml}"
: "${WEIGHTS_OUT:=${WEIGHTS_DIR}/tooth_detect.pt}"
: "${EPOCHS:=200}"

python -m dental_rad_cli.training.tooth_detect \
  --data "${DATA_YAML}" \
  --out  "${WEIGHTS_OUT}" \
  --epochs "${EPOCHS}"
