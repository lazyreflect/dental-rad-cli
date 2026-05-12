#!/usr/bin/env bash
# Train YOLOv8x-seg single-class tooth segmentation.

. "$(dirname "$0")/_common.sh"
activate_venv

: "${DATA_YAML:=${DATA_ROOT}/yolo_tooth_seg/dataset.yaml}"
: "${WEIGHTS_OUT:=${WEIGHTS_DIR}/segmentation_tooth.pt}"
: "${EPOCHS:=200}"

python -m dental_rad_cli.training.segmentation \
  --target tooth \
  --data "${DATA_YAML}" \
  --out "${WEIGHTS_OUT}" \
  --epochs "${EPOCHS}"
