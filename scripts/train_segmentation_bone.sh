#!/usr/bin/env bash
# Train YOLOv8x-seg single-class bone segmentation.

. "$(dirname "$0")/_common.sh"
activate_venv

: "${DATA_YAML:=${DATA_ROOT}/yolo_seg_bone/dataset.yaml}"
: "${WEIGHTS_OUT:=${WEIGHTS_DIR}/segmentation_bone.pt}"
: "${EPOCHS:=200}"

python -m dental_rad_cli.training.segmentation \
  --target bone \
  --data "${DATA_YAML}" \
  --out "${WEIGHTS_OUT}" \
  --epochs "${EPOCHS}"
