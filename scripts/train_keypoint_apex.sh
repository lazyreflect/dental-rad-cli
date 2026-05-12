#!/usr/bin/env bash
# Train Keypoint R-CNN for apex landmark (1 keypoint/tooth).

. "$(dirname "$0")/_common.sh"
activate_venv

: "${DATASET_DIR:=${DATA_ROOT}/coco_keypoints}"
: "${WEIGHTS_OUT:=${WEIGHTS_DIR}/keypoint_apex.pt}"
: "${EPOCHS:=200}"

python -m dental_rad_cli.training.keypoints \
  --landmark apex \
  --dataset-dir "${DATASET_DIR}" \
  --out "${WEIGHTS_OUT}" \
  --epochs "${EPOCHS}"
