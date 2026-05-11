#!/usr/bin/env bash
# Train Keypoint R-CNN for CEJ landmark (2 keypoints/tooth).

. "$(dirname "$0")/_common.sh"
activate_venv

: "${DATASET_DIR:=${DATA_ROOT}/keypoints}"
: "${WEIGHTS_OUT:=${WEIGHTS_DIR}/keypoint_cej.pt}"
: "${EPOCHS:=200}"

python -m dental_rad_cli.training.keypoints \
  --landmark cej \
  --dataset-dir "${DATASET_DIR}" \
  --out "${WEIGHTS_OUT}" \
  --epochs "${EPOCHS}"
