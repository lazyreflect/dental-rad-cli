#!/usr/bin/env bash
# Run held-out evaluations for both perception heads.
#
# Headline metrics:
#   - cej_collapse_rate   (CEJ keypoint head, DenPAR Testing split, 200 PAs)
#   - caries_map50        (caries head, Baasils ICCMS test split, 20 BWs)
#
# These are the brutal one-number metrics for any autoresearch loop.
# Lower is better for the first; higher for the second.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"

if [ -f "${VENV_DIR}/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "${VENV_DIR}/bin/activate"
elif [ -f "${VENV_DIR}/Scripts/activate" ]; then
  # shellcheck disable=SC1091
  . "${VENV_DIR}/Scripts/activate"
fi
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "=== CEJ keypoint head — DenPAR Testing split ==="
if [ -d "${REPO_ROOT}/data/denpar/Dataset/Testing/Images" ] && [ -f "${REPO_ROOT}/weights/keypoint_cej.pt" ]; then
  python "${REPO_ROOT}/scripts/eval_keypoint_cej.py" || echo "(CEJ eval failed)"
else
  echo "(skipped — data/denpar/Dataset/Testing/Images or weights/keypoint_cej.pt missing)"
fi

echo
echo "=== Caries head — Baasils test split ==="
if [ -f "${REPO_ROOT}/data/prepared/yolo_caries/data.yaml" ] && [ -f "${REPO_ROOT}/weights/caries.pt" ]; then
  python "${REPO_ROOT}/scripts/eval_caries.py" || echo "(caries eval failed)"
else
  echo "(skipped — data/prepared/yolo_caries/data.yaml or weights/caries.pt missing)"
fi
