#!/usr/bin/env bash
# Download the Renielaz Dental Caries X-ray dataset from Roboflow and
# convert it to our internal 3-class YOLOv8 layout.
#
# Idempotent: if data/caries/data.yaml or data/prepared/yolo_caries/data.yaml
# already exists, the corresponding step is skipped.
#
# Requires ROBOFLOW_API_KEY in the environment (free tier API key works).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${REPO_ROOT}/data/caries"
PREPARED_DIR="${REPO_ROOT}/data/prepared/yolo_caries"

if [ -z "${ROBOFLOW_API_KEY:-}" ]; then
  echo "caries: ROBOFLOW_API_KEY not set in environment." >&2
  echo "        Get a free key at https://app.roboflow.com/settings/api" >&2
  exit 1
fi

mkdir -p "${RAW_DIR}" "${PREPARED_DIR}"

# Activate venv if available so the `roboflow` package is on PYTHONPATH.
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
if [ -f "${VENV_DIR}/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "${VENV_DIR}/bin/activate"
elif [ -f "${VENV_DIR}/Scripts/activate" ]; then
  # shellcheck disable=SC1091
  . "${VENV_DIR}/Scripts/activate"
fi
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "caries: step 1/2 — download Renielaz from Roboflow → ${RAW_DIR}"
python - <<PY
import os
from pathlib import Path
from dental_rad_cli.data.caries_adapter import download_renielaz

raw = Path(r"${RAW_DIR}")
out = download_renielaz(raw)
print(f"caries: roboflow export at {out}")
PY

echo "caries: step 2/2 — re-map ICCMS classes → 3-class internal layout → ${PREPARED_DIR}"
python - <<PY
from pathlib import Path
from dental_rad_cli.data.caries_adapter import (
    build_yolo_caries_dataset,
    download_renielaz,
)

raw = Path(r"${RAW_DIR}")
# download_renielaz is idempotent; calling it again returns the existing path.
roboflow_root = download_renielaz(raw)
yaml = build_yolo_caries_dataset(roboflow_root, Path(r"${PREPARED_DIR}"))
print(f"caries: data.yaml at {yaml}")
PY

echo "caries: ready at ${PREPARED_DIR}"
