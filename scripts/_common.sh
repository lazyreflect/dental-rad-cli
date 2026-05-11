#!/usr/bin/env bash
# Shared bootstrap for all training scripts. Sourced, not executed.
# POSIX-compatible; works on Linux + Windows-WSL.

set -euo pipefail

# Resolve repo root from this script's location.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT

# Default venv location; override via VENV_DIR env if needed.
: "${VENV_DIR:=${REPO_ROOT}/.venv}"

# Default dataset / weights / logs roots; override via env to suit the host.
: "${DATA_ROOT:=${REPO_ROOT}/data/denpar/prepared}"
: "${WEIGHTS_DIR:=${REPO_ROOT}/weights}"
: "${LOGS_DIR:=${REPO_ROOT}/logs}"

# CUDA on the RTX 4090 box. Override CUDA_VISIBLE_DEVICES if multi-GPU.
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

# Surface non-fatal Ultralytics warnings without dumping their banner.
export YOLO_VERBOSE="${YOLO_VERBOSE:-False}"

mkdir -p "${WEIGHTS_DIR}" "${LOGS_DIR}"

activate_venv() {
  if [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1091
    . "${VENV_DIR}/bin/activate"
  elif [ -f "${VENV_DIR}/Scripts/activate" ]; then
    # Windows-WSL layout with a Windows-style venv.
    # shellcheck disable=SC1091
    . "${VENV_DIR}/Scripts/activate"
  else
    echo "warn: no venv found at ${VENV_DIR}; relying on system Python." >&2
  fi
  export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
}
