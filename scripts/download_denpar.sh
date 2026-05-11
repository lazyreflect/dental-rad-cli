#!/usr/bin/env bash
# Download DenPAR v3 (Zenodo record 16645076) into data/denpar/ if not
# already present. Idempotent: skips download + unzip when the unpacked
# Dataset/ tree already exists.
#
# Works on macOS + Linux. Uses curl (preferred — present on both),
# falls back to wget if curl is unavailable. Uses unzip from PATH.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DENPAR_DIR="${REPO_ROOT}/data/denpar"
DATASET_DIR="${DENPAR_DIR}/Dataset"
ZIP_PATH="${DENPAR_DIR}/DenPAR.zip"
ZENODO_URL="https://zenodo.org/api/records/16645076/files/DenPAR%20Radiographs%20Dataset.zip/content"

mkdir -p "${DENPAR_DIR}"

if [ -d "${DATASET_DIR}" ] && [ -d "${DATASET_DIR}/Training/Images" ]; then
  echo "denpar: Dataset/ already present at ${DATASET_DIR}; skipping download."
  exit 0
fi

if [ ! -f "${ZIP_PATH}" ]; then
  echo "denpar: downloading from Zenodo (record 16645076) → ${ZIP_PATH}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --retry-delay 5 -o "${ZIP_PATH}" "${ZENODO_URL}"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=3 --waitretry=5 -O "${ZIP_PATH}" "${ZENODO_URL}"
  else
    echo "denpar: neither curl nor wget available; install one and retry." >&2
    exit 1
  fi
else
  echo "denpar: zip already present at ${ZIP_PATH}; reusing."
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "denpar: 'unzip' not available; install it and retry." >&2
  exit 1
fi

echo "denpar: unzipping → ${DENPAR_DIR}"
unzip -q -o "${ZIP_PATH}" -d "${DENPAR_DIR}"

if [ ! -d "${DATASET_DIR}/Training/Images" ]; then
  echo "denpar: unzip completed but Dataset/Training/Images not found — verify Zenodo zip layout." >&2
  exit 1
fi

echo "denpar: ready at ${DATASET_DIR}"
