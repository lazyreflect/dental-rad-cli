#!/usr/bin/env bash
# Materialize the YOLO + COCO-keypoint subsets from raw DenPAR v3.
# Idempotent: re-running overwrites label/annotation files but skips
# image copies that already exist.
#
# Outputs (under data/prepared/):
#   yolo_tooth_detect/ dataset.yaml + images/{train,val,test}/*.jpg + labels/.../*.txt
#   yolo_tooth_seg/    same shape, 1 class = tooth (instance polygons)
#   yolo_bone_seg/     same shape, 1 class = bone (single polygon per image)
#   coco_keypoints/    {train,val,test}/{images/*.jpg, annotations.json}  COCO-keypoints
#
# Run download_denpar.sh first if Dataset/ is not present.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "${REPO_ROOT}/scripts/_common.sh"
activate_venv

DENPAR_ROOT="${REPO_ROOT}/data/denpar"
PREPARED_ROOT="${REPO_ROOT}/data/prepared"

if [ ! -d "${DENPAR_ROOT}/Dataset/Training/Images" ]; then
  echo "prepare_datasets: Dataset/ not found — run scripts/download_denpar.sh first." >&2
  exit 1
fi

mkdir -p "${PREPARED_ROOT}"

python - <<PY
import logging
from pathlib import Path
from dental_rad_cli.data.denpar_adapter import (
    build_yolo_dataset,
    build_coco_keypoints,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("prepare_datasets")

denpar_root = Path("${DENPAR_ROOT}")
prepared = Path("${PREPARED_ROOT}")

# YOLO detection (3 classes per paper-exact arch: bg, single, double)
yaml = build_yolo_dataset(denpar_root, prepared / "yolo_tooth_detect", "tooth_detect")
log.info("yolo_tooth_detect dataset.yaml: %s", yaml)

# YOLO tooth segmentation (1 class)
yaml = build_yolo_dataset(denpar_root, prepared / "yolo_tooth_seg", "tooth_seg")
log.info("yolo_tooth_seg dataset.yaml: %s", yaml)

# YOLO bone segmentation (1 class)
yaml = build_yolo_dataset(denpar_root, prepared / "yolo_bone_seg", "bone_seg")
log.info("yolo_bone_seg dataset.yaml: %s", yaml)

# COCO-keypoints (single layout; trainer slices per landmark)
ann = build_coco_keypoints(denpar_root, prepared / "coco_keypoints", "cej")
log.info("coco_keypoints train annotations: %s", ann)
PY

echo "prepare_datasets: outputs written under ${PREPARED_ROOT}"
