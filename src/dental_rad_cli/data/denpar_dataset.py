"""PyTorch ``Dataset`` classes for DenPAR v3-derived training data.

Three datasets are provided:

- ``DenParDetectionDataset`` — YOLO-style detection. Provided for
  symmetry / tests; in practice Ultralytics' YOLO trainer reads the
  ``dataset.yaml`` + ``.txt`` label files directly (no custom Dataset
  needed). Use this when you need a torch-native iterator over the
  detection split (e.g., for evaluating a non-Ultralytics detector).

- ``DenParKeypointDataset`` — Keypoint R-CNN training. Returns
  ``(image_tensor, target_dict)`` where ``target_dict`` matches the
  torchvision detection model spec: ``boxes``, ``labels``,
  ``keypoints``, ``image_id``. CLAHE is applied at load time —
  identical preprocessing at train and inference.

  This dataset is functionally equivalent to ``CocoKeypointSlice`` in
  ``training/keypoints.py``. Both consume the same COCO files. This
  module-level class exists so callers outside the training module
  (e.g., dataset diagnostics, sanity-check scripts) don't have to
  import from ``training.``.

- ``DenParSegmentationDataset`` — YOLO-seg style. Same situation as
  detection — Ultralytics reads the directory layout directly.
  Provided for symmetry and for non-Ultralytics segmentation
  experiments.

All three datasets are deterministic — file enumeration uses
``sorted()`` and no train/val split shuffling. DenPAR v3 already
defines splits at the directory level; we never re-split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Literal

import cv2
import numpy as np
import torch
import torch.utils.data
from PIL import Image

from ..training.preprocess import apply_clahe

Split = Literal["train", "val", "test"]

# COCO landmark slices — must match
# ``training/keypoints.py::_LANDMARK_SLICES``. The slices index into a
# per-annotation flat keypoint list of length ``6*3=18`` and select
# 1-2 points per tooth depending on landmark.
_LANDMARK_SLICES: Final[dict[str, tuple[int, int]]] = {
    "cej": (0, 2),
    "bone": (2, 4),
    "apex": (4, 5),
}
_LANDMARK_NUM_KEYPOINTS: Final[dict[str, int]] = {
    "cej": 2,
    "bone": 2,
    "apex": 1,
}

Landmark = Literal["cej", "bone", "apex"]


# ---------------------------------------------------------------------------
# Detection (YOLO format)
# ---------------------------------------------------------------------------


class DenParDetectionDataset(torch.utils.data.Dataset):
    """Iterate over a DenPAR-derived YOLO detection split.

    Reads from the YOLO directory layout produced by
    ``denpar_adapter.build_yolo_dataset(..., target="tooth_detect")``:
    ``<root>/images/<split>/*.jpg`` + ``<root>/labels/<split>/*.txt``.

    Each ``.txt`` label has rows ``class cx cy w h`` with all values
    normalized to ``[0, 1]``.

    ``__getitem__`` returns ``(image_tensor, target_dict)`` where:

    - ``image_tensor``: CHW float32 in [0,1] (no CLAHE — YOLO models
      consume raw RGB per the methodology brief §1.1).
    - ``target_dict``:
        - ``boxes``:  FloatTensor[N, 4] in absolute pixel xyxy
        - ``labels``: Int64Tensor[N] using 1=single, 2=double (we
          shift the YOLO 0/1 convention back to 1/2 to keep label
          semantics aligned with the keypoint dataset)
        - ``image_id``: Int64Tensor[1]
    """

    def __init__(self, root: Path, split: Split) -> None:
        self.root = Path(root)
        self.split = split
        self.images_dir = self.root / "images" / split
        self.labels_dir = self.root / "labels" / split
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"missing images dir: {self.images_dir}")
        if not self.labels_dir.is_dir():
            raise FileNotFoundError(f"missing labels dir: {self.labels_dir}")
        self._stems: list[str] = sorted(p.stem for p in self.images_dir.glob("*.jpg"))

    def __len__(self) -> int:
        return len(self._stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self._stems[idx]
        img_path = self.images_dir / f"{stem}.jpg"

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"failed to read image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = rgb.shape[:2]
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        boxes: list[list[float]] = []
        labels: list[int] = []

        label_path = self.labels_dir / f"{stem}.txt"
        if label_path.is_file():
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0])
                cx, cy, w, h = (float(x) for x in parts[1:])
                # de-normalize to absolute xyxy
                x1 = (cx - 0.5 * w) * img_w
                y1 = (cy - 0.5 * h) * img_h
                x2 = (cx + 0.5 * w) * img_w
                y2 = (cy + 0.5 * h) * img_h
                boxes.append([x1, y1, x2, y2])
                labels.append(cls + 1)  # YOLO 0/1 -> torchvision 1/2

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }
        return img_tensor, target


# ---------------------------------------------------------------------------
# Keypoint (COCO-keypoints format)
# ---------------------------------------------------------------------------


class DenParKeypointDataset(torch.utils.data.Dataset):
    """Per-image dataset for Keypoint R-CNN training.

    Reads from the COCO-keypoint layout produced by
    ``denpar_adapter.build_coco_keypoints``:
    ``<root>/<split>/images/*.jpg`` + ``<root>/<split>/annotations.json``.

    The COCO file declares 6 keypoints per tooth; this class slices to
    one ``landmark`` group at construction (cej / bone / apex) and
    returns only that group's keypoints in the target dict.

    CLAHE preprocessing is applied at load time — the same
    ``apply_clahe`` function used at inference, so training-and-inference
    image statistics match.

    Returns ``(image_tensor, target)`` where ``target`` has:
        boxes:     FloatTensor[N, 4]   absolute xyxy pixels
        labels:    Int64Tensor[N]      1=single, 2=double
        keypoints: FloatTensor[N, K, 3]  K = 1 or 2 per ``landmark``
        image_id:  Int64Tensor[1]

    Images with no annotations are dropped (Keypoint R-CNN cannot
    learn from empty samples).
    """

    def __init__(
        self,
        root: Path,
        split: Split,
        landmark: Landmark,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        self.images_dir = self.split_dir / "images"
        self.ann_path = self.split_dir / "annotations.json"
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"missing images dir: {self.images_dir}")
        if not self.ann_path.is_file():
            raise FileNotFoundError(f"missing annotations: {self.ann_path}")

        with self.ann_path.open() as fh:
            coco = json.load(fh)

        self._images: list[dict] = coco["images"]
        self._img_by_id: dict[int, dict] = {im["id"]: im for im in self._images}

        self._anns_by_img: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            self._anns_by_img.setdefault(ann["image_id"], []).append(ann)

        # Drop empty-annotation images (deterministic — sorted by id).
        self._image_ids: list[int] = sorted(
            im["id"] for im in self._images if self._anns_by_img.get(im["id"])
        )

        slc = _LANDMARK_SLICES[landmark]
        self._slice_start, self._slice_end = slc
        self._num_keypoints = _LANDMARK_NUM_KEYPOINTS[landmark]
        self._landmark: Landmark = landmark

    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        img_id = self._image_ids[idx]
        img_info = self._img_by_id[img_id]
        img_path = self.images_dir / img_info["file_name"]

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"failed to read image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = apply_clahe(rgb)

        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        boxes: list[list[float]] = []
        labels: list[int] = []
        kps_per_tooth: list[list[list[float]]] = []

        for ann in self._anns_by_img[img_id]:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(int(ann["category_id"]))
            flat = ann["keypoints"]
            # flat is length 18 = 6 * 3. Slice K = end - start keypoints.
            k_start_flat = self._slice_start * 3
            k_end_flat = self._slice_end * 3
            sliced = flat[k_start_flat:k_end_flat]
            triples: list[list[float]] = [
                [float(sliced[i]), float(sliced[i + 1]), float(sliced[i + 2])]
                for i in range(0, len(sliced), 3)
            ]
            kps_per_tooth.append(triples)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "keypoints": torch.tensor(kps_per_tooth, dtype=torch.float32).reshape(
                -1, self._num_keypoints, 3
            ),
            "image_id": torch.tensor([img_id], dtype=torch.int64),
        }
        return img_tensor, target


# ---------------------------------------------------------------------------
# Segmentation (YOLO-seg format)
# ---------------------------------------------------------------------------


class DenParSegmentationDataset(torch.utils.data.Dataset):
    """Iterate over a DenPAR-derived YOLO-seg split.

    Reads ``<root>/images/<split>/*.jpg`` + ``<root>/labels/<split>/*.txt``
    where each label row is ``class x1 y1 x2 y2 ... xN yN`` (normalized
    polygon).

    Provided for symmetry / non-Ultralytics workflows; the production
    training path passes the dataset.yaml to ``YOLO.train()`` and never
    instantiates this class.

    Returns ``(image_tensor, target)`` where ``target`` has:
        labels:   Int64Tensor[N]
        polygons: list[Tensor[K_i, 2]]  absolute pixel coordinates per instance
        image_id: Int64Tensor[1]
    """

    def __init__(self, root: Path, split: Split) -> None:
        self.root = Path(root)
        self.split = split
        self.images_dir = self.root / "images" / split
        self.labels_dir = self.root / "labels" / split
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"missing images dir: {self.images_dir}")
        if not self.labels_dir.is_dir():
            raise FileNotFoundError(f"missing labels dir: {self.labels_dir}")
        self._stems: list[str] = sorted(p.stem for p in self.images_dir.glob("*.jpg"))

    def __len__(self) -> int:
        return len(self._stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self._stems[idx]
        img_path = self.images_dir / f"{stem}.jpg"

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"failed to read image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = rgb.shape[:2]
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        labels: list[int] = []
        polygons: list[torch.Tensor] = []

        label_path = self.labels_dir / f"{stem}.txt"
        if label_path.is_file():
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 7:  # class + at least 3 (x,y) pairs
                    continue
                cls = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                if len(coords) % 2 != 0:
                    continue
                arr = np.array(coords, dtype=np.float32).reshape(-1, 2)
                arr[:, 0] *= img_w
                arr[:, 1] *= img_h
                labels.append(cls)
                polygons.append(torch.from_numpy(arr))

        target = {
            "labels": torch.tensor(labels, dtype=torch.int64),
            "polygons": polygons,
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }
        return img_tensor, target
