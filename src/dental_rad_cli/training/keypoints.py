"""Keypoint R-CNN training — one model per landmark type.

Per methodology brief §1.2, the upstream pipeline trains the same
architecture (torchvision `keypointrcnn_resnet50_fpn`) three separate
times, once per landmark type. Each model emits its landmark(s) per
detected tooth; the inference path merges the three outputs into a
6-keypoint-per-tooth structure (CEJ pair, bone-crest pair, apex
1-or-2).

Landmark → num_keypoints mapping (per task spec from Joseph):
  cej   → 2  (mesial CEJ, distal CEJ)
  bone  → 2  (mesial bone-crest, distal bone-crest; aka AEAC)
  apex  → 1  (single apex point; brief notes upstream uses 2 but
              collapsed to 1 here because single-rooted teeth have one
              apex and double-rooted teeth get a separate per-root model
              evaluation downstream)

Dataset format (COCO-keypoints, per task constraints):
  <dataset_dir>/
    train/
      images/*.jpg
      annotations.json   (COCO-keypoint format, all 3 landmark groups)
    val/
      images/*.jpg
      annotations.json
    test/                (optional)

The COCO `keypoints` field per annotation is a flat list of length
3 * num_keypoints_total = 3 * 6 = 18 (cej_l, cej_r, bone_l, bone_r,
apex_a, apex_b — each with x,y,vis). We slice based on `landmark`:
  cej  → keypoints[0:2 ]    (indices 0..1)
  bone → keypoints[2:4 ]    (indices 2..3)
  apex → keypoints[4:5 ]    (indices 4..4 only — first apex point)

Visibility convention: 0 = absent, >0 = visible (used as numeric weight).

Hyperparameters (brief §1.2):
  Adam(lr=0.0001, weight_decay=1e-6)
  constant LR (no scheduler.step in upstream active loop)
  batch_size: train=8, val=4
  patience: 10 (brief says 10; task allows 10-30; we default 20)
  num_classes=3  (bg + single-rooted + double-rooted)

Save format: state_dict ONLY (not whole model — torchvision-version-fragile).
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Final, Literal

import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision
from torch import nn
from torchvision.models.detection import keypointrcnn_resnet50_fpn

from .preprocess import apply_clahe

_LOG = logging.getLogger(__name__)

# Hyperparameters from brief §1.2.
_LR: Final[float] = 1e-4
_WEIGHT_DECAY: Final[float] = 1e-6
_BATCH_TRAIN: Final[int] = 8
_BATCH_VAL: Final[int] = 4
_NUM_CLASSES: Final[int] = 3  # bg + single-rooted + double-rooted
_EARLY_STOP_PATIENCE: Final[int] = 20  # brief says 10-30; pick middle

# Slice indices into the per-tooth flat keypoint list (COCO order:
# cej_l, cej_r, bone_l, bone_r, apex_a, apex_b).
_LANDMARK_SLICES: Final[dict[str, tuple[int, int]]] = {
    "cej": (0, 2),
    "bone": (2, 4),
    "apex": (4, 5),  # single apex point per tooth
}
_LANDMARK_NUM_KEYPOINTS: Final[dict[str, int]] = {
    "cej": 2,
    "bone": 2,
    "apex": 1,
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CocoKeypointSlice(torch.utils.data.Dataset):
    """COCO-keypoint dataset that slices to a single landmark group.

    Returns torchvision-detection-style targets:
        boxes:     FloatTensor[N, 4]  (xyxy)
        labels:    Int64Tensor[N]     (1=single-rooted, 2=double-rooted)
        keypoints: FloatTensor[N, K, 3]   K = 1 or 2 depending on landmark
        image_id:  Int64Tensor[1]
    """

    def __init__(self, split_dir: Path, landmark: str) -> None:
        self.split_dir = Path(split_dir)
        self.images_dir = self.split_dir / "images"
        self.ann_path = self.split_dir / "annotations.json"
        if not self.ann_path.exists():
            raise FileNotFoundError(f"missing annotations.json at {self.ann_path}")

        with self.ann_path.open() as fh:
            coco = json.load(fh)

        self._images: list[dict] = coco["images"]
        self._img_by_id: dict[int, dict] = {im["id"]: im for im in self._images}

        # Group annotations by image_id.
        self._anns_by_img: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            self._anns_by_img.setdefault(ann["image_id"], []).append(ann)

        # Filter out images with no annotations — Keypoint R-CNN cannot
        # learn from empty samples.
        self._image_ids: list[int] = [
            im["id"]
            for im in self._images
            if self._anns_by_img.get(im["id"])
        ]

        slc = _LANDMARK_SLICES[landmark]
        self._slice_start, self._slice_end = slc
        self._num_keypoints = _LANDMARK_NUM_KEYPOINTS[landmark]
        self._landmark = landmark

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

        # to_tensor: HWC uint8 -> CHW float32 in [0,1]
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        anns = self._anns_by_img[img_id]
        boxes: list[list[float]] = []
        labels: list[int] = []
        kps: list[list[list[float]]] = []

        for ann in anns:
            # COCO bbox is [x, y, w, h]; torchvision wants [x1, y1, x2, y2].
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])

            # Per dataset convention: category_id 1 = single-rooted,
            # category_id 2 = double-rooted. Adapter normalizes this.
            labels.append(int(ann.get("category_id", 1)))

            flat = ann["keypoints"]  # length 3 * total_keypoints (e.g. 18)
            # Reshape to (total_keypoints, 3).
            kp_all = np.asarray(flat, dtype=np.float32).reshape(-1, 3)
            sliced = kp_all[self._slice_start : self._slice_end]
            if sliced.shape[0] != self._num_keypoints:
                # Pad with invisible points if annotation is short.
                pad = np.zeros((self._num_keypoints - sliced.shape[0], 3), dtype=np.float32)
                sliced = np.vstack([sliced, pad])
            kps.append(sliced.tolist())

        if not boxes:
            # All annotations rejected (degenerate boxes). Skip by returning
            # next image; recursion bounded by len(dataset).
            return self.__getitem__((idx + 1) % len(self._image_ids))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "keypoints": torch.as_tensor(kps, dtype=torch.float32),
            "image_id": torch.as_tensor([img_id], dtype=torch.int64),
        }
        return img_tensor, target


def _collate(
    batch: list[tuple[torch.Tensor, dict]],
) -> tuple[list[torch.Tensor], list[dict]]:
    """Detection collate — keeps images + targets as lists (variable size)."""
    imgs, targets = zip(*batch, strict=True)
    return list(imgs), list(targets)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _build_model(num_keypoints: int) -> nn.Module:
    """Build Keypoint R-CNN with pretrained backbone, fresh head.

    Mirrors the brief's `pretrained=False, pretrained_backbone=True` shape
    using the current torchvision API (`weights=None,
    weights_backbone=DEFAULT`).
    """
    model = keypointrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT,
        num_classes=_NUM_CLASSES,
        num_keypoints=num_keypoints,
    )
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _move_targets(targets: list[dict], device: torch.device) -> list[dict]:
    return [{k: v.to(device) for k, v in t.items()} for t in targets]


def _train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for imgs, targets in loader:
        imgs = [im.to(device) for im in imgs]
        targets = _move_targets(targets, device)
        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += float(loss.detach().cpu())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def _val_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    # torchvision detection models only return loss dicts in train mode.
    # We temporarily set train() but disable grad to compute val loss.
    model.train()
    total = 0.0
    n = 0
    for imgs, targets in loader:
        imgs = [im.to(device) for im in imgs]
        targets = _move_targets(targets, device)
        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())
        total += float(loss.detach().cpu())
        n += 1
    return total / max(n, 1)


def train(
    landmark: Literal["cej", "bone", "apex"],
    dataset_dir: Path,
    weights_out: Path,
    epochs: int = 200,
) -> Path:
    """Train one Keypoint R-CNN model for a single landmark group.

    Args:
        landmark: Which landmark group to train on. Determines the
            num_keypoints (CEJ=2, bone=2, apex=1) and which slice of the
            COCO-keypoints array is used as supervision.
        dataset_dir: Directory containing `train/`, `val/` subdirectories
            with `images/` and `annotations.json` in COCO-keypoint format.
        weights_out: Destination `.pt` path; saved as state_dict.
        epochs: Maximum training epochs (default 200); early stopping may
            terminate sooner via `_EARLY_STOP_PATIENCE` on val loss.

    Returns:
        Absolute path to the saved state_dict file.
    """
    if landmark not in _LANDMARK_SLICES:
        raise ValueError(f"landmark must be cej/bone/apex; got {landmark!r}")

    dataset_dir = Path(dataset_dir).resolve()
    weights_out = Path(weights_out).resolve()
    weights_out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOG.info("training keypoint model: landmark=%s device=%s", landmark, device)

    train_ds = CocoKeypointSlice(dataset_dir / "train", landmark)
    val_ds = CocoKeypointSlice(dataset_dir / "val", landmark)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=_BATCH_TRAIN,
        shuffle=True,
        collate_fn=_collate,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=_BATCH_VAL,
        shuffle=False,
        collate_fn=_collate,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    num_keypoints = _LANDMARK_NUM_KEYPOINTS[landmark]
    model = _build_model(num_keypoints=num_keypoints).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        params, lr=_LR, weight_decay=_WEIGHT_DECAY
    )

    best_val = float("inf")
    best_state: dict | None = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, device)
        val_loss = _val_one_epoch(model, val_loader, device)
        _LOG.info(
            "epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  patience=%d/%d",
            epoch,
            epochs,
            train_loss,
            val_loss,
            epochs_without_improvement,
            _EARLY_STOP_PATIENCE,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= _EARLY_STOP_PATIENCE:
                _LOG.info("early stopping at epoch %d (patience exhausted)", epoch)
                break

    if best_state is None:
        best_state = model.state_dict()

    # Save a small wrapper dict so loaders can reconstruct the model
    # without remembering num_keypoints out-of-band.
    payload = {
        "state_dict": best_state,
        "num_keypoints": num_keypoints,
        "num_classes": _NUM_CLASSES,
        "landmark": landmark,
        "best_val_loss": best_val,
    }
    torch.save(payload, weights_out)
    return weights_out


if __name__ == "__main__":  # pragma: no cover
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train Keypoint R-CNN for one landmark.")
    parser.add_argument("--landmark", choices=("cej", "bone", "apex"), required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    final = train(args.landmark, args.dataset_dir, args.out, args.epochs)
    print(f"saved: {final}")
