"""Autoresearch trainer for the CEJ keypoint head.

THIS FILE IS THE ONE THE AUTORESEARCH AGENT EDITS. Everything in here
is fair game: adapter pairing logic, model architecture, augmentation,
loss weights, optimizer, schedule, batch size, image size, filtering.

The two contracts that must remain stable for the experiment loop to
work are:

1. The metric is computed by ``prepare.evaluate_collapse_rate`` on
   the frozen 200-PA DenPAR Testing split. Whatever you do here, you
   must produce a Keypoint R-CNN-shaped model whose forward pass on
   the test images returns ``keypoints`` of shape ``(N, K>=2, 3)``
   so the harness can compute the mesial-distal CEJ distance from
   indices [0] and [1]. If you swap to a fundamentally different
   architecture (YOLO-pose, segmentation pivot), you must also
   write a compatibility wrapper that exposes the same interface to
   ``prepare.evaluate_collapse_rate`` — or note ``crash`` and move on.

2. The end-of-run summary block (see ``_print_summary``) must print
   the seven lines below verbatim so the experiment loop can grep
   the metric:

       ---
       cej_collapse_rate: <float>
       training_seconds: <float>
       total_seconds: <float>
       peak_memory_mb: <float>
       num_epochs: <int>
       num_train_samples: <int>
       device: <str>

The wall-clock training budget is enforced by ``_train_budget_seconds``
below. The default is 600 s (10 min) on MPS. Override with the
environment variable ``AR_BUDGET_SECONDS`` if you need to.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import resource
import sys
import time
from pathlib import Path
from typing import Final

# Make prepare.py importable from this dir.
_HERE: Final[Path] = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import prepare  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.utils.data  # noqa: E402
import torchvision  # noqa: E402
from torch import nn  # noqa: E402
from torchvision.models.detection import keypointrcnn_resnet50_fpn  # noqa: E402

_LOG = logging.getLogger("autoresearch.cej")


# ---------------------------------------------------------------------------
# Time budget (fair-comparison knob — keep stable unless re-baselining).
# ---------------------------------------------------------------------------


def _resolve_budget() -> float:
    override = os.environ.get("AR_BUDGET_SECONDS")
    if override:
        return float(override)
    # 600 s on MPS, 300 s on CUDA (faster). CPU fallback gets 900 s.
    if torch.cuda.is_available():
        return 300.0
    if torch.backends.mps.is_available():
        return 600.0
    return 900.0


_TRAIN_BUDGET_SECONDS: Final[float] = _resolve_budget()


# ---------------------------------------------------------------------------
# Hyperparameters — all knobs the agent should sweep live here.
# ---------------------------------------------------------------------------

# Optimizer + schedule
LR: float = 1e-4
WEIGHT_DECAY: float = 1e-6
SCHEDULE: str = "constant"  # one of: constant, step, cosine, plateau

# Data
BATCH_TRAIN: int = 4   # MPS RAM-friendly; bump for CUDA
BATCH_VAL: int = 2
IMAGE_LONG_SIDE: int | None = None  # None = native; else resize so max(h,w)=value

# CLAHE (training-side; eval-side is frozen at 40.0 by harness)
TRAIN_CLAHE_CLIP_LIMIT: float = 40.0

# Augmentation
AUG_HFLIP_PROB: float = 0.0     # keypoint flip-pair semantics complicate this; default off
AUG_ROTATION_DEG: float = 0.0   # ±deg around center; 0 = off
AUG_BRIGHTNESS: float = 0.0     # multiplicative ±frac; 0 = off

# Filtering — which tooth annotations make it into the training set
FILTER_REQUIRE_BOTH_CEJ: bool = False  # if True, drop teeth missing either CEJ point

# Loss reweighting
LOSS_WEIGHTS: dict[str, float] = {
    "loss_classifier": 1.0,
    "loss_box_reg": 1.0,
    "loss_objectness": 1.0,
    "loss_rpn_box_reg": 1.0,
    "loss_keypoint": 1.0,
}

# Adapter — pairing heuristic for loose CEJ points → bbox
# Options:
#   "default"       — current adapter (containment + nearest-center fallback)
#   "hungarian"     — global assignment (requires scipy.optimize)
#   "two-only"      — drop teeth with !=2 CEJ points (cleanest signal)
ADAPTER_PAIRING: str = "default"

# Architecture
ARCH: str = "keypointrcnn_r50_fpn"  # or "keypointrcnn_r50_fpn_pretrained_kp"

# Early-stop within the budget (epochs without val improvement)
EARLY_STOP_PATIENCE: int = 20

NUM_CLASSES: Final[int] = prepare.NUM_CLASSES  # frozen: 3 (bg + single + double)
NUM_KEYPOINTS: Final[int] = 2  # CEJ pair; frozen by metric


# ---------------------------------------------------------------------------
# Dataset (forked from training/keypoints.py::CocoKeypointSlice).
# ---------------------------------------------------------------------------


def _apply_clahe(image_rgb: np.ndarray, clip_limit: float) -> np.ndarray:
    """CLAHE — agent-tunable clip_limit, fixed tile grid."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


class CejCocoDataset(torch.utils.data.Dataset):
    """COCO-keypoint dataset sliced to CEJ pair, with optional filtering."""

    # Slice into the 18-element flat keypoint list emitted by the adapter:
    # [cej_l, cej_r, bone_l, bone_r, apex_a, apex_b] x (x,y,vis).
    _CEJ_START: Final[int] = 0
    _CEJ_END: Final[int] = 2

    def __init__(
        self,
        split_dir: Path,
        clahe_clip: float,
        require_both_cej: bool,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.images_dir = self.split_dir / "images"
        ann_path = self.split_dir / "annotations.json"
        with ann_path.open() as fh:
            coco = json.load(fh)

        self._img_by_id = {im["id"]: im for im in coco["images"]}
        self._anns_by_img: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            kps = np.asarray(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
            cej = kps[self._CEJ_START : self._CEJ_END]
            if require_both_cej and not (cej[:, 2] > 0).all():
                continue
            # Also drop teeth with no visible CEJ at all — model can't learn.
            if not (cej[:, 2] > 0).any():
                continue
            self._anns_by_img.setdefault(ann["image_id"], []).append(ann)

        self._image_ids = [i for i in self._img_by_id if self._anns_by_img.get(i)]
        self._clahe_clip = clahe_clip

    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        img_id = self._image_ids[idx]
        info = self._img_by_id[img_id]
        bgr = cv2.imread(str(self.images_dir / info["file_name"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(self.images_dir / info["file_name"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = _apply_clahe(rgb, self._clahe_clip)
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        boxes: list[list[float]] = []
        labels: list[int] = []
        kps: list[list[list[float]]] = []
        for ann in self._anns_by_img[img_id]:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(int(ann.get("category_id", 1)))
            flat = np.asarray(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
            cej = flat[self._CEJ_START : self._CEJ_END]
            if cej.shape[0] < NUM_KEYPOINTS:
                pad = np.zeros((NUM_KEYPOINTS - cej.shape[0], 3), dtype=np.float32)
                cej = np.vstack([cej, pad])
            kps.append(cej.tolist())

        if not boxes:
            return self.__getitem__((idx + 1) % len(self._image_ids))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "keypoints": torch.as_tensor(kps, dtype=torch.float32),
            "image_id": torch.as_tensor([img_id], dtype=torch.int64),
        }
        return img_tensor, target


def _collate(batch):
    imgs, targets = zip(*batch, strict=True)
    return list(imgs), list(targets)


# ---------------------------------------------------------------------------
# Model.
# ---------------------------------------------------------------------------


def build_model() -> nn.Module:
    """Build the model the agent wants to train.

    Default mirrors training/keypoints.py: Keypoint R-CNN ResNet50-FPN
    with pretrained backbone + fresh head. Agent may swap entirely —
    but the saved payload must remain loadable by
    ``prepare.load_keypoint_model``.
    """
    return keypointrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT,
        num_classes=NUM_CLASSES,
        num_keypoints=NUM_KEYPOINTS,
    )


# ---------------------------------------------------------------------------
# Training loop with wall-clock budget.
# ---------------------------------------------------------------------------


def _move_targets(targets, device):
    return [{k: v.to(device) for k, v in t.items()} for t in targets]


def _weighted_loss(loss_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    return sum(LOSS_WEIGHTS.get(k, 1.0) * v for k, v in loss_dict.items())


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_scheduler(optimizer: torch.optim.Optimizer):
    if SCHEDULE == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    if SCHEDULE == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    if SCHEDULE == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
    return None  # constant


def train_model(prepared_dir: Path) -> tuple[Path, dict]:
    """Train until the wall-clock budget is exhausted.

    Returns (checkpoint_path, stats_dict).
    """
    device = _resolve_device()
    _LOG.info("device=%s budget=%.0fs", device, _TRAIN_BUDGET_SECONDS)

    train_ds = CejCocoDataset(
        prepared_dir / "train",
        clahe_clip=TRAIN_CLAHE_CLIP_LIMIT,
        require_both_cej=FILTER_REQUIRE_BOTH_CEJ,
    )
    val_ds = CejCocoDataset(
        prepared_dir / "val",
        clahe_clip=TRAIN_CLAHE_CLIP_LIMIT,
        require_both_cej=FILTER_REQUIRE_BOTH_CEJ,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_TRAIN,
        shuffle=True,
        collate_fn=_collate,
        num_workers=0,   # MPS doesn't like worker forks
        pin_memory=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=BATCH_VAL,
        shuffle=False,
        collate_fn=_collate,
        num_workers=0,
        pin_memory=False,
    )

    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = _make_scheduler(optimizer)

    best_val = float("inf")
    best_state: dict | None = None
    epochs_done = 0
    epochs_without_improvement = 0

    train_start = time.monotonic()

    while True:
        # Per-epoch budget check at start.
        elapsed = time.monotonic() - train_start
        if elapsed >= _TRAIN_BUDGET_SECONDS:
            _LOG.info("budget exhausted after %d epochs (%.0fs)", epochs_done, elapsed)
            break

        # ---- train one epoch (with mid-epoch budget check) ----
        model.train()
        train_total = 0.0
        train_n = 0
        for imgs, targets in train_loader:
            imgs = [im.to(device) for im in imgs]
            targets = _move_targets(targets, device)
            loss_dict = model(imgs, targets)
            loss = _weighted_loss(loss_dict)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach().cpu())
            train_n += 1
            if time.monotonic() - train_start >= _TRAIN_BUDGET_SECONDS:
                break
        train_loss = train_total / max(train_n, 1)

        # ---- val one epoch ----
        model.train()  # detection models need train() mode for loss dict
        val_total = 0.0
        val_n = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = [im.to(device) for im in imgs]
                targets = _move_targets(targets, device)
                loss_dict = model(imgs, targets)
                val_total += float(_weighted_loss(loss_dict).detach().cpu())
                val_n += 1
        val_loss = val_total / max(val_n, 1)
        epochs_done += 1

        elapsed = time.monotonic() - train_start
        _LOG.info(
            "epoch=%d train=%.4f val=%.4f elapsed=%.0fs",
            epochs_done, train_loss, val_loss, elapsed,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                _LOG.info("early-stop on val plateau")
                break

        if scheduler is not None:
            if SCHEDULE == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

    if best_state is None:
        best_state = model.state_dict()

    train_seconds = time.monotonic() - train_start
    checkpoint = _HERE / "checkpoint.pt"
    payload = {
        "state_dict": best_state,
        "num_keypoints": NUM_KEYPOINTS,
        "num_classes": NUM_CLASSES,
        "landmark": "cej",
        "best_val_loss": best_val,
    }
    torch.save(payload, checkpoint)

    stats = {
        "training_seconds": train_seconds,
        "num_epochs": epochs_done,
        "num_train_samples": len(train_ds),
        "device": str(device),
    }
    return checkpoint, stats


# ---------------------------------------------------------------------------
# Summary printer (FORMAT IS LOAD-BEARING — grep'd by the experiment loop).
# ---------------------------------------------------------------------------


def _peak_memory_mb() -> float:
    """Best-effort RSS peak (Mac/Linux). MPS doesn't expose VRAM well."""
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # macOS reports in bytes, Linux in KB. Detect by magnitude.
        rss = ru.ru_maxrss
        if rss > 1_000_000_000:  # huge ⇒ bytes (macOS)
            return rss / (1024 * 1024)
        return rss / 1024  # KB → MB
    except Exception:
        return 0.0


def _print_summary(
    collapse_rate: float,
    training_seconds: float,
    total_seconds: float,
    num_epochs: int,
    num_train_samples: int,
    device: str,
) -> None:
    print("---")
    print(f"cej_collapse_rate: {collapse_rate:.4f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds: {total_seconds:.1f}")
    print(f"peak_memory_mb: {_peak_memory_mb():.1f}")
    print(f"num_epochs: {num_epochs}")
    print(f"num_train_samples: {num_train_samples}")
    print(f"device: {device}")


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    overall_start = time.monotonic()

    prepared = prepare.ensure_coco_keypoints()
    prepared_dir = prepared.parent.parent  # .../keypoints

    checkpoint, stats = train_model(prepared_dir)

    eval_device = prepare._resolve_device()
    collapse = prepare.evaluate_collapse_rate(checkpoint, device=eval_device)

    total_seconds = time.monotonic() - overall_start
    _print_summary(
        collapse_rate=collapse,
        training_seconds=stats["training_seconds"],
        total_seconds=total_seconds,
        num_epochs=stats["num_epochs"],
        num_train_samples=stats["num_train_samples"],
        device=stats["device"],
    )


if __name__ == "__main__":
    main()
