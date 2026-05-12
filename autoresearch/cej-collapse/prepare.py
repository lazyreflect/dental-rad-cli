"""Fixed measurement harness for the CEJ-collapse autoresearch loop.

DO NOT MODIFY THIS FILE. The autoresearch agent edits only ``train.py``.
``prepare.py`` defines the ground-truth metric — change it and every
prior result on ``results.tsv`` becomes incomparable.

What this provides
------------------
- ``ensure_coco_keypoints(...)``: idempotently materializes the
  DenPAR v3 → COCO-keypoint dataset under
  ``data/denpar/prepared/keypoints/{train,val,test}/`` using the
  current ``denpar_adapter.build_coco_keypoints``. The agent's
  ``train.py`` may use these files OR may re-derive its own training
  data (e.g. with a different pairing heuristic). The TEST split
  is the frozen evaluation surface — every experiment is measured
  on the same 200 DenPAR Testing PAs.
- ``evaluate_collapse_rate(model_path, device=...)``: loads keypoint
  weights, runs inference on all 200 PAs in
  ``data/denpar/Dataset/Testing/Images/``, and returns the fraction of
  high-confidence predictions whose mesial-distal CEJ distance is
  less than ``COLLAPSE_THRESHOLD_PX`` pixels. Lower is better. The
  baseline (commit-of-record at the start of the autoresearch run)
  is **0.3071** on ``weights/keypoint_cej.pt``.
- A small set of fixed constants the harness uses (score threshold,
  collapse threshold, CLAHE clip limit). The agent's ``train.py``
  must NOT change these.

Notes
-----
* CLAHE-at-eval is fixed at the harness level (clip=40.0, the value
  that produced the baseline). The agent may try different CLAHE
  values during TRAINING, but the eval forward pass always normalizes
  using the same preprocessing.
* The harness loads weights in the same payload format
  ``training/keypoints.py`` produces: a dict with keys
  ``state_dict``, ``num_keypoints``, ``num_classes``.
* MPS is the default device. Joseph is on a Mac M4 Max. CUDA path
  is supported as a fallback for whoever runs this elsewhere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

# Add the repo's src/ to sys.path so we can import dental_rad_cli without
# an editable install.
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_SRC: Final[Path] = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
from torchvision.models.detection import keypointrcnn_resnet50_fpn  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen constants — DO NOT CHANGE during autoresearch.
# ---------------------------------------------------------------------------

#: Confidence threshold for "high-confidence prediction". Predictions
#: below this score are excluded from the collapse-rate denominator.
SCORE_THRESHOLD: Final[float] = 0.5

#: Distance in pixels below which a mesial-distal CEJ pair is considered
#: collapsed (the failure mode we're driving down).
COLLAPSE_THRESHOLD_PX: Final[float] = 10.0

#: CLAHE clip limit used at EVAL time (matches the baseline-producing
#: preprocessor). Trainers may augment differently, but the eval forward
#: pass uses this fixed value.
EVAL_CLAHE_CLIP_LIMIT: Final[float] = 40.0
EVAL_CLAHE_TILE_GRID: Final[tuple[int, int]] = (8, 8)

#: Where the prepared COCO-keypoint dataset lives. Agent's train.py
#: SHOULD point at this when using the default adapter; can re-derive
#: into a sibling path if exploring different pairing heuristics.
DEFAULT_PREPARED_DIR: Final[Path] = _REPO_ROOT / "data" / "denpar" / "prepared" / "keypoints"

#: DenPAR v3 root (containing ``Dataset/``).
DENPAR_ROOT: Final[Path] = _REPO_ROOT / "data" / "denpar"

#: The 200 held-out PAs the metric is computed on. NEVER train on these.
TEST_IMAGES_DIR: Final[Path] = DENPAR_ROOT / "Dataset" / "Testing" / "Images"

#: Number of object classes used by Keypoint R-CNN (bg + single + double).
#: Frozen — payload format depends on this.
NUM_CLASSES: Final[int] = 3


# ---------------------------------------------------------------------------
# Image preprocessing (FROZEN — eval side).
# ---------------------------------------------------------------------------


def _eval_clahe(image_rgb: np.ndarray) -> np.ndarray:
    """CLAHE preprocessing used at eval time. Frozen.

    Matches the baseline trainer's preprocessor with the harness-fixed
    clip limit. The agent's train.py may apply different augmentation
    during training but the eval forward pass uses THIS function.
    """
    clahe = cv2.createCLAHE(
        clipLimit=EVAL_CLAHE_CLIP_LIMIT,
        tileGridSize=EVAL_CLAHE_TILE_GRID,
    )
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _load_image_tensor(path: Path) -> torch.Tensor:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = _eval_clahe(rgb)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


# ---------------------------------------------------------------------------
# COCO dataset preparation (idempotent).
# ---------------------------------------------------------------------------


def ensure_coco_keypoints(prepared_dir: Path = DEFAULT_PREPARED_DIR) -> Path:
    """Build the COCO-keypoint dataset if not already present.

    Returns the path to the train-split annotations.json. Idempotent —
    skips work if all three split annotations exist.
    """
    prepared_dir = Path(prepared_dir)
    expected = [
        prepared_dir / "train" / "annotations.json",
        prepared_dir / "val" / "annotations.json",
        prepared_dir / "test" / "annotations.json",
    ]
    if all(p.exists() for p in expected):
        return expected[0]

    # Import lazily so prepare.py is importable even if the adapter
    # has a transient issue.
    from dental_rad_cli.data.denpar_adapter import build_coco_keypoints

    return build_coco_keypoints(DENPAR_ROOT, prepared_dir, landmark="cej")


# ---------------------------------------------------------------------------
# Model loading (matches training/keypoints.py payload format).
# ---------------------------------------------------------------------------


def _build_keypoint_model(num_keypoints: int) -> torch.nn.Module:
    """Build the same architecture used by training/keypoints.py.

    The agent's train.py may swap to a different architecture, but if
    it does, it MUST save weights in a format this loader recognizes
    (state_dict + num_keypoints + num_classes), OR override this
    builder by setting a sibling ``build_model`` hook (see notes in
    train.py).
    """
    return keypointrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT,
        num_classes=NUM_CLASSES,
        num_keypoints=num_keypoints,
    )


def load_keypoint_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    """Load a Keypoint R-CNN model from the canonical payload.

    Payload schema (from training/keypoints.py::train):
        {"state_dict": ..., "num_keypoints": int, "num_classes": int,
         "landmark": str, "best_val_loss": float}
    """
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(
            f"unexpected weight payload at {model_path}; expected "
            "training/keypoints.py format"
        )
    num_kp = int(payload.get("num_keypoints", 2))
    model = _build_keypoint_model(num_keypoints=num_kp)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Metric.
# ---------------------------------------------------------------------------


def evaluate_collapse_rate(
    model_path: Path,
    device: str | torch.device = "mps",
) -> float:
    """Compute CEJ collapse rate on the 200 DenPAR Testing PAs.

    Definition:
        For each test image, run the keypoint model; collect predictions
        with ``scores >= SCORE_THRESHOLD`` (high-confidence). For each
        such prediction, compute the Euclidean distance between the two
        predicted CEJ keypoints. The collapse rate is:

            (# high-conf preds with dist < COLLAPSE_THRESHOLD_PX)
            ---------------------------------------------------
                       (# high-conf preds total)

    Lower is better. Baseline on ``weights/keypoint_cej.pt`` is 0.3071.

    Returns:
        Float in [0, 1]. Returns 1.0 if there are no high-conf preds
        (degenerate — model collapsed entirely) so the agent can still
        compare against the baseline.
    """
    device = torch.device(device) if isinstance(device, str) else device
    model = load_keypoint_model(Path(model_path), device)

    images = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(
            f"no test images in {TEST_IMAGES_DIR} — DenPAR v3 not unpacked?"
        )

    total = 0
    collapsed = 0

    with torch.no_grad():
        for img_path in images:
            tensor = _load_image_tensor(img_path).to(device)
            try:
                outputs = model([tensor])
            except RuntimeError:
                # MPS occasionally chokes on a particular tensor shape;
                # fall back to CPU for that one image rather than abort.
                model_cpu = model.to("cpu")
                outputs = model_cpu([tensor.to("cpu")])
                model.to(device)

            out = outputs[0]
            scores = out["scores"].detach().cpu().numpy()
            kps = out["keypoints"].detach().cpu().numpy()  # (N, K, 3)

            mask = scores >= SCORE_THRESHOLD
            if not mask.any():
                continue

            kps_hc = kps[mask]
            if kps_hc.shape[1] < 2:
                continue

            # Distance between keypoint[0] and keypoint[1] per prediction.
            dx = kps_hc[:, 0, 0] - kps_hc[:, 1, 0]
            dy = kps_hc[:, 0, 1] - kps_hc[:, 1, 1]
            dists = np.hypot(dx, dy)

            total += int(dists.shape[0])
            collapsed += int((dists < COLLAPSE_THRESHOLD_PX).sum())

    if total == 0:
        return 1.0
    return collapsed / total


# ---------------------------------------------------------------------------
# Sanity check entrypoint.
# ---------------------------------------------------------------------------


def _resolve_device(preferred: str = "mps") -> torch.device:
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _main() -> None:
    """Sanity-check entrypoint: ensure data is ready, optionally eval baseline."""
    print(f"repo root: {_REPO_ROOT}")
    print(f"denpar root: {DENPAR_ROOT}  exists={DENPAR_ROOT.exists()}")
    print(f"test images: {TEST_IMAGES_DIR}  count={len(list(TEST_IMAGES_DIR.glob('*.jpg')))}")

    ann = ensure_coco_keypoints()
    print(f"prepared train annotations: {ann}")

    weights = _REPO_ROOT / "weights" / "keypoint_cej.pt"
    if not weights.exists():
        print(f"NOTE: baseline weights not found at {weights}; skipping eval")
        print("ready")
        return

    device = _resolve_device()
    print(f"device: {device}")
    print(f"evaluating baseline at {weights} ...")
    rate = evaluate_collapse_rate(weights, device=device)
    print(f"baseline cej_collapse_rate: {rate:.4f}  (expected ~0.3071)")
    print("ready")


if __name__ == "__main__":
    _main()
