# Dental Radiograph CV Pipeline — Methodology Reference Brief

Source: read-only inspection of upstream repo at `/tmp/upstream-ref/`
(Wimalasiri et al., Scientific Reports 2026 / arxiv 2506.20522).
Repo has no LICENSE → all-rights-reserved. This document is a
methodology extraction. Reimplementers must write fresh code from
this brief without referencing upstream source.

---

## 1. Per-Component Summary

### 1.1 Tooth Detection (`Detection Tasks/`)

- **Files:** `yolov8-detect.py`, `config.yaml`
- **Model family:** Ultralytics YOLO. File is named `yolov8-detect.py` but
  the actual checkpoint loaded is `YOLO('yolov9e.pt')` (line 7);
  `yolov8n.pt` is commented out. Treat as **YOLOv9e** (the largest YOLOv9
  variant) in upstream; reimplementation may pin **YOLOv8x** or
  **YOLOv8m** for parity with rest of pipeline and weight availability.
- **Backbone:** built-in CSP backbone for the chosen variant.
- **Hyperparameters (verbatim from `yolov8-detect.py`):**
  - `epochs=200`
  - `imgsz=640`
  - `lr0=0.0001`
  - `optimizer='Adam'`
  - `batch=4`
  - `patience=25`
  - `device=[1]`
  - `resume=False`
- **Classes (`config.yaml`):** `0: single`, `1: double` — two classes
  encoding single-rooted vs multi-rooted tooth. **Not** a per-tooth-ID
  classifier. Note: rad-cli only needs binary "is tooth" detection unless
  downstream keypoint stage cares about root count (it does — see §1.2
  `num_classes=3`).
- **Input/output:** RGB 640×640 input; YOLO bbox output `[x, y, w, h,
  conf, class]`.
- **Loss:** Ultralytics default (CIoU + DFL + classification BCE).
- **Augmentations:** Ultralytics defaults (no Albumentations custom
  pipeline used at detection stage).
- **Tricks:** Adam (not SGD), small batch (4), high patience (25).

### 1.2 Keypoint Detection (`Keypoint Detection/`)

- **Files:** `main.py`, `model.py`, `dataset.py`, `train_loop.py`,
  `train_val_funcs.py`, `data_augmentation.py`, `data_prep.py`,
  `evaluation_map.py`, `evaluation_oks.py`, `Keypoint_R_CNN_Script.py`
  (monolithic notebook-as-script).
- **Architecture:** `torchvision.models.detection.keypointrcnn_resnet50_fpn`
  with `pretrained=False, pretrained_backbone=True`.
- **num_classes:** `3` (background + single-rooted tooth + double-rooted
  tooth). `model.py` line "num_classes=3 # e.g., 1 background + 2 object
  types".
- **num_keypoints:** **PARAMETERIZED, NOT FIXED.** `main.py` has
  `get_model(num_keypoints=)` (empty arg — broken on master).
  `Keypoint_R_CNN_Script.py:283` sets `num_keypoints=2`. `note.txt` says
  "code can be adapted for all types of keypoints" — i.e., **one model
  per keypoint type**: train separately for (a) CEJ keypoints (2/tooth),
  (b) AEAC / bone-crest keypoints (2/tooth), (c) APEX keypoints
  (1–2/tooth). `dataset.py:31` slice `keypoints_original[4:]` extracts
  ONLY the apex pair from a 3-pair annotation file (CEJ at [0:2], AEAC
  at [2:4], APEX at [4:]). Reshape hardcoded `(-1, 2, 2)` confirms
  2 keypoints per tooth per model.
- **Anchor generator (defined but not wired in):** sizes `(32, 64, 128,
  256, 512)`, aspect ratios `(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0)`.
  Line `rpn_anchor_generator=anchor_generator` is **commented out** —
  the model runs with default RPN anchors. Reimplementer choice: enable
  the custom anchors (small + extreme aspect ratios suit narrow teeth)
  or stay default.
- **Hyperparameters (`Keypoint_R_CNN_Script.py:293–296`):**
  - Optimizer: `torch.optim.Adam(lr=0.0001, weight_decay=1e-6)`
  - Scheduler: `StepLR(step_size=4, gamma=0.6)` — but `lr_scheduler.step()`
    is commented out in the active training loop. Practical behavior:
    constant LR.
  - Alternative commented-out: `SGD(lr=0.0001, momentum=0.95,
    weight_decay=0.00005)` + `StepLR(step_size=5, gamma=0.3)`.
  - `n_epochs = 1` in the committed script (intentionally a stub for the
    upstream maintainer; real training is N epochs with early stopping).
  - Early-stopping `patience=10` on val loss.
- **Batch:** train 8 / val 4 / test 4 (`data_prep.py`).
- **Augmentations (`data_augmentation.py`):** Only **CLAHE** with
  `clip_limit=40.0, tile_grid_size=(8,8), p=1.0`. All other augmentations
  (HorizontalFlip, VerticalFlip, RandomBrightnessContrast, Rotate,
  Normalize) are commented out. Same CLAHE applied at val time —
  enhancement is always-on, not a randomized augmentation.
- **Input:** RGB image read by `cv2.imread` → `cv2.cvtColor(BGR2RGB)` →
  Albumentations CLAHE → `F.to_tensor` (scales 0–1). No resizing in the
  pipeline; original DenPAR resolution preserved.
- **Output:** Keypoint R-CNN standard — per-detection
  `{boxes, labels, scores, keypoints, keypoints_scores}` where
  `keypoints` shape is `[N_dets, num_keypoints, 3]` (x, y, visibility).
- **Loss:** sum of `loss_classifier + loss_box_reg + loss_objectness +
  loss_rpn_box_reg + loss_keypoint` (torchvision standard).
- **Tricks:** CLAHE-as-only-aug is the load-bearing image-quality trick;
  removing it likely tanks accuracy on dim DenPAR images.

### 1.3 Tooth + Bone Segmentation (`Segmentation Tasks/`)

- **Files:** `yolov8-seg.py`, `config.yaml`
- **Model:** `YOLO('yolov8x-seg.pt')` — YOLOv8 **extra-large**
  instance-segmentation variant.
- **Hyperparameters:** identical to detection — `epochs=200, imgsz=640,
  lr0=0.0001, optimizer='Adam', batch=4, patience=30, resume=False`.
- **Classes (`config.yaml`):** `nc:1, names:['bone']`. Comment in config
  says "use names:['tooth'] for tooth detection". So **two separately-
  trained models**: one with class `bone`, one with class `tooth`. Same
  script, two configs, two checkpoints.
- **Input/output:** 640×640; standard YOLO-seg output (bboxes + binary
  mask per instance).
- **Loss:** Ultralytics seg defaults (box + cls + DFL + segmentation).
- **Augmentations:** Ultralytics defaults.

### 1.4 Severity + Pattern Logic (`Severity and Bone Loss Pattern Calculation/`)

- **Files:** `0_mask_to_centerline.ipynb`, `1_bone_loss_angle_and_pattern.ipynb`
- **CRITICAL FINDING:** The upstream notebooks compute **pattern (H vs
  A)** but **do NOT compute % bone loss / severity stages**. Severity
  staging (mild/moderate/severe) is described in the paper but is
  **not** in this code. Reimplementer must derive % bone loss from
  keypoints separately — see §3.

- **What the notebooks do (notebook 0 — mask to centerline):**
  1. Denormalize YOLO-seg polygon `.txt` outputs (multiply by image
     width/height).
  2. For each bone polygon, build a circular doubly-linked list of
     vertices, find leftmost + rightmost as "ends," walk both halves
     forward and backward, average paired points → midline / centerline
     of the bone polygon. Output: a polyline representing the bone
     ridge.

- **What the notebooks do (notebook 1 — pattern classification):**
  1. `classify_image(keypoints)` — decide **maxillary vs mandibular**
     per IOPA: compare CEJ y-coordinates (`keypoints[i][0][1]` and
     `[i][1][1]`) against apex y-coordinates (`[i][4:]`). If CEJ is
     above (smaller y) the apex → **mandibular**; else **maxillary**.
     Majority vote across teeth.
  2. `remove_points_inside_masks(bones, masks)` — ray-cast each bone
     polyline point against each tooth polygon; drop interior points.
  3. For each remaining bone-line endpoint:
     - Find nearest tooth-mask vertex.
     - Walk `MASK_SEARCH_DISTANCE=20` indices along the mask polygon to
       get a second point on the apex-side of the tooth (sign chosen
       based on maxillary/mandibular classification).
     - Extend that mask edge by `LINE_EXTENSION` (20 or 40 in different
       cells) to form a "tooth-tangent" line.
     - Get bone-tangent vector from endpoint to `bone[POINTS_AWAY]`
       where `POINTS_AWAY = len(bone) // 3` or `// 4` (varies across
       cells — pick `// 4`).
     - Compute angle between the two vectors via dot product.
  4. **Pattern classification rule:**
     - `angle ≤ 55°` (one cell uses 50°) → **Angular (A)** ≡ vertical
       defect.
     - `angle > 55°` → **Horizontal (H)** ≡ horizontal bone loss.
     - Constant `ANGULAR_ANGLE = 55`. Use **55°** for reimplementation
       (the 50° cell appears to be an older draft).
- **Constants summary** (use these):
  - `LINE_EXTENSION = 20` (notebook constant, varies to 40 — use 20)
  - `MASK_SEARCH_DISTANCE = 20`
  - `SKIP_THRESHOLD = 20` (if bone endpoint is >20 px from nearest mask
    vertex, skip that endpoint)
  - `ANGULAR_ANGLE = 55` degrees
  - `POINTS_AWAY = len(bone) // 4`

---

## 2. Dataset Format Expected

### 2.1 What upstream assumes (DenPAR v2 flat layout, record 14181645)

- **Detection / segmentation (YOLO):**
  ```
  <root>/images/train/*.jpg
  <root>/images/val/*.jpg
  <root>/labels/train/*.txt   # YOLO format: class cx cy w h (normalized)
  <root>/labels/val/*.txt
  ```
  Path hardcoded in `config.yaml`:
  `/storage/scratch1/.../YOLO-detection/detect/images/train`.

- **Keypoint R-CNN (custom JSON):**
  ```
  <root>/train/images/*.jpg
  <root>/train/annotations/*.json
  <root>/val/images/*.jpg
  <root>/val/annotations/*.json
  <root>/test/images/*.jpg
  <root>/test/annotations/*.json
  ```
  Each `.json` (one per image):
  ```json
  {
    "bboxes":    [[x1,y1,x2,y2], ...],         // pascal_voc absolute pixels
    "keypoints": [[[x,y,vis], [x,y,vis], ...], ...],  // 3 pairs per tooth: CEJ[0:2], AEAC[2:4], APEX[4:6]
    "labels":    [1, 2, 1, ...]                // 1=single-rooted, 2=double-rooted
  }
  ```
  Visibility convention: `0` = absent, nonzero = visible (used as
  numeric weight in OKS).

### 2.2 DenPAR v3 (record 16645076) — what we use

DenPAR v3 ships split into `Training/`, `Validation/`, `Testing/`
subdirectories. The adapter needs to:

1. **Map v3 → v2 layout** (or directly to upstream's expected layout).
   ~20-line script:
   - Lowercase split names: `Training → train`, `Validation → val`,
     `Testing → test`.
   - For each split, find `images/` and the annotation files (DenPAR v3
     ships COCO JSON + YOLO txt — check the Zenodo readme on first
     extraction).
   - For YOLO models: symlink or copy images + label `.txt` into the
     `<root>/images/<split>/` + `<root>/labels/<split>/` shape.
   - For Keypoint R-CNN: convert COCO keypoint annotations into the
     per-image JSON shape above. Extract `bboxes`, `keypoints` (group
     in pairs per tooth: CEJ, AEAC, APEX), and `labels` (single vs
     double from `category_id`).
2. **Class label normalization:** DenPAR v3 may use different category
   IDs than upstream's `0:single, 1:double`. Map explicitly.
3. **Apex pair size variance:** `classify_image` in the severity
   notebook handles both `len(keypoint_array) == 6` (2 apex coords) and
   `== 5` (1 apex, single-rooted tooth). The keypoint training data has
   6 keypoints/tooth always, but single-rooted teeth have only 1
   apex — the second apex slot is presumably `[0,0,0]` (visibility=0).

---

## 3. Severity + Pattern Math (Exact Formulas)

### 3.1 Maxillary vs Mandibular Classifier

For each tooth's keypoint array `kp = [cej_l, cej_r, aeac_l, aeac_r, apex_a, apex_b]`:
```
cej_ys  = [y for y in (kp[0][1], kp[1][1]) if y != 0]
apex_ys = [y for y in (kp[4][1], kp[5][1]) if y != 0]   # or just kp[4][1] if 5-element
if not cej_ys or not apex_ys: classify as mandibular (default)
above = sum(1 for c in cej_ys if c < min(apex_ys))   # smaller y = higher in image
below = len(cej_ys) - above
tooth_label = "mandibular" if above >= below else "maxillary"
```
Aggregate across teeth in the IOPA: majority vote (ties → maxillary).

### 3.2 Pattern (Horizontal vs Angular) — Per Bone-Line Endpoint

```
For each bone (polyline of midpoints, denormalized to pixels):
  For end_idx in [0, -1]:
    p = bone[end_idx]
    # nearest tooth-mask vertex
    (q, mask_i, q_idx) = argmin_{m in masks, v in m} dist(p, v)
    if dist(p, q) >= SKIP_THRESHOLD: continue

    # second mask point, sign chosen by maxillary/mandibular
    delta = MASK_SEARCH_DISTANCE
    q2 = masks[mask_i][(q_idx + delta) % len(masks[mask_i])]
    if (classification == "mandibular" and q2.y < p.y) or
       (classification == "maxillary"  and q2.y > p.y):
        q2 = masks[mask_i][(q_idx - delta) % len(masks[mask_i])]

    # extend mask edge
    dx, dy = (q.x - q2.x), (q.y - q2.y)
    a = (q.x - LINE_EXTENSION*dx, q.y - LINE_EXTENSION*dy)
    b = (q.x + LINE_EXTENSION*dx, q.y + LINE_EXTENSION*dy)
    mask_vec = (b.x - a.x, b.y - a.y)

    # bone tangent vector
    k = len(bone) // 4
    bp = bone[k] if end_idx == 0 else bone[-1-k]
    bone_vec = (bp.x - p.x, bp.y - p.y)

    cos_theta = dot(bone_vec, mask_vec) / (|bone_vec| * |mask_vec|)
    theta_deg = degrees(arccos(cos_theta))

    pattern = "A" (angular/vertical) if theta_deg <= 55 else "H" (horizontal)
```

### 3.3 % Bone Loss / Severity — **NOT IN UPSTREAM CODE**

Upstream notebooks do not compute this. Standard clinical formula
(reimplementer should ship this; derive from paper, not from source):

```
For a tooth with CEJ keypoint c (midpoint of cej_l/cej_r),
            bone-crest keypoint b (midpoint of aeac_l/aeac_r),
            apex keypoint a (midpoint of present apex points):
  L_total = ||a - c||       # CEJ-to-apex distance (root length proxy)
  L_loss  = ||b - c||       # CEJ-to-bone-crest (the lost portion)
  pct_bone_loss = 100 * L_loss / L_total
```
Stage thresholds (paper / Stage I-III AAP staging — confirm in paper):
- `< 15%` → Stage I (mild)
- `15–33%` → Stage II (moderate)
- `> 33%` → Stage III (severe)

Use Euclidean distance on 2D pixel coords. If only one keypoint of a
pair is present (visibility>0), use that one; if a pair is missing
entirely, skip the tooth.

---

## 4. Inference Path

Upstream has no unified inference script — each model trains and
evaluates separately. Reimplementer should compose:

1. **Load image** → RGB → CLAHE (`clip=40.0, tile=(8,8)`) for keypoint
   model only; YOLO models eat raw RGB.
2. **Tooth detection** (YOLOv8/9 detect): get bboxes + `single/double`
   class. Confidence threshold default 0.25; NMS IoU 0.45 (YOLO
   defaults). Output: list of tooth bboxes.
3. **Bone segmentation** (YOLOv8-seg, `bone` class): get bone-region
   polygon(s).
4. **Tooth segmentation** (YOLOv8-seg, `tooth` class): get tooth
   polygons.
5. **Keypoint detection** (Keypoint R-CNN × 3 models, one each for CEJ
   / AEAC / APEX): each model returns per-tooth keypoints; merge into
   the 6-keypoint per-tooth structure.
6. **Centerline extraction** (notebook 0 algorithm): bone polygon →
   midline polyline.
7. **Maxillary/mandibular classification** from keypoints.
8. **Per-bone endpoint pattern** (H vs A) via §3.2.
9. **Per-tooth % bone loss + stage** via §3.3.
10. **Findings JSON**: list of teeth with bbox, class, keypoints, pct,
    stage, nearest-bone-pattern label.
11. **Annotated PNG**: draw bboxes, keypoints (color-coded: CEJ green,
    AEAC yellow, APEX purple per notebook 1), bone polyline blue, mask
    polygons green, A-endpoints circled red.
12. **Note draft**: template-fill from JSON.

---

## 5. Caries Detection

**Confirmed absent from upstream.** Grep for `caries|decay|cavity`
across all upstream files returns zero hits. The repo is bone-loss
only. Our project adds a separate YOLOv8s caries detector trained on
the Roboflow BW caries dataset — no contamination concern.

---

## 6. Reimplementation Gotchas

1. **`main.py` is broken on master:** line `get_model(num_keypoints=)`
   has empty arg. Don't blindly run it. Real value is 2 per the
   monolithic script.
2. **Three keypoint models, not one:** the slice `keypoints_original[4:]`
   in `dataset.py:31` extracts only APEX. To train CEJ + AEAC + APEX
   separately, swap the slice (`[:2]`, `[2:4]`, `[4:]`) and retrain.
   Reshape `(-1, 2, 2)` is hardcoded — works because each pair is 2
   keypoints; do not change to 3.
3. **LR scheduler is commented out** in the active training loop —
   constant LR is the actual behavior. `n_epochs=1` in committed
   script — clearly a stub. Set to real value (paper likely 50-100)
   and re-enable scheduler.step() or remove the scheduler.
4. **CLAHE is the only augmentation.** All commented-out augmentations
   (flips, rotations, brightness, normalize) are NOT used. CLAHE with
   `clip_limit=40.0` is aggressive (default is 4.0) — load-bearing for
   dim DenPAR contrast.
5. **`always_apply=True` is deprecated** in newer Albumentations; use
   `p=1.0` only (already set).
6. **Anchor generator defined but not wired** (commented). Default RPN
   anchors are what actually runs.
7. **`num_classes=3`** in keypoint R-CNN bakes in single/double rooted
   classification. If we don't care about root count, set to 2 (bg +
   tooth) — but then the `labels` field in annotations must be
   uniformly `1`.
8. **`SKIP_THRESHOLD=20`** silently drops bone endpoints far from any
   tooth mask — reasonable, but means some teeth report no pattern.
9. **`POINTS_AWAY` varies (`// 3` vs `// 4`)** across notebook cells.
   Use `// 4` (later cell, more conservative tangent).
10. **`ANGULAR_ANGLE` is 55° in one cell, 50° in another.** Use 55°
    (the explicit named constant; the 50° is a literal in a draft cell).
11. **Centerline algorithm assumes polygons are simple/convex-ish.** A
    bone polygon with branches or holes will produce a corrupted
    midline. Mitigation: keep only the largest contour per bone-class
    detection before passing to centerline.
12. **`maxillary/mandibular` defaults to mandibular** when CEJ or apex
    coords are missing — silent fallback. Reimplementation should log
    when this fallback fires (likely a low-quality IOPA).
13. **Path absoluteness:** every `config.yaml` and notebook hardcodes
    `/storage/scratch1/e18-4yp-bone-loss/...` paths. Strip these in
    fresh code; use a config object.
14. **`torch.save(model, save_path)`** saves the entire model object
    (not state_dict) — fragile across torchvision versions. Switch to
    `state_dict()` save/load.
15. **`A.CLAHE(always_apply=True)`** is per-call deprecated; modern API
    uses `p=1.0`.
16. **OKS evaluation uses uniform sigmas** (`torch.ones(N)/N`) — not
    COCO-style per-keypoint sigmas. Numbers are not comparable to COCO
    benchmarks.
17. **`reportlab` PDF generation** appears in notebook 1 — purely for
    side-by-side comparison reports; not needed in inference path.
18. **`wandb` import is commented out** — not required.
19. **`transforms` import** (`import transforms`) in `Keypoint_R_CNN_Script.py`
    references a torchvision detection helper that may not exist
    standalone; can be dropped.

---

## 7. What NOT to Reimplement

- **`Keypoint_R_CNN_Script.py`** — it's a notebook-converted-to-script
  with extensive commented-out experimentation (mean/std calculation,
  alternative optimizers, wandb logging, dataset-statistics block). The
  modular files (`main.py`, `model.py`, `dataset.py`, etc.) are the
  canonical structure — clone that shape, not the monolith.
- **PDF comparison report generation** (notebook 1, multiple cells
  using `reportlab.canvas.Canvas`) — purely for paper figures, not
  pipeline output.
- **`!rm -rf angles` / `!zip -r angular_marked.zip`** shell magics —
  notebook scaffolding.
- **Ground-truth visualization cells** (notebook 0, "FOR DISPLAYING
  GROUND TRUTH" block) — debugging helpers.
- **OKS evaluation with uniform sigmas** (`evaluation_oks.py`) — the
  uniform-sigma OKS is not a meaningful metric; skip or replace with
  per-keypoint MSE in pixel space.
- **Repeated maxillary/mandibular classifier** — it appears
  copy-pasted ~4 times in notebook 1. Factor to one function.
- **The denormalization step** (notebook 0, `process_txt_file`) — only
  needed if consuming YOLO-format normalized output; if your inference
  path already keeps masks in pixel coords, skip.
- **`logging_utils.py`** — appends to `log_file.txt`; use stdlib
  `logging` instead.
- **`save_model.py`** as-is — saves full model object; replace with
  state_dict save.
- **`data_prep.py` default-empty paths** (`train_path=''`) — replace
  with a proper config.
- **The commented-out anchor generator** in `model.py` — either enable
  it deliberately or remove the dead code.
