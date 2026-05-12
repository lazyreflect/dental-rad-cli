# Bitewing Bone-Loss + Architecture Deep-Dive

**Author:** research-session Claude (cold start)
**Date:** 2026-05-12
**Status:** Research only — no code written. Findings inform perio pipeline scoping
for bitewing-mode and architecture re-selection.
**Trigger:** Wimalasiri Keypoint R-CNN pipeline (DenPAR v3 / arxiv 2506.20522) is
periapical-only. Running it on bitewings produces catastrophic failures because
apex is cut off (denominator of the % bone-loss formula vanishes), the image
shows both arches, and 6-8 teeth occlude. Two questions: who has solved this
on BW, and what architectures dodge the apex requirement?

---

## TL;DR

**Bitewing-specific BL work is real but smaller than the PA literature, and
nobody has shipped a permissively-licensed end-to-end pipeline with code + data.**
The four published BW-focused groups (AlGhaihab/Denti.AI 2025, Lee/Columbia 2025,
Erturk/Necmettin Erbakan 2024, Akarsu et al. MDPI 2026) consistently solve the
"no apex on BW" problem in one of three ways:

1. **Switch unit:** use **mm distance CEJ→bone-crest** instead of % of root
   length (Denti.AI, Lee/Columbia, Erturk Eigen-CAM staging via coronal/middle
   third bins, Overjet commercial). Clinical RBL threshold then becomes the
   AAP/EFP cutoff (≥2 mm = mild, ≥4 mm = moderate, etc.) rather than the
   Wimalasiri %.
2. **Sidestep landmarks entirely:** train a **classifier** that bins
   AAP stage I/II/III/IV directly from the BW image (Erturk YOLOv8m-cls;
   Eigen-CAM for explainability). No keypoint geometry; no apex needed.
3. **Polynomial fit across multiple teeth in one arch** to recover an
   implicit reference axis (Lee/Columbia 5-network conglomerate). Apex
   not required because the arch shape constrains the reference.

**Strongest open candidates today:**
- **Banks et al. 2025 (arxiv 2503.13477)** — only end-to-end open keypoint
  pipeline (MIT license, weights on Google Drive, CC-BY data on Zenodo). PA-only
  in their evaluation but the architecture (HRNet / YOLOv8-Pose / DeepPose /
  RTMPose comparison + tooth-segmentation post-hoc alignment) is directly
  transferable. Their PRCK metric and stage-agnostic annotation discipline
  are the load-bearing contributions for our use case.
- **Sapiens-based GeoSapiens (arxiv 2507.04710)** — foundation model + LoRA +
  3-patient few-shot achieves 93% SDR@2mm on dental landmarks (CBCT, not BW —
  but the few-shot transferability is the interesting bit). MIT license, full
  code at `xmed-lab/GeoSapiens`.
- **finetune-SAM (mazurowski-lab)** — Apache 2.0, generic medical SAM/SAM2
  adapter framework, supports points/boxes/LoRA. No dental BW reference impl
  but adaptable. Probe-able in a weekend.

**Architecture recommendation (anticipating the "do we change?" question
without making the decision):**
- **For BW mode**: a **bone-crest polyline segmentation** head (matches what
  v3 already does for bone) + a **CEJ polyline segmentation** head + a
  **per-tooth bbox detector** for grouping is the lowest-risk replacement
  for the apex-dependent Keypoint R-CNN math. Distance is computed pixel-wise
  between the two polylines within each tooth bbox, calibrated to mm via
  tooth-type anatomical prior (mean MD width tables).
- **For PA mode**: keep Wimalasiri-style Keypoint R-CNN; the apex denominator
  is what makes that approach defensible for percent-of-root reporting.

**No-apex problem is solved in the literature.** It is the **classifier-only**
groups and the **mm-distance** groups that get usable BW numbers. The Wimalasiri
% formulation is genuinely PA-specific — porting it to BW is the wrong shape of
fix.

---

## 1. Constraints recap

| Constraint                  | Requirement                                        |
|-----------------------------|----------------------------------------------------|
| License                     | CC-BY 4.0 or more permissive (paper + code + data) |
| Modality                    | Bitewing (preferred), or works on both BW + PA     |
| Open access                 | Downloadable today, not IRB-gated                  |
| Training compute            | RTX 4090 or M-series Mac                           |
| Apex-in-frame requirement   | Must not require it (BW geometry)                  |
| Multi-tooth bbox            | Must handle 6-8 teeth, two arches per image        |

---

## 2. Part 1 — Bitewing-specific literature survey

### 2.1 AlGhaihab et al. 2025 — Denti.AI evaluation [verified]

| Property              | Value                                                  |
|-----------------------|--------------------------------------------------------|
| Citation              | AlGhaihab et al., *Diagnostics* 15(5):576, 2025 [verified] |
| Country               | USA (multi-institutional)                              |
| Modality              | **Bitewing + Periapical** [verified]                   |
| Code                  | Not released [verified]                                |
| Data                  | "Upon reasonable request" — restricted [verified]      |
| Architecture          | Faster R-CNN (tooth) + ResNet (numbering) + FPN+ResNet (landmarks: CEJ, restoration margin, bone level, root apex) [verified] |
| Approach              | Keypoint landmark detection on each tooth crop         |
| **No-apex handling**  | **For BW: "RBL = CEJ→bone-crest distance ≥ 2 mm"** (millimeter threshold). For PA: "≥15% root length." [verified] |
| BW metrics            | Sens 65%, Spec 90%, PPV 88%, NPV 70%, Acc 76%, **MAE 0.499 mm** [verified] |
| PA metrics            | Sens 76%, Spec 86%, PPV 83%, NPV 80%, Acc 81%, MAE 0.046% [verified] |
| License               | Paper CC-BY; software/data restricted [verified]       |

**Critical finding for our pipeline:** Denti.AI explicitly uses a **dual-mode**
formulation. On BW they switch to **mm CEJ→bone-crest** because apex is
unavailable; on PA they use the Wimalasiri-style **% of root length**. This is
the load-bearing engineering insight from the BW literature — the unit changes,
the architecture (per-tooth keypoint head) does not.

**Fit-to-our-use-case:** Methodology is the canonical reference. Code/data are
not open, so they are a reference design, not an artifact to fork. The MAE
0.499 mm on BW is the **performance ceiling target** for any open replication.

---

### 2.2 Lee et al. 2025 — Columbia 5-network conglomerate [verified]

| Property              | Value                                                  |
|-----------------------|--------------------------------------------------------|
| Citation              | Lee et al., *BMC Oral Health*, 2025 (s12903-025-05677-0) [verified] |
| Country               | USA (Columbia University)                              |
| Modality              | **Bitewing only** [verified]                           |
| Code                  | Not released [verified]                                |
| Data                  | 550 BW radiographs, single institution; restricted [verified] |
| Architecture          | **5-network conglomerate** [verified]:                 |
|                       | - 2× Faster R-CNN (Inception-ResNet-V2) for ABCL + CEJ landmark coords |
|                       | - DeepLab v3+ for ABCL semantic segmentation           |
|                       | - 2× DeepLab v3+ for tooth segmentation (upper + lower arches) |
|                       | - **Polynomial curve fit** to ABCL + CEJ points across each arch |
| **No-apex handling**  | **Implicit arch-shape reference**: polynomial fit across multiple teeth in one arch produces a smooth CEJ curve and a smooth ABCL curve. ACH (alveolar crestal height) is computed as the perpendicular distance between them. **No per-tooth apex needed.** [verified] |
| Metrics               | Overall accuracy 94% (vs. 68% dental professionals); 82-87% severe-perio detection; segmentation 0.93-0.96 global accuracy; CEJ object detection AP 0.72, ABCL AP 0.65 [verified] |
| License               | Paper CC-BY 4.0; data restricted [verified]            |

**Critical finding:** This is the **architectural template most worth copying**
for BW. The 5-network conglomerate has three load-bearing ideas:

1. **Two heads per landmark type** — bounding-box detector AND semantic
   segmentation network, with the segmentation refining the box-derived
   keypoints. The CEJ-collapse failure mode (Wimalasiri's known issue) is
   exactly what this dual-head approach mitigates.
2. **Polynomial across multiple teeth in one arch** as the implicit reference
   axis. This is the BW-native answer to "no apex." The smooth curve is
   anatomically motivated (CEJ heights are continuous along an intact arch).
3. **Per-arch separation** (upper vs lower tooth segmentation networks).
   Avoids the "multi-tooth bbox blob" failure mode by hard-conditioning on
   arch membership.

**Fit-to-our-use-case:** Highest design ROI per page of the BW literature.
The 5-network shape is heavier than Wimalasiri Keypoint R-CNN, but **every
component is open-source standard** (Faster R-CNN, DeepLab v3+, polynomial
fit). Replicable in ~2-3 weeks if dataset acquired.

---

### 2.3 Erturk, Öziç, Tassoker 2024 — Eigen-CAM BW staging [verified]

| Property              | Value                                                  |
|-----------------------|--------------------------------------------------------|
| Citation              | Erturk et al., *J Imaging Inform Med*, 2024 (10.1007/s10278-024-01218-3) [verified] |
| Country               | Turkey (Necmettin Erbakan University)                  |
| Modality              | **Bitewing only** [verified]                           |
| Code                  | Not released [verified]                                |
| Data                  | 1,752 BW images; "shared upon reasonable request" [verified] |
| Architecture          | **YOLOv8m-cls** (4-class image classifier) + Eigen-CAM [verified] |
| Approach              | **No keypoints, no segmentation, no math.** Image-level classifier directly outputs AAP/EFP stage. |
| **No-apex handling**  | **Sidestep entirely.** Classifier learns the visual signature of each stage. Eigen-CAM heatmap shows what the network attends to (validation, not measurement). [verified] |
| Classes               | Healthy / Mild (<15% coronal third) / Moderate (15-33% coronal third) / Severe (>33%, mid+apical with furcation) [verified] |
| Metrics               | 5-fold CV: Acc 83.45%, Prec 81.74%, Recall 80.88%, F1 81.09% [verified] |
| License               | Paper "exclusive licence to Society for Imaging Informatics in Medicine 2024" — **not CC-BY** [unverified — could be CC-BY-NC; needs probe] |
| Caveat                | Authors explicitly note: "severe alveolar bone losses cannot be detected with horizontal bite-wing images." Vertical BW (rare) required for severe-tier signal. [verified] |

**Critical finding:** This is the **"escape hatch"** approach. If keypoint
geometry on BW proves brittle, a classifier-only baseline can ship a
stage label without ever computing a CEJ-bone distance. Trade-off: no
mm number, no per-tooth granularity, no explainability beyond CAM.

**Fit-to-our-use-case:** Useful as a **fallback head** running parallel
to the geometric pipeline. If geometry says "incomputable," the
classifier still emits an AAP tier. Worth keeping in the architecture
roadmap as a defensive layer.

---

### 2.4 Akarsu et al. 2026 — Multi-class BW segmentation [verified]

| Property              | Value                                                  |
|-----------------------|--------------------------------------------------------|
| Citation              | *Diagnostics* 16(2):322, 2026 (mdpi.com/2075-4418/16/2/322) [verified] |
| Country               | [unverified — needs probe; MDPI listing]               |
| Modality              | **Bitewing only** [verified]                           |
| Code                  | Not released [verified]                                |
| Data                  | 1,197 BW with 7,860 labels across 8 classes; restricted [verified] |
| Architecture          | **YOLOv8x-seg** instance segmentation, 8 classes [verified] |
| Approach              | Per-finding polygon segmentation: alveolar bone loss, dental calculus, furcation, caries, cervical margin gaps, open contacts, overhanging fillings, secondary caries |
| **No-apex handling**  | **Bone loss as a region-of-interest segmentation**, not a measurement. Polygon labels enclose the visually affected area; no CEJ/bone-crest geometry, no apex needed. [verified] |
| Metrics (BL class)    | Precision 0.84, Recall 0.93, **F1 0.88** [verified]    |
| Metrics (overall)     | mAP@0.5 = 0.30, mAP@0.5:0.95 = 0.10 (low-frequency classes drag mean) [verified] |
| License               | MDPI is CC-BY 4.0 by default [verified for paper; data restricted] |

**Critical finding:** Segmentation-as-detection. Bone loss is treated as a
binary "affected region exists" finding rather than a quantitative
measurement. F1 0.88 on bone-loss class is the **strongest BW BL number
in the open literature.**

**Fit-to-our-use-case:** A **YOLOv8-seg head for the "bone-loss-present"
finding** is a low-effort, high-yield product feature (binary alarm bell)
even when the mm measurement is uncertain. Worth a parallel head in the
v0 BW pipeline alongside any landmark approach.

---

### 2.5 Wimalasiri et al. 2025 — DenPAR v3 / arxiv 2506.20522 [verified]

| Property              | Value                                                  |
|-----------------------|--------------------------------------------------------|
| Citation              | Wimalasiri et al., arxiv 2506.20522, 2025 [verified]   |
| Country               | [unverified — SriLanka group / Simula collaboration]   |
| Modality              | **Periapical only** [verified]                         |
| Code                  | [unverified — not located in this pass]                |
| Data                  | DenPAR (Zenodo 13998619), 1,000 IOPAs, CC-BY [verified — Nature Sci Data 2025] |
| Architecture          | YOLOv8 (tooth detection) + Keypoint R-CNN (landmarks: CEJ, bone, apex per tooth) [verified] |
| **Apex handling**     | **Apex required as denominator** of % bone-loss formula. Frame must contain the apex. [verified] |
| Metrics               | ICC up to 0.80 (severity), 87% accuracy (pattern classification) [verified] |
| License               | arxiv (likely CC-BY), DenPAR CC-BY [verified for DenPAR] |

**Critical finding:** **Cannot be used on BW.** Apex denominator is the
load-bearing geometric assumption.

**Fit-to-our-use-case:** Keep for PA mode only. Current implementation is
correct for PA. The mismatch on BW is structural, not a model-quality issue.

---

### 2.6 Banks, Thengane et al. 2025 — Stage-agnostic keypoint detection [verified]

| Property              | Value                                                  |
|-----------------------|--------------------------------------------------------|
| Citation              | Banks, Thengane et al., arxiv 2503.13477, 2025 [verified] |
| Country               | UK (Surrey, KCL) + Peru + India [verified]             |
| Modality              | **Periapical only in evaluation** (architecture transferable) [verified] |
| Code                  | **MIT license** at `Banksylel/Bone-Loss-Keypoint-Detection-Code` [verified] |
| Weights               | Google Drive (base + fine-tuned) [verified]            |
| Data                  | **CC-BY on Zenodo 17272200**, 192 PAs, 582 teeth, 3,520 keypoints [verified] |
| Architecture          | 4-way comparison: **YOLOv8-Pose, HRNet, DeepPose (ResNet50+RLE), RTMPose-tiny** [verified] |
| Post-processing       | **Heuristic tooth-segmentation alignment**: predicted keypoints aligned to tooth boundaries via auxiliary segmentation, partitioned mesial/distal, edge-association filtering [verified] |
| **Apex handling**     | Uses **root level (RL)** not apex as denominator: `PBL = ||CEJ−BL||² / ||CEJ−RL||²`. Still requires RL to be visible. [verified] |
| Metrics               | YOLOv8-Pose: PRCK⁰·⁵ = 0.912; HRNet: PRCK⁰·⁰⁵ = 0.375 (fine precision) [verified] |
| Annotation discipline | **Stage-agnostic** — landmarks labeled regardless of disease presence (avoids the "no CEJ labeled because tooth healthy" trap that contaminates many datasets) [verified] |

**Critical findings:**
1. **First fully-open keypoint-detection pipeline.** Code + weights + data + paper
   under permissive licenses. Highest-fidelity reference implementation
   available for the keypoint architecture family.
2. **4-architecture benchmark.** Their finding — YOLOv8-Pose wins coarse,
   HRNet wins fine — should inform our architecture pick for ANY landmark task.
3. **Stage-agnostic annotation** is a discipline rule, not an architecture
   choice. Translatable to BW: label CEJ + bone-crest on every tooth in the
   BW, regardless of whether that tooth has bone loss.
4. **Heuristic post-processing** is the antidote to "CEJ-collapse" — by
   aligning keypoints to tooth-segmentation boundaries, the model can't
   produce two CEJ points on the same horizontal line in the middle of the tooth.

**Fit-to-our-use-case:** **The single highest-ROI reference implementation
to study.** PA-only in their data but the architecture comparison +
post-processing trick transfer directly to BW. Apex requirement is
weaker than Wimalasiri (RL not apex) but still PA-shape. The annotation
discipline and the heatmap-vs-coordinate architecture comparison are the
load-bearing transfers.

---

### 2.7 HUNT4 / AI-Dentify (Pérez de Frutos et al. 2024) — confirmed NOT bone-loss labeled [verified]

| Property              | Value                                                  |
|-----------------------|--------------------------------------------------------|
| Citation              | Pérez de Frutos et al., *BMC Oral Health*, 2024 [verified] |
| Modality              | **Bitewing only** [verified]                           |
| Data                  | **13,887 BWs from HUNT4 Oral Health Study (Norway)** [verified] |
| Annotations           | **Proximal caries only (5 classes)** [verified] — **no bone-loss labels** |
| License               | Paper CC-BY 4.0; **dataset is HUNT4-gated**, not on public hosting [verified — no download URL in paper] |
| Architecture          | RetinaNet (ResNet50), YOLOv5-M, EfficientDet (D0/D1) compared [verified] |
| Code                  | [unverified — needs probe in supplementary; SINTEF Digital group] |

**Critical finding:** **HUNT4 BW set is NOT bone-loss labeled.** The 13,887
images are caries-only. Bone-loss researchers cannot piggyback on this
dataset for our task. If access were granted (HUNT4 requires Norwegian
Biobank application — not on the open-access track), re-labeling for
bone loss would be the dataset bottleneck, not the imaging.

**Fit-to-our-use-case:** **Reject.** Not bone-loss labeled, not open access.

---

### 2.8 Other BW-mentioning papers (lower-ROI quick scan)

- **Jundaeng/Chamchong/Nithikathkul 2025** (PMC11797906, Thailand): YOLOv8 3-model
  conglomerate (teeth + CEJ + bone). **Panoramic, not BW.** F1 0.90 for CEJ
  detection. CC-BY 4.0. Data restricted. Architecture transferable, modality is not.
- **Ameli et al. 2024** (Frontiers, 10.3389/fdmed.2024.1479380, Canada): U-Net
  segmentation of bone-loss-area polygon + YOLOv9 for apex. **Periapical only.**
  Data restricted. CC-BY. Polygon-based bone-loss area segmentation is a
  load-bearing idea (resembles v3's bone polyline approach).
- **Lee et al. 2021** (PMC9026777, USA): three U-Net segmentation networks
  (bone area, tooth, CEJ line). **Periapical only.** 693 PAs. Code "by request."
  Dice >0.91. Predecessor to the Lee/Columbia 2025 paper (§2.2) — same lead
  author, similar approach scaled up.
- **Tai-Jung Lin et al. 2024** (PMC11312231, Taiwan): YOLOv8 + Mask R-CNN +
  CLAHE preprocessing. **Periapical.** CC-BY 4.0. Data restricted (Chang Gung
  Memorial Hospital IRB). Kernel-based overlay between tooth and bone masks
  for CEJ/ALC localization.
- **Sundsdal et al. (cited as 2022)**, "two-stage deep learning architecture
  for radiographic staging" (BMC Oral Health, s12903-022-02119-z): two-stage
  detection + classification, **bitewing among modalities**. [needs probe]
- **Alqaderi / Tufts Dental AI Lab (medRxiv 2023)**, *Deep Learning Approach
  to Measure Alveolar Bone Loss After COVID-19* (medrxiv 2023.11.20.23298788):
  searched, but webfetch 403'd; the 97% F1 claim referenced in the user prompt
  is **[unverified]** — needs manual access. The Tufts AI lab's other work
  (Ahmad/Saleh/Alharbi/Jeong/Zavras/Alqaderi 2024 on routine blood tests as
  perio predictors) is unrelated to imaging.

---

### 2.9 Commercial groups (Overjet, Pearl, VideaHealth, Denti.AI) — no useful open artifacts

- **Overjet** ([dental Assist FDA-cleared 2023]): publishes "≥3 mm CEJ→bone =
  RBL present" as their threshold. **No peer-reviewed paper located** that
  details architecture. No code release. [verified — narrative review
  PMC12775797 notes lack of peer-reviewed publications for most commercial
  tools]
- **Pearl, VideaHealth**: same shape — FDA clearance, no published architecture.
- **Denti.AI**: covered as AlGhaihab 2025 (§2.1) — the only one of the commercial
  groups with a serious published evaluation.

**Fit-to-our-use-case:** Trade press references, not engineering artifacts.
Useful only for confirming the **mm-threshold convention** (≥2 mm Denti.AI,
≥3 mm Overjet) and the **FDA-cleared shipping benchmark.**

---

## 3. Part 2 — Alternative architectures survey

### 3.1 CEJ as polyline segmentation (not keypoints)

**Reference:** Lee et al. 2021 (PMC9026777) annotate CEJ as a **polyline** on
every PA image. DeepLab variants and U-Net segmentation networks predict the
polyline pixel-wise; the polynomial fit then converts pixel mask → smooth curve.
Dice >0.91 reported.

**Why it dodges CEJ-collapse:** Keypoint regression heads place two discrete
points per tooth side, which can collapse to the same location under low
contrast (the failure mode triggering this research). Polyline segmentation
predicts a **continuous mask** — the network outputs every pixel that lies on
the CEJ, then post-processing fits a curve. There's no "two points on top of
each other" because there are no discrete points.

**v3 already does this for bone.** The bone polyline head in our current
pipeline is the same shape. Symmetry: add a CEJ polyline head.

**Fit:** **Highest-ROI architectural change for v0 BW.** Two parallel
segmentation heads (CEJ polyline + bone polyline) on the same backbone,
distance computed perpendicular between them within each tooth bbox.
Apex never enters the math.

---

### 3.2 Joint CEJ + bone-crest segmentation (segmentation-only pipeline)

**Reference:** Jundaeng et al. 2025 (PMC11797906, panoramic): two YOLOv8
segmentation models — one for teeth + CEJ landmarks, one for bone level
+ CEJ. F1 0.90 reported. **Pipeline does not use keypoints at all.**

**Why it works for BW:** Same logic as §3.1, but a single model predicts both
the CEJ region and the bone-crest region jointly. Distance derived from
segmentation overlap.

**Fit:** Solid, but the dual-head version (§3.1) gives finer control and
matches the v3 polyline shape that already exists.

---

### 3.3 Heatmap regression with HRNet / ViTPose / RTMPose

**Reference:** Banks et al. 2025 (arxiv 2503.13477) compare four pose-estimation
backbones for dental keypoints. HRNet wins fine-grained precision (PRCK⁰·⁰⁵
= 0.375); YOLOv8-Pose wins coarse (PRCK⁰·⁵ = 0.912); RTMPose-tiny weakest.

**Why HRNet is interesting for our case:** HRNet maintains high-resolution
feature maps throughout the network without downsampling+upsampling. Fine
spatial detail at the original resolution is exactly what CEJ localization
needs. Wimalasiri's Keypoint R-CNN uses a ResNet-50 FPN — coarse final
features, then upsampled. HRNet skips that lossy step.

**ViTPose** (transformer-based heatmap regression) is widely used outside
dental. No dental BW reference found in this pass [needs probe].

**Fit:** **Drop-in replacement for the Keypoint R-CNN backbone** if we want
to keep the keypoint formulation. Even on PA where the % formula needs
apex, HRNet should outperform the FPN backbone on CEJ-collapse cases.

**Open code:** HRNet has `HRNet/HRNet-Human-Pose-Estimation` (PyTorch); MMPose
(`open-mmlab/mmpose`) wraps HRNet, RTMPose, ViTPose under one config-driven
training framework. Apache 2.0.

---

### 3.4 Two-stage: tooth detection → per-crop keypoint head

**Reference:** Almost every published dental-keypoint pipeline (Wimalasiri,
Lee, Banks) uses this two-stage shape. The Banks 2025 paper makes it explicit:
a separate tooth-segmentation network produces ROIs; the keypoint head trains
on **cropped, oriented tooth images** rather than the full radiograph.

**Why it helps for BW:** The 6-8 teeth + 2 arches problem becomes 6-8
independent single-tooth crops. Each crop is the geometry Wimalasiri trained
for (one tooth, roughly upright, crown at top, root below). The keypoint
head doesn't need to learn arch-position; the cropper has done that.

**Fit:** **Already implicit in v3's design.** Worth verifying that the
crop step is actually robust on BW (the "multi-tooth bbox blob" failure
mode reported in the trigger suggests the tooth detector itself is the
weak link, not the downstream keypoint head). If tooth detection is fixed,
the keypoint head may need less surgery than feared.

---

### 3.5 Anchor-free landmark detection (CenterNet, DEKR)

**Reference:** CenterNet (Zhou et al., `xingyizhou/CenterNet`, BSD-3) and
DEKR (`HRNet/DEKR`, MIT) predict heatmaps + offsets without anchors. CHaRNet
(arxiv 2501.13073) applies conditioned heatmap regression to dental landmarks
in 3D intraoral scans.

**Why it might fit BW:** No predefined anchors means the model doesn't have
to learn "skip these 7 anchors per tooth and pick the right one." More
parameter-efficient. Works well at small-object scale (a CEJ point in a
2048x1536 BW is ~2-3 px).

**Open question:** None of these have been benchmarked on BW bone-loss
landmarks. [needs probe — high signal/effort ratio]

---

### 3.6 Distance-from-CEJ measurement without apex (mm with anatomical prior)

**Reference:** AlGhaihab et al. 2025 (§2.1) on BW: "RBL = CEJ→bone-crest
distance ≥ 2 mm." Overjet's published clinical threshold: ≥3 mm. The 2018
AAP/EFP classification stages by absolute mm of attachment loss.

**Why it works:** No "0% bone loss" reference is needed. The clinical
threshold is **already in millimeters** by AAP/EFP convention. The architectural
requirement collapses to: predict CEJ pixel coords, predict bone-crest pixel
coords, compute Euclidean distance, convert to mm via image calibration.

**mm calibration on BW:** Two paths:
1. **DICOM PixelSpacing tag** if present (often is for digital sensors —
   needs verification per vendor).
2. **Tooth-type anatomical prior**: known mean MD width of each tooth (e.g.,
   maxillary first molar ≈ 10.5 mm MD). Identify tooth via FDI labeling,
   look up MD width, calibrate pixel-mm ratio from segmentation. The
   AAP threshold then converts to a pixel threshold per tooth.

**Fit:** **Required architectural piece** if we ship a BW pipeline. Either
DICOM PixelSpacing or tooth-prior calibration must be added; the % formula
cannot survive transplant to BW.

---

### 3.7 Bone-level segmentation alone (no CEJ needed)

**Reference:** Hypothetical / partially explored. v3 already predicts the
bone polyline. If a per-tooth-type anatomical prior fixed where the CEJ
"should" be relative to the tooth bbox (e.g., "CEJ sits at the bbox top
edge for mandibular molars"), the CEJ network becomes unnecessary —
just compare the bone polyline to the bbox-relative CEJ prior.

**Why this is fragile:** Tooth bboxes drift; restorations move the visual
"bbox top" away from the anatomical CEJ; supra-erupted teeth (common in
opposing missing teeth) put the CEJ much higher than the prior expects.
**Not recommended** unless backed by per-tooth segmentation that explicitly
identifies crown vs root region.

**Fit:** Reject as the only approach. Worth keeping as a sanity check
sidecar — if bone-only path strongly disagrees with CEJ-bone-distance path,
flag the tooth for human review.

---

### 3.8 Foundation models / SAM / SAM2 for landmark or boundary detection

**SAM-based dental work confirmed in 2024-2025:**

- **Tooth-ASAM** (Nature Sci Reports, s41598-025-96301-2): SAM adapter for
  tooth segmentation, multimodal. [unverified — modality includes BW? needs probe]
- **SAM2 dental adaptation** (PMC11675754): SAM2 + adapter modules for tooth
  segmentation on **panoramic** images, with ScConv and gated attention.
  Outperforms vanilla U-Net on UFBA-UESC dataset.
- **finetune-SAM** (`mazurowski-lab/finetune-SAM`, **Apache 2.0**): generic
  medical-image SAM fine-tuning framework. Supports binary + multi-class,
  prompt-based (point/box/hybrid), parameter-efficient (LoRA, adapters).
  Tested on 17 medical-radiology datasets. **Not dental-specific but
  directly adaptable.**
- **SAM2-UNet** (`WZH0120/SAM2-UNet`, Visual Intelligence 2026, license
  [unverified]): SAM2 as a U-Net encoder. Medical segmentation results
  on multiple datasets.
- **MedSAM2** (arxiv 2504.03600): SAM2 + memory attention for 3D medical
  imaging. CT/MRI-focused; not directly applicable to 2D BW.

**Fit:** Promising **for the segmentation paths (§3.1, §3.2, §3.7)** —
a SAM-finetuned encoder could replace the DeepLab v3+ backbone with
likely better sample efficiency. **Not promising for keypoints** —
SAM is a segmentation foundation model, not a landmark one.

---

### 3.9 Multimodal vision-language models for landmark identification

**State of the art (mid-2026):**

- Gemini / Claude / GPT-4o-class models can describe radiographic findings
  textually with moderate accuracy (peer-reviewed dental literature on
  this is sparse; the *npj Digital Medicine* HC-Net+ paper trains a
  task-specific model rather than using a general VLM).
- **No published dental VLM landmark-localization benchmark** found in
  this pass.
- Domain-specific medical VLMs (LLaVA-Med, Med-PaLM, RadFM) exist but
  none target intraoral radiographs as a primary task.

**Fit:** **Reject for v0.** Localization precision required for mm-distance
measurement is below current VLM resolution. Worth reconsidering when a
dental-specific VLM ships (likely 2027-2028 timeframe).

---

### 3.10 Self-supervised pretraining on unlabeled radiographs

**Reference:** GeoSapiens (arxiv 2507.04710) uses Sapiens — a MAE-pretrained
human-centric ViT — for dental landmark detection with **3-patient few-shot
training** + LoRA. 93.4% SDR@2mm on dental CBCT landmarks. MIT license,
full code at `xmed-lab/GeoSapiens`.

The Sapiens foundation model (Meta AI) was pretrained on 300M human images
via Masked Autoencoder self-supervision. Few-shot transfer to dental
landmarks with geometric priors (perpendicularity / parallelism loss)
beats from-scratch by a wide margin.

**Why this is interesting:**
- Bootstraps from a foundation model rather than training from scratch.
- Geometric loss enforces anatomical constraints — directly useful for
  CEJ-bone-crest "should be roughly parallel" priors.
- Works with very few labeled images (3 patients, 347 images for their task).

**Fit:** **High exploratory value** for BW landmark detection. The few-shot
property is the load-bearing claim — if it transfers, hand-labeling 50-100
BWs (vs. acquiring a 1000+-image dataset) becomes viable. Modality jump
from CBCT to BW radiograph is non-trivial but the framework itself is
modality-agnostic.

**Caveat:** CBCT slices are 3D-rendered, BWs are projection radiographs.
The pretraining domain (human bodies) may transfer better to one than the
other; empirical question.

---

## 4. Comparison tables

### 4.1 BW literature

| Paper                  | Modality | Open? | Code | Data           | Architecture                  | BW BL approach               | License    |
|------------------------|----------|-------|------|----------------|-------------------------------|------------------------------|------------|
| AlGhaihab 2025         | BW + PA  | No    | No   | Restricted     | Faster RCNN + ResNet + FPN-Res| **mm CEJ→bone-crest ≥2mm**   | Paper CC-BY |
| Lee 2025               | BW       | No    | No   | Restricted 550 | 5-net (FR-CNN + DeepLab×3)    | **Polynomial across arch**   | CC-BY 4.0  |
| Erturk 2024            | BW       | No    | No   | Restricted 1752| YOLOv8m-cls + Eigen-CAM       | **Classifier, no geometry**  | "Exclusive licence" — [unverified] |
| Akarsu 2026            | BW       | No    | No   | Restricted 1197| YOLOv8x-seg, 8 classes        | **Polygon affected-region**  | Paper CC-BY |
| Wimalasiri 2025        | PA       | Partial| [?] | DenPAR CC-BY   | YOLOv8 + Keypoint R-CNN       | **% root length (apex req)** | DenPAR CC-BY |
| **Banks 2025**         | PA       | **Yes** | **MIT** | **CC-BY 192 PAs** | **YOLOv8/HRNet/DeepPose/RTMPose** | **% root level (no apex)** | **All open** |
| HUNT4 AI-Dentify       | BW       | No    | [?]  | HUNT4-gated    | Various (caries only)         | **Not BL-labeled**           | CC-BY paper |

### 4.2 Alternative architectures

| Approach                              | Code available | Dental BW reference? | Apex needed? | Fit (1-5) |
|---------------------------------------|----------------|---------------------|--------------|-----------|
| CEJ polyline segmentation             | Standard tooling (Segformer, DeepLab) | Yes (Lee 2021)      | No           | **5**     |
| Joint CEJ + bone-crest segmentation   | Standard       | Yes (Jundaeng 2025) | No           | **5**     |
| Heatmap regression (HRNet/RTMPose)    | MMPose Apache 2.0 | Yes (Banks 2025)   | Optional     | **4**     |
| Two-stage tooth→keypoint              | Banks 2025 MIT | Yes                 | Optional     | **4**     |
| Anchor-free (CenterNet/DEKR)          | BSD-3 / MIT    | Partial (3D CHaRNet)| Optional     | **3**     |
| mm distance with anatomical prior     | Standard       | Yes (AlGhaihab)     | **No**       | **5**     |
| Bone-only seg + bbox-relative CEJ prior | Standard     | No                  | No           | **2**     |
| SAM2 fine-tune (`finetune-SAM`)       | Apache 2.0     | Adjacent (panoramic)| No           | **3**     |
| Multimodal VLM                        | Closed (Anthropic/OpenAI/Google) | No  | N/A          | **1**     |
| Self-supervised foundation (Sapiens)  | MIT (GeoSapiens) | Adjacent (CBCT)   | No           | **3**     |

---

## 5. Insights — how the literature handles "no apex on BW"

Three families of solutions, in increasing architectural complexity:

**Family A — Change the unit (lowest complexity).**
mm CEJ→bone-crest distance, threshold by AAP/EFP convention (≥2 mm = mild,
≥4 mm = moderate, etc.). Architecture stays keypoint-based; only the math
at the head changes. **AlGhaihab/Denti.AI 2025 path.** Requires DICOM
PixelSpacing OR tooth-type prior for mm calibration.

**Family B — Change the reference (medium complexity).**
Use a multi-tooth arch-fit polynomial as the implicit reference axis,
replacing the apex denominator. Lee/Columbia 2025 path. Requires
polynomial fitting + per-arch segmentation. More robust against single-tooth
detection failures because the arch curve smooths them out.

**Family C — Drop the geometry entirely (highest abstraction).**
Image-level classifier emits AAP stage directly. Erturk 2024 path.
Sacrifices granularity and mm numbers; gains simplicity and robustness.
Worth running in parallel as a defensive sidecar.

**Hybrid (recommended for our use case):**
- Family A as the primary measurement head (mm CEJ→bone-crest per tooth).
- Family B's polynomial smoothing as a per-arch consistency check
  (flag teeth whose bone-crest deviates >X mm from the arch curve as
  candidate localized defects).
- Family C as a sidecar classifier — if the geometric pipeline says
  "incomputable," still emit an AAP stage label.

---

## 6. Recommendation

### 6.1 Modality split

**Two pipelines, shared infrastructure.**

- **PA pipeline:** Keep Wimalasiri Keypoint R-CNN with % root length.
  Apex denominator is what makes the published % numbers defensible. No
  reason to change. DenPAR v3 is the canonical training set.
- **BW pipeline:** **Distinct architecture.** Do not attempt to run the
  PA model on BW; the structural mismatch is real.

### 6.2 BW architecture

**Primary recommendation:** Dual polyline segmentation + tooth bbox + mm
calibration. Specifically:

1. **Backbone:** Shared encoder (HRNet-W48 or Segformer-B2). Choose by
   what's already in the v3 codebase.
2. **Head 1:** Per-tooth bbox detector (YOLOv8 or Faster R-CNN — match v3).
3. **Head 2:** CEJ polyline segmentation (DeepLab v3+ or Segformer).
   Trained with the same shape as v3's bone polyline head — drop-in symmetric.
4. **Head 3:** Bone-crest polyline segmentation (already exists in v3).
5. **Postprocessing:** Per-tooth, compute perpendicular distance between
   the two polylines within the bbox. Calibrate via DICOM PixelSpacing
   when present; fall back to tooth-type anatomical prior.
6. **Sidecar:** YOLOv8m-cls AAP stage classifier (Erturk 2024 shape)
   running in parallel. When primary pipeline emits "incomputable," sidecar
   emits the stage label.

**Why this shape:** Maximally reuses v3's existing bone polyline head.
Avoids the CEJ-collapse keypoint failure mode (segmentation, not points).
Avoids the apex requirement (Family A unit change). Adds Family C
robustness via the sidecar.

**Alternative recommendation (more experimental):** Replace the keypoint
head with **HRNet heatmap regression** per Banks 2025's findings. Same
modality, same data shape, lower CEJ-collapse risk than Faster-RCNN's FPN
backbone. Add the heuristic tooth-segmentation post-processing from
Banks 2025 (MIT-licensed reference code available).

### 6.3 Training data path (constraint: no hand-labeling per user instruction)

**No fully-open BW bone-loss dataset exists** that is permissively licensed
+ downloadable today + bone-loss-labeled. All four BW-published groups
released only the paper, not the data. Practical paths:

1. **Adapt Banks 2025 Zenodo data (192 PAs, CC-BY)** as a pretraining
   substrate; transfer learning to BW with a small unlabeled-BW
   self-supervised pretraining step.
2. **Probe Roboflow Universe** for any "bitewing bone loss" datasets
   recently uploaded under permissive license — [needs probe] — there
   were 730-1197 BW Roboflow datasets located for caries/multi-class
   that may have BL labels.
3. **Synthetic augmentation:** the 2026 *Diffusion-generated synthetic
   dental radiographs* line (PMC12565194) suggests synthetic BWs with
   labeled bone loss are feasible. [needs probe — license and quality]
4. **Wait for HUNT4 BW set extension** — the 13,887 caries-labeled BWs
   exist; if HUNT4 ever ships a bone-loss-labeled version, that becomes
   the canonical dataset overnight. **[needs probe — track HUNT4 / SINTEF
   Digital releases.]**

### 6.4 Architecture for PA mode — should it change?

**No.** Wimalasiri Keypoint R-CNN works on the modality it was designed
for. The known CEJ-collapse failures are a downstream issue worth fixing
(per Banks 2025: switch to HRNet heatmap regression + tooth-segmentation
post-processing) but those are within-modality refinements, not a
modality change.

The shared infrastructure across PA and BW pipelines is:
- Tooth detection backbone
- Bone polyline head (v3 already has it; BW reuses)
- mm calibration utilities (BW path; PA can also use as a sanity check)

The non-shared infrastructure:
- PA: apex keypoint head + % formula head
- BW: CEJ polyline head + mm formula head + AAP sidecar classifier

---

## 7. Open follow-ups

1. **[Needs probe] Roboflow Universe BW bone-loss datasets** — confirm
   whether the Akarsu 2026 (1197 BWs) or Lee 2025 (550 BWs) datasets are
   mirrored on Roboflow under any reasonable license.
2. **[Needs probe] Alqaderi/Tufts medRxiv 2023.11.20.23298788** — confirm
   whether the 97% F1 BW bone-loss claim referenced in the user prompt is
   verified; webfetch 403'd. Direct PDF download recommended.
3. **[Needs probe] Wimalasiri code release** — could not locate a public
   code repo for arxiv 2506.20522. Author Github / Simula research page.
4. **[Needs probe] HUNT4 access path** — Norwegian Biobank application
   process; long lead time; bone-loss labeling status unknown.
5. **[Needs probe] HRNet vs FPN ablation on DenPAR v3** — quick
   experiment, swap Keypoint R-CNN's FPN backbone for HRNet, measure
   CEJ-collapse rate on the eval set. Banks 2025 predicts win, but
   never tested on DenPAR directly.
6. **[Needs probe] Sapiens / GeoSapiens few-shot transfer to BW** —
   high-signal experiment, 5-day budget. Worth doing before committing
   to the segmentation pipeline.
7. **[Needs probe] DICOM PixelSpacing tag coverage** — verify across
   the BW images already on disk in `dental-rad-cli/data/` (or wherever
   BWs live) whether mm calibration is available without anatomical prior.
8. **[Needs probe] Synthetic BW diffusion (PMC12565194)** — license +
   realism for periodontal-specific findings. Generative augmentation
   path for sparse BW data.

---

## 8. Sources

### Bitewing-specific papers
- AlGhaihab et al. 2025, *Diagnostics* 15(5):576 — https://www.mdpi.com/2075-4418/15/5/576 — PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11899607/
- Lee et al. 2025, *BMC Oral Health* — https://link.springer.com/article/10.1186/s12903-025-05677-0 — PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11872301/
- Erturk, Öziç, Tassoker 2024, *J Imaging Inform Med* — https://link.springer.com/article/10.1007/s10278-024-01218-3 — PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11811320/
- Akarsu et al. 2026, *Diagnostics* 16(2):322 — https://www.mdpi.com/2075-4418/16/2/322
- AlGhaihab pilot (ScienceDirect 2024) — https://www.sciencedirect.com/science/article/pii/S2212440324008034
- Mertoglu/Saglam "Enhancing Periodontal Bone Loss Diagnosis" — https://www.mdpi.com/2076-3417/15/12/6832
- Alqaderi et al. (medRxiv) — https://www.medrxiv.org/content/10.1101/2023.11.20.23298788v1.full *[unverified — webfetch 403'd]*

### Periapical / panoramic comparators
- Wimalasiri et al. 2025, arxiv 2506.20522 — https://arxiv.org/abs/2506.20522
- DenPAR dataset, Zenodo 13998619 — https://zenodo.org/records/13998619 (also Nature Sci Data 2025: https://www.nature.com/articles/s41597-025-05906-9)
- Banks/Thengane et al. 2025, arxiv 2503.13477 — https://arxiv.org/html/2503.13477v3 — code: https://github.com/Banksylel/Bone-Loss-Keypoint-Detection-Code — data: https://zenodo.org/records/17272200
- Lee et al. 2021, *J Periodontal Implant Sci* — https://pmc.ncbi.nlm.nih.gov/articles/PMC9026777/
- Jundaeng/Chamchong/Nithikathkul 2025 — https://www.frontiersin.org/journals/dental-medicine/articles/10.3389/fdmed.2024.1509361/full — PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11797906/
- Ameli et al. 2024, *Frontiers* — https://www.frontiersin.org/journals/dental-medicine/articles/10.3389/fdmed.2024.1479380/full
- Tai-Jung Lin et al. 2024 — https://pmc.ncbi.nlm.nih.gov/articles/PMC11312231/
- Pérez de Frutos et al. 2024 (HUNT4 AI-Dentify) — https://bmcoralhealth.biomedcentral.com/articles/10.1186/s12903-024-04120-0 — arxiv: https://arxiv.org/abs/2310.00354

### Alternative architectures
- GeoSapiens / Sapiens-based dental landmarks — https://arxiv.org/html/2507.04710 — code: https://github.com/xmed-lab/GeoSapiens
- finetune-SAM (Mazurowski Lab) — https://github.com/mazurowski-lab/finetune-SAM
- SAM2 dental adaptation — https://pmc.ncbi.nlm.nih.gov/articles/PMC11675754/
- Tooth-ASAM (SAM adapter) — https://www.nature.com/articles/s41598-025-96301-2
- SAM2-UNet — https://github.com/WZH0120/SAM2-UNet — https://link.springer.com/article/10.1007/s44267-025-00106-w
- MedSAM2 (3D) — https://arxiv.org/abs/2504.03600
- MMPose framework (HRNet, RTMPose, ViTPose) — https://github.com/open-mmlab/mmpose
- HRNet pose estimation — https://github.com/HRNet/HRNet-Human-Pose-Estimation
- CenterNet — https://github.com/xingyizhou/CenterNet
- CHaRNet (conditioned heatmap regression, 3D dental) — https://arxiv.org/html/2501.13073v3

### Reviews + dataset catalogs
- FDA-Approved AI Solutions in Dental Imaging review — https://pmc.ncbi.nlm.nih.gov/articles/PMC12775797/
- Publicly Available Dental Datasets review — https://pmc.ncbi.nlm.nih.gov/articles/PMC11633071/
- Detection of perio bone loss systematic review + meta — https://academic.oup.com/dmfr/article/54/2/89/7917334
- ITU/sergiouribe AI Dental Datasets List — https://github.com/sergiouribe/dental_datasets_itu
- BRAR multimodal dataset (panoramic, periodontal) — https://www.nature.com/articles/s41597-025-06400-y

### Synthetic data / pretraining
- Diffusion synthetic dental radiographs 2026 — https://pmc.ncbi.nlm.nih.gov/articles/PMC12565194/
- Deep Learning in Dentistry systematic review 2025 — https://www.medrxiv.org/content/10.1101/2025.10.01.25337082v1.full.pdf

### Clinical reference (mm thresholds)
- Evaluation of alveolar crest bone loss via premolar BW — https://pmc.ncbi.nlm.nih.gov/articles/PMC4216398/
- Jacobs 2024, *Periodontology 2000* — https://onlinelibrary.wiley.com/doi/10.1111/prd.12580
- Overjet AI Overview (Apteryx/Planet DDS) — https://apteryximaging.planetdds.com/hc/en-us/articles/11226098680091-Overjet-AI-Overview
