# Perio / Bone-Loss Deep Dive — Academic Survey

Date: 2026-05-12
Scope: open-source dental radiograph CLI, perception layer for tooth detection +
CEJ / bone-crest / apex keypoints + tooth & bone segmentation + per-site % bone
loss + horizontal-vs-angular pattern classification.
Modality of interest: intraoral periapical (PA) and bitewing (BW). Panoramic
out of scope.
Confidence convention: `[verified]` = page contents read this session;
`[needs verification]` = inferred from search abstracts; `[blocked]` = could
not retrieve (403 / restricted).

---

## 1. TL;DR

**Problem statement.** The current pipeline trains on **DenPAR v3** (Zenodo
16645076, CC-BY 4.0, 1,000 IOPA) following Wimalasiri et al. 2026 (arXiv
2506.20522, Sci Rep). DenPAR's CEJ keypoint annotation is shipped as a flat
2-D point list per image with **no per-tooth grouping**; the trainer assigns
points to teeth via bbox containment + nearest-center, producing clean
2-keypoint supervision for only ~42% of teeth. The CEJ head consequently
collapses (mesial ≡ distal) on 35–60% of teeth at inference. This is a
**monoculture-on-one-paper** failure mode: the pipeline inherits the upstream
dataset's annotation schema gap.

**Primary recommendation — augment, don't replace.** Use **Banks et al. 2025
("perio-KPT")** as a **second supervisory signal** for the CEJ keypoint head
alongside DenPAR. perio-KPT (Zenodo 14711842 v1.0 / 17272200 v2.0) ships
**192 IOPA + 3,588 panoramic auxiliary images** with CEJ-mesial / CEJ-distal
keypoints already grouped per-tooth in **YOLOv8-pose format with visibility
flags**, plus a fully open MIT-licensed training repo
(github.com/Banksylel/Bone-Loss-Keypoint-Detection-Code) that targets the
exact head we are struggling with. Annotation methodology is "stage-agnostic"
(labels landmarks regardless of disease presence) which is the discipline our
DenPAR adapter lacks. **However:** as of 2026-05-12 the Zenodo files are
**restricted-access** (login + institutional affiliation request, even on v1).
The arXiv paper says license is CC-BY but Zenodo records show CC-BY-NC-SA 2.0
Generic — an internal inconsistency. License resolution + access request is
the unblocking step. Personal-scope + non-commercial-research use is
plausibly compatible; redistribution likely is not.

**Fallback recommendation — CEJ adapter fix on DenPAR alone.** If perio-KPT
access cannot be obtained, the cleanest fix without a new dataset is to
**rewrite the CEJ assignment adapter** to enforce per-tooth pairing: cluster
DenPAR's flat CEJ point list per-image by tooth bbox, reject teeth that
don't receive exactly 2 CEJ points (≈58% of teeth) rather than padding with
nearest-center fallbacks, and either (a) train the keypoint head only on
the clean ≈42% subset with `visibility=0` for the unclean ones, or (b)
derive CEJ-mesial / CEJ-distal from the tooth segmentation mask boundary
intersected with a learned enamel-cement boundary. This is a per-tooth
supervision-quality fix, not a dataset-quality fix.

**Tertiary signals.** Lin et al. 2024 (MDPI Diagnostics, Taiwan, 140 PA,
Roboflow CEJ + ALC polygon annotation, RMSE < 0.09) and Lee et al. 2025
(Columbia, 550 BW, ICC 0.98 inter-rater, CEJ+ABCL boxes) demonstrate the
*signal* that per-tooth CEJ/ABCL annotation works clinically — but **neither
dataset is public**. Both are read as evidence that the **annotation shape**
matters more than dataset size.

**Datasets that look big but are dead-ends:** Chen 2023 (8,000 PA — no
public release), HUNT4 (13,887 BW — Norwegian ethics committee approval
required, not redistributable), Tufts Dental Database (panoramic-only),
Alqaderi 2026 Tufts medRxiv (1,063 PA+BW — single-institution, dataset not
released).

**Datasets that are open but wrong-shape:** PRAD-10K (MICCAI 2025, 10K PA)
is open by application but annotates *9 anatomical structures with
pixel-level segmentation* — no CEJ keypoints or bone-level annotation.
Usable as a tooth-segmentation pretraining corpus, not as a perio signal.

---

## 2. Constraints recap

| Constraint | Threshold |
|---|---|
| License | CC-BY 4.0 or more permissive for **redistribution**; CC-BY-NC acceptable for **training-only** in a personal-scope project |
| Modality | Intraoral PA or BW only; panoramic excluded |
| Size | ≥500 images preferred (smaller acceptable as auxiliary signal) |
| Accessibility | Downloadable today, not IRB-pending or institutional-only |
| Annotation type | Per-tooth keypoints (CEJ, apex, bone crest), segmentation masks, or pixel-precise bone-level lines |
| Use case | Train a perception model that predicts CEJ-m / CEJ-d / apex / bone-crest-m / bone-crest-d per tooth |

Personal-scope means: no FDA path, no commercial redistribution of trained
weights, no patient-facing decision support. Training on
research-use-only data is acceptable provided the trained model and its
*outputs* don't constitute redistribution of the underlying data.
(`[verified]` interpretation — confirm independently.)

---

## 3. Per-candidate evaluation (ranked)

### 3.1 ⭐ Banks et al. 2025 — "Periodontal Bone Loss Analysis via Keypoint Detection With Heuristic Post-Processing" (perio-KPT)

| Field | Value |
|---|---|
| Paper | arXiv 2503.13477 v3 (2025-10-20); submitted 2025-03-05 |
| Authors | Ryan Banks, Vishal Thengane, María Eugenia Guerrero, Nelly Maria García-Madueño, Yunpeng Li, Hongying Tang, Akhilanand Chaurasia |
| Institutions | University of Surrey (UK); UNMSM and USMP (Peru); King's College London (UK); King George's Medical University (India) |
| Dataset | perio-KPT — Zenodo 14711842 (v1.0, 33.0 GB) and 17272200 (v2.0, 37.4 GB) |
| Size | 192 IOPA + 15 external-validation IOPA + 3,588 auxiliary panoramic radiographs |
| Modality | Intraoral periapical (primary); panoramic (auxiliary tooth-segmentation pretraining) |
| Annotation | Per-tooth YOLOv8-pose format. Keypoints: CEJ-mesial, CEJ-distal, BL-mesial, BL-distal, RL-mesial, RL-distal, plus ARR (alveolar ridge resorption) keypoints and PLS (periodontal ligament space) bboxes. Tooth bboxes are *rotated* (oriented). 3 tooth classes: single-root, double-root, triple-root. Each keypoint has a visibility flag (0/1/2) so missing landmarks are explicit rather than imputed. Total 3,520 keypoints. |
| Code | github.com/Banksylel/Bone-Loss-Keypoint-Detection-Code — **MIT license** `[verified]`. Includes YOLOv8n-pose base weights, fine-tuned weights via Google Drive, train/eval/predict scripts, custom ultralytics config files |
| Architectures benchmarked | YOLOv8-Pose (end-to-end), HRNet (top-down), DeepPose+ResNet50+RLE, RTMPose-tiny + RTMDet-tiny |
| Reported metrics | Best YOLOv8 PRCK⁰·⁵ = 0.912 (validation), PRCK⁰·⁰⁵ = 0.404 (validation). PBL classification mesial Dice 0.508 (HRNet) / 0.425 (YOLOv8). Tooth orientation NMSE 0.0046. |
| Heuristic post-processing | Aligns predicted keypoints to tooth boundaries using an auxiliary instance segmentation model — directly addresses the "keypoint floats off tooth" failure mode |
| Paper license | CC-BY 4.0 (per arXiv) `[verified]` |
| Zenodo license | CC-BY-NC-SA 2.0 Generic `[verified]` — **inconsistent with paper claim**; resolve before relying on it for redistribution |
| Access | Files marked "Restricted" on both Zenodo records `[verified]` 2026-05-12. Requires login + access request describing institutional affiliation. **Public application form, no fee documented.** |

**Fit-to-use-case: very high if accessible.** This is the closest existing
work to the exact pipeline being built. Per-tooth grouping is native to the
annotation format — no adapter rewrite needed. The MIT-licensed code is
directly reusable. Sample size (192 IOPA) is smaller than DenPAR (1,000)
but the *annotation quality per tooth* is the bottleneck, not image count,
and 192 IOPA × ~6 teeth ≈ 1,150 clean per-tooth keypoint sets exceeds
DenPAR's ≈420 clean teeth at present.

**Gotchas:**
- Zenodo access restriction is the load-bearing blocker. Worth a same-day
  application. Application is described as standard institutional-affiliation
  + intended-use form; "personal academic study" framing may or may not pass.
- License inconsistency between paper (CC-BY) and Zenodo (CC-BY-NC-SA 2.0
  Generic) means downstream redistribution of derived weights is ambiguous.
  Training a model and using it personally is almost certainly OK under
  NC-SA; releasing trained weights publicly under MIT would require
  resolving the upstream license question with the authors directly.
- v2 (37.4 GB) bundles auxiliary panoramic data — useful for tooth
  segmentation pretraining but you only need the 192 IOPA + annotations
  (`0_Baseline/` + `1_Experiment/` folders) for the perio signal.
- Single-institution annotation in Peru-Spain-India consortium; demographics
  / radiograph hardware not matched to typical North American GP intraoral
  sensors. Bias is unknown but probably moderate.

### 3.2 Lin, Mao et al. 2024 — "Evaluation of the Alveolar Crest and CEJ in Periodontitis Using Object Detection on Periapical Radiographs"

| Field | Value |
|---|---|
| Paper | MDPI Diagnostics 14(15):1687 (2024) |
| Authors | Tai-Jung Lin, Yi-Cheng Mao, Yuan-Jin Lin, Chin-Hao Liang, Yi-Qing He, Yun-Chen Hsu, Shih-Lun Chen, Tsung-Yi Chen, Chiung-An Chen, Kuo-Chen Li, Patricia Angela R. Abu |
| Institution | Chang Gung Memorial Hospital (Taiwan); IRB 202301730B0 |
| Dataset | 140 original PA → 420 augmented (training); 57 val (171 aug); 84 test. ~281 unique PA total. |
| Modality | Intraoral periapical |
| Annotation | Polygon-marked CEJ and ALC (alveolar crest level) per-tooth, by 5 dental practitioners with ≥5 years experience using Roboflow. CLAHE preprocessing applied. |
| Code | Not stated as released `[verified from PMC abstract]` |
| Data availability | "Inquiries directed to corresponding authors" — **not public** |
| Reported metrics | YOLOv8 tooth detection 97.01% accuracy with CLAHE; tooth mask 93.48%, bone mask 96.95%, tooth-mask DSC 0.9478, CEJ/ALC positioning RMSE < 0.09 (min 0.0209). |

**Fit:** Methodologically informative — demonstrates that CEJ + ALC polygon
annotation in Roboflow produces RMSE < 0.09 on PA with modest data. Not
directly usable; the dataset isn't released. Read as **methodology
template**, not as a dataset candidate.

### 3.3 Lee et al. 2025 — "Evaluation by dental professionals of an AI-based application to measure alveolar bone loss"

| Field | Value |
|---|---|
| Paper | BMC Oral Health 25, art 5677 (March 2025) |
| Institution | Columbia University; de-identified database 387 patients, 2019-12-01 to 2020-08-24 |
| Dataset | 550 BW radiographs |
| Modality | **Bitewing only** |
| Annotation | Board-certified oral radiologist (20+ yrs); 60×60 px boxes around ABCL and CEJ; intra-rater ICC 0.98 |
| Architecture | 5-network conglomerate: 2 object detectors (CEJ box + ABCL box) → semantic segmentation → tooth segmentation; train/val/test 7:2:1 |
| Reported metrics | 94% overall accuracy (vs. 68% for dental professionals on same images); semantic seg 0.9567 tooth / 0.9281 alveolar bone; AP 0.72 (CEJ), 0.65 (ABCL); 82–87% accuracy for severe (>5 mm) periodontal bone loss classification |
| Code | Not stated as released |
| Dataset availability | "Additional data may be requested from the authors" — **not public** |

**Fit:** Tantalizing because it's the rare **bitewing-only** alveolar bone
loss dataset with high-quality per-tooth CEJ/ABCL boxes (ICC 0.98) — but
private. Worth an email to the authors requesting CC-BY release; low
probability of success but cheap to try. Architecture pattern (separate
detectors for CEJ vs ABCL → segmentation) is a useful template if the
per-keypoint head continues to collapse.

### 3.4 Alqaderi et al. 2026 (medRxiv) — Tufts dental caries + bone loss on PA/BW

| Field | Value |
|---|---|
| Paper | medRxiv 10.64898/2026.04.12.26350726v1 (April 2026) `[blocked — 403 on direct fetch]` |
| Institution | Tufts University School of Dental Medicine (Axium EHR); Dr. Hend Alqaderi corresponding |
| Dataset | 1,063 PA + BW radiographs |
| Modality | Intraoral PA and BW (mixed) |
| Reported metrics | "97% F1 bone loss" claimed in upstream framing; exact metric `[needs verification — page 403]` |
| Code | `[needs verification]` — search returns no public repo |
| Dataset | `[needs verification]` — single-institution retrospective EHR pull; redistribution unlikely under standard Tufts IRB |

**Fit:** Probably zero unless the authors release. Single-institution Axium
EHR retrospective pulls are essentially never released under CC-BY due to
US HIPAA constraints on de-identified clinical data redistribution. Worth
tracking but not actionable. The 97% F1 number is probably true on their
held-out test split and probably overstated relative to external
generalization.

### 3.5 PRAD / PRAD-10K (MICCAI 2025)

| Field | Value |
|---|---|
| Paper | arXiv 2504.07760; MICCAI 2025 poster |
| Institution | Nankai University (China) |
| Dataset | 10,000 PA with pixel-level segmentation of 9 classes (tooth, alveolar bone, pulp, root canal filling, denture crown, dental fillings, implant, orthodontic devices, apical periodontitis) |
| Modality | Periapical |
| Annotation | Pixel-level segmentation by endodontists. **No CEJ keypoints, no per-tooth bone-level lines.** |
| Code | github.com/nkicsl/PRAD — open `[verified]` |
| Dataset license | CC-BY-NC 4.0 `[verified]` |
| Access | **By application** — institutional form + email to aics@nankai.edu.cn, ~14 working days `[verified]` |

**Fit:** Wrong shape for perio. PRAD-10K's "alveolar bone" class is a
**segmentation mask of the bone region**, not a bone-level *line* or per-tooth
bone-crest *keypoint*. The bone mask is useful as an auxiliary segmentation
pretraining signal — would help the bone-segmentation head — but the perio
keypoint problem (CEJ collapse) is unaffected. Worth applying for to
strengthen the tooth and bone segmentation heads.

### 3.6 AlGhaihab et al. 2025 — UNC preliminary BW+PA evaluation (39 images)

| Field | Value |
|---|---|
| Paper | MDPI Diagnostics 15(5):576 (2025) |
| Authors | AlGhaihab, Moretti, Reside, Tuzova, Huang, Tyndall |
| Institution | UNC Chapel Hill; data sourced from Denti.AI Technology Inc. |
| Dataset | 39 radiographs (22 PA + 17 BW), 316 tooth surfaces |
| Annotation | Consensus panel of 3 board-certified specialists. Kappa 0.69 (PA), 0.83 (BW). |
| Code | Not released |
| Data availability | "Available from corresponding author upon reasonable request" — **not public** |
| Reported metrics | PA: 76% sens / 86% spec / 0.046% MAE. BW: 65% sens / 90% spec / 0.499 mm MAE. |

**Fit:** Too small (39 images) to be a training corpus. Useful as a
**bias check** — they're evaluating a commercial product (Denti.AI) on a
small US specialist-annotated set, which gives realistic mid-2020s baseline
sensitivity/specificity numbers for the PA+BW intraoral perception task
(~76% sens / 86% spec on PA; ~65% sens / 90% spec on BW).

### 3.7 Ameli et al. 2024 — Alberta, automating bone loss measurement on PA

| Field | Value |
|---|---|
| Paper | Frontiers in Dental Medicine 5:1479380 (2024) |
| Institution | University of Alberta |
| Dataset | 1,000 PA train (572 patients) + 1,582 PA test (210 patients) |
| Modality | Periapical |
| Annotation | Polygon-mask binary annotation by a dentist + periodontist via Roboflow; consensus labels; YOLO-v9 for apex bbox; U-Net for bone-loss segmentation |
| Code | Not stated |
| Data availability | "Not publicly available; from authors upon reasonable request" — **not public** |
| Reported metrics | U-Net seg: 95.62% accuracy, 97.26% precision, 63.90% recall, 81.69% IoU. YOLOv9 apex: 66.7% mAP. ICC for bone-loss measurement > 0.94. Stage III/IV F1 0.945, Grade C F1 0.83. |

**Fit:** Largest *private* PA bone-loss dataset published (~2,582 PA total
with bone-loss + apex annotation). High-quality reported numbers but
inaccessible. Worth an email; if released would be the dominant alternative
to DenPAR.

### 3.8 Chen et al. 2023 — 8,000 PA (the alleged big one)

| Field | Value |
|---|---|
| Paper | Journal of Dental Sciences 18(3):1301–1309 (July 2023) |
| Dataset | 8,000 PA with 27,964 teeth |
| Modality | Periapical |
| Annotation | VIA labeling platform; YOLOv5 tooth detection; ensemble of VGG-16 + U-Net |
| Reported metrics | "97% accuracy" via U-Net ensemble |
| Code / data | **Not released** — single-institution retrospective `[needs verification of code search]` |

**Fit:** Despite the size, this paper is widely cited as "8,000 PA" but the
dataset has never been released in any public repository. Treat as
**unavailable**.

### 3.9 Lee, Kabir, Jiang et al. 2022 — Use of DL to Measure Alveolar Bone Level

| Field | Value |
|---|---|
| Paper | arXiv 2109.12115; J Clin Periodontol 49:260 (2022) |
| Institution | UT Health Houston (Chun-Teh Lee, Tanjida Kabir, Xiaoqian Jiang, Shayan Shams) |
| Dataset | 693 PA |
| Modality | Periapical |
| Annotation | Three-segmentation-network pipeline: bone-area, tooth, CEJ — each as a segmentation mask, not a keypoint |
| Reported metrics | DSC > 0.91; RBL stage AUC 0.89/0.90/0.90 for stages I/II/III; case diagnosis accuracy 0.85 |
| Code / data | **Not released** |

**Fit:** Reference architecture — segmenting CEJ as a thin pixel band rather
than as paired keypoints sidesteps the per-tooth pairing problem entirely.
Worth considering as an architecture pivot: replace the CEJ keypoint head
with a CEJ *line* segmentation head, then derive mesial/distal keypoints
from line endpoints intersected with the tooth bbox. This would also work
on DenPAR data alone if the loose 2D point list can be densified into a
short line segment.

### 3.10 Tsoromokos et al. 2022 — ACTA/Amsterdam pilot PA

| Field | Value |
|---|---|
| Paper | Int Dent J 72(5):621–627 (2022) |
| Institution | ACTA Amsterdam |
| Dataset | 1,546 approximal sites across 54 participants, mandibular PA |
| Modality | Periapical (mandibular only) |
| Reported metrics | ICC 0.601 overall; ICC 0.763 for non-molar (incisor/canine/premolar) |
| Code / data | **Not released** |

**Fit:** Read as a sober baseline — modest dataset, modest ICC, honestly
reported. Confirms molars are harder than anteriors (consistent with our
own observation that CEJ head collapses preferentially on molars).

### 3.11 Bayrakdar et al. 2020 / Kurt-Bayrakdar et al. 2024 / Uzun Saylan 2023 / Ryu 2023 / Sunnetci 2022 / Chang 2020

All **panoramic-only**. Excluded from the core analysis by the modality
constraint. Summarized in the comparison table for completeness. Note that
Wimalasiri's Table 1 framing of "Bayrakdar 2020 (panoramic, 1139+1137)"
appears to refer to two separate Bayrakdar publications; the closest single
publication search-verified is Bayrakdar et al. Cumhuriyet Dental Journal
2020 with 2,276 panoramic.

### 3.12 HUNT4 / AI-Dentify (Pérez de Frutos et al. 2024)

| Field | Value |
|---|---|
| Paper | BMC Oral Health 24, art 220 (March 2024); arXiv 2310.00354 |
| Institution | NTNU / HUNT Research Centre, Norway |
| Dataset | 13,887 bitewings, annotated by 6 experts |
| Modality | Bitewing |
| Annotation | Bounding-box caries + perio findings (per AI-Dentify focus on caries primarily); not per-tooth CEJ keypoints |
| Code | NTNU Open arXiv release of model code (per AI-Dentify framing) `[needs verification]` |
| Dataset availability | **Restricted.** Norwegian Regional Ethical Committee approval + HUNT Research Centre approval required; data is not deposited in open repositories by policy. `[verified]` |

**Fit:** Zero for our redistribution-friendly path. HUNT data cannot be
re-shared even after access; downstream models trained on it can be
published but the data substrate is locked behind Norwegian ethics review.

### 3.13 PhysioNet Multimodal Dental Dataset (16,203 PA among CBCT + panoramic)

| Field | Value |
|---|---|
| Source | PhysioNet, doi 10.13026/h1tt-fc69 |
| Modality | Mixed CBCT + panoramic + intraoral; 16,203 PA images among the intraoral subset |
| License | CC-BY-NC-ND 4.0 |
| Annotation | Box-level labeling (not keypoint, not pixel-level bone-level lines) |
| Access | Restricted PhysioNet credentialed access |

**Fit:** Large PA corpus but annotation shape (boxes, multiple classes) is
not perio-specific. Useful as a tooth-detection pretraining corpus if
PhysioNet credentialing is acceptable; not a perio signal source.

### 3.14 Sunnetci 2022 / Krois 2019 (panoramic-only)

Out of scope by modality. Both released to literature but datasets are
private.

---

## 4. Comparison table

| Paper / dataset | Year | Modality | Size | Data public? | Code public? | License (data) | Per-tooth CEJ keypoints? | Reported metric | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **DenPAR v3 / Wimalasiri** | 2024/26 | PA | 1,000 | ✅ Zenodo open | ✅ GitHub (no license stated) | CC-BY 4.0 | ❌ flat 2-D point list, no grouping | ICC 0.80 / pattern acc 87% | **current baseline; CEJ adapter is the bug** |
| **Banks / perio-KPT** | 2025 | PA (+aux panoramic) | 192 IOPA + 3,588 aux | ⚠ Zenodo restricted (req. login) | ✅ MIT GitHub | CC-BY-NC-SA 2.0 (Zenodo) / CC-BY (paper) | ✅ YOLO-pose, visibility-flagged | PRCK⁰·⁵ 0.912 val | **⭐ primary alternative if access granted** |
| **PRAD-10K** | 2025 | PA | 10,000 | ⚠ by application (~14 days) | ✅ GitHub | CC-BY-NC 4.0 | ❌ pixel-seg of 9 classes, no CEJ | seg benchmarks | **aux pretraining for tooth/bone seg** |
| Lin/Mao 2024 (Chang Gung) | 2024 | PA | ~281 unique | ❌ author request | ❌ | n/a | ✅ polygon (Roboflow) | RMSE <0.09 | methodology template only |
| Lee 2025 (Columbia) | 2025 | BW | 550 | ❌ author request | ❌ | n/a | ✅ 60×60 boxes, ICC 0.98 | 94% acc | bitewing template only |
| Alqaderi 2026 (Tufts medRxiv) | 2026 | PA+BW | 1,063 | ❌ single-institution | ❌ | n/a | unverified | 97% F1 claimed | unavailable, unverified |
| Ameli 2024 (Alberta) | 2024 | PA | 2,582 | ❌ author request | ❌ | n/a | seg + apex box | F1 0.945 stage III/IV | unavailable |
| Chen 2023 (JDS) | 2023 | PA | 8,000 | ❌ unreleased | ❌ | n/a | n/a | 97% acc claimed | unavailable |
| Lee/Kabir 2022 (UTHealth) | 2022 | PA | 693 | ❌ | ❌ | n/a | ❌ (seg) | DSC>0.91; AUC ~0.90 | architecture template |
| Tsoromokos 2022 (ACTA) | 2022 | PA mand | 446 | ❌ | ❌ | n/a | per-site | ICC 0.601 | sober baseline |
| Alotaibi 2022 (KSAU-HS) | 2022 | PA anterior | 1,724 | ❌ ROMEXIS internal | ❌ | n/a | classification | CNN | unavailable |
| AlGhaihab 2025 (UNC) | 2025 | PA+BW | 39 | ❌ | ❌ | n/a | n/a | 76/86% sens/spec | too small + private |
| HUNT4 / AI-Dentify | 2024 | BW | 13,887 | ❌ Norwegian ethics-gated | ✅ NTNU Open | n/a | ❌ caries-focused boxes | various | data unreshareable |
| PhysioNet Multimodal | — | PA+CBCT+pan | 16,203 PA | ⚠ credentialed | n/a | CC-BY-NC-ND 4.0 | ❌ boxes | n/a | aux only |
| Bayrakdar 2020 | 2020 | panoramic | 2,276 | ❌ | ❌ | n/a | n/a | binary | **panoramic, out of scope** |
| Sunnetci 2022 | 2022 | panoramic | 1,432 | ❌ | ❌ | n/a | n/a | 81% | out of scope |
| Ryu 2023 | 2023 | panoramic | 4,083 | ❌ | ❌ | n/a | n/a | Faster R-CNN | out of scope |
| Uzun Saylan 2023 | 2023 | panoramic | 685 | ❌ | ❌ | n/a | n/a | YOLOv5 | out of scope |
| Chang 2020 | 2020 | panoramic | 340 | ❌ Seoul Nat'l IRB-restricted | ❌ | n/a | n/a | ICC 0.91 | out of scope |
| Kurt-Bayrakdar 2024 | 2024 | panoramic | 1,121 | ❌ | ❌ | n/a | n/a | AUC 0.910/0.733 | out of scope |
| Krois 2019 | 2019 | panoramic | 2,001 segments | ❌ | ❌ | n/a | n/a | 0.81 acc | out of scope |
| Tufts Dental Database | 2021 | panoramic | 1,000 | ⚠ restricted | ❌ | unspecified | rich anno but panoramic | n/a | out of scope |

---

## 5. Decision tree

```
Q1: Is the perception failure (CEJ head collapse) caused by missing data,
    or by missing per-tooth supervision?
    ├── If missing data → need more annotated images
    └── If missing per-tooth supervision → fix annotation pipeline, not
        dataset scale
              ↓
Q2: Can the existing DenPAR points be re-grouped per-tooth with high
    confidence?
    ├── Yes (≈42% of teeth currently): train the keypoint head only on
    │   the clean subset, set `visibility=0` on ambiguous teeth, accept
    │   smaller effective N
    └── No (≈58% of teeth): need a second supervisory source
              ↓
Q3: Is Banks et al. perio-KPT accessible?
    ├── Yes (Zenodo request approved): use it as the per-tooth CEJ
    │   supervisory signal; co-train with DenPAR on tooth detection + bone
    │   segmentation. Decision branch is well-supported by an MIT codebase.
    └── No: fall back to (a) DenPAR clean-subset training + visibility
              masks, OR (b) pivot CEJ head from keypoint to thin-line
              segmentation (Lee/Kabir 2022 pattern), OR (c) email Columbia
              and Alberta groups requesting CC-BY release of their datasets.
              ↓
Q4: Does the tooth-segmentation and bone-segmentation head need more
    pretraining data?
    ├── Yes → apply for PRAD-10K (CC-BY-NC 4.0, 10K PA with bone+tooth
    │         pixel masks, ~14-day approval)
    └── No → skip
```

---

## 6. Recommendation

1. **Today.** Submit a Zenodo access request on perio-KPT v1.0
   (14711842, 33 GB) describing the intended use as personal-scope
   open-source CLI development for dental radiograph analysis. Same
   request copied to v2.0 (17272200, 37.4 GB).

2. **Today, parallel.** Apply for PRAD-10K via aics@nankai.edu.cn — same
   day, ~14 working days lead time. Purely an auxiliary tooth + bone
   segmentation pretraining corpus.

3. **In parallel — does not block on either dataset.** Implement the
   DenPAR CEJ adapter fix: per-tooth bbox-grouped clustering of the flat
   point list, strict 2-keypoint-per-tooth gate, visibility=0 for teeth
   that fail the gate. Re-run training on the clean ≈42% subset. Measure
   whether CEJ collapse rate drops on the held-out set independent of any
   new dataset. This isolates "adapter bug" from "annotation gap" as the
   root cause.

4. **If perio-KPT access is granted.** Co-train the CEJ keypoint head on
   the combined corpus (DenPAR clean subset + perio-KPT). Banks's
   YOLOv8-pose configuration is the closest in shape to the current
   pipeline and the heuristic post-processing (snap keypoint to tooth
   mask boundary) drops in directly.

5. **If perio-KPT access is denied or delayed beyond 2 weeks.** Consider
   the architecture pivot: replace the CEJ keypoint head with a thin-line
   CEJ segmentation head following Lee/Kabir 2022. Derive mesial/distal
   keypoints from line endpoints. This sidesteps the per-tooth pairing
   problem inside the annotation schema.

6. **Defer.** Author emails to Columbia (Lee 2025) and Alberta (Ameli
   2024) requesting CC-BY release. Low yield, but free to send.

---

## 7. Open follow-ups

- **License resolution on perio-KPT.** Paper claims CC-BY, Zenodo records
  show CC-BY-NC-SA 2.0 Generic. Direct email to Banks et al. requesting
  clarification of redistribution rights, particularly for downstream
  trained weights. `[blocker on any future open-source weight release]`
- **Wimalasiri code license.** GitHub repo
  github.com/chathurawimalasiri/analysis-in-detecting-alveolar-bone-loss
  has no license file `[verified]`. Reuse of training code without a
  license is legally ambiguous (default copyright). Email request to add
  a permissive license, or clean-room reimplement from the paper (which
  is what is already being done).
- **PRAD-10K access timeline.** ~14 working days quoted; track actual.
- **Alqaderi medRxiv 2026 full text.** medRxiv direct fetch returned 403
  in this session; manually retrieve the PDF and verify whether code or
  data are released. Specifically check whether the 97% F1 number is on
  bone-loss segmentation, classification, or detection.
- **HUNT4 secondary release pathway.** Check whether NTNU has any
  derivative non-PHI artifact (e.g., model weights, synthetic annotations,
  feature embeddings) released under a more permissive license.
- **Banks's auxiliary panoramic set provenance.** 3,588 panoramic
  radiographs bundled in perio-KPT v2 — confirm whether these are from a
  separately-licensed corpus (likely DENTEX or similar) and whether the
  license is compatible.
- **DenPAR companion paper.** Sci Data 2025 paper on DenPAR (Rasnayaka,
  Bandara et al.) may contain the explicit CEJ annotation methodology
  description that clarifies whether per-tooth grouping is documented
  anywhere — current adapter assumption that points are loose may be a
  trainer-side bug rather than a dataset gap. Worth a careful re-read.

---

## 8. Sources

### Primary candidate paths

- [Banks et al. 2025 — Periodontal Bone Loss Analysis via Keypoint Detection (arXiv)](https://arxiv.org/abs/2503.13477)
- [Banks et al. 2025 — full HTML v3](https://arxiv.org/html/2503.13477v3)
- [perio-KPT v1.0 (Zenodo 14711842)](https://zenodo.org/records/14711842)
- [perio-KPT v2.0 (Zenodo 17272200)](https://zenodo.org/records/17272200)
- [Bone-Loss-Keypoint-Detection-Code (GitHub, MIT)](https://github.com/Banksylel/Bone-Loss-Keypoint-Detection-Code)

### DenPAR / Wimalasiri (current baseline)

- [Wimalasiri et al. 2026 (Sci Rep)](https://www.nature.com/articles/s41598-026-38061-1)
- [Wimalasiri et al. 2026 (arXiv 2506.20522)](https://arxiv.org/abs/2506.20522)
- [Wimalasiri training repo (no license)](https://github.com/chathurawimalasiri/analysis-in-detecting-alveolar-bone-loss)
- [Wimalasiri secondary code repo](https://github.com/chathurawimalasiri/Intraoral-periapical-radiography-codes)
- [DenPAR v3 Zenodo record](https://zenodo.org/records/16645076)
- [DenPAR v1 / earlier Zenodo record](https://zenodo.org/records/13998619)
- [DenPAR Scientific Data paper](https://www.nature.com/articles/s41597-025-05906-9)

### Other intraoral perio papers (all data-restricted)

- [Lin/Mao et al. 2024 — Alveolar Crest + CEJ Object Detection (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11312231/)
- [Lin/Mao et al. 2024 (MDPI Diagnostics)](https://www.mdpi.com/2075-4418/14/15/1687)
- [Lee et al. 2025 Columbia BW evaluation (BMC Oral Health)](https://link.springer.com/article/10.1186/s12903-025-05677-0)
- [Lee et al. 2025 Columbia BW evaluation (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11872301/)
- [Ameli et al. 2024 Alberta (Frontiers Dent Med)](https://www.frontiersin.org/journals/dental-medicine/articles/10.3389/fdmed.2024.1479380/full)
- [Lee/Kabir/Jiang 2022 (arXiv 2109.12115)](https://arxiv.org/abs/2109.12115)
- [Lee/Kabir/Jiang 2022 PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9026777/)
- [Tsoromokos 2022 (Int Dent J via SD)](https://www.sciencedirect.com/science/article/pii/S002065392200034X)
- [AlGhaihab 2025 UNC preliminary (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11899607/)
- [AlGhaihab 2025 UNC preliminary (MDPI Diagnostics)](https://www.mdpi.com/2075-4418/15/5/576)
- [Chen 2023 8000 PA (J Dent Sci, PubMed)](https://pubmed.ncbi.nlm.nih.gov/37404656/)
- [Alotaibi 2022 (BMC Oral Health)](https://link.springer.com/article/10.1186/s12903-022-02436-3)
- [Sunnetci 2022 (Biomed Signal Process Control)](https://www.sciencedirect.com/science/article/abs/pii/S1746809422003664)
- [Alqaderi 2026 medRxiv (full PDF)](https://www.medrxiv.org/content/10.64898/2026.04.12.26350726v1.full.pdf)
- [Bitewing multi-class detection 2025 (1197 BW)](https://www.mdpi.com/2075-4418/16/2/322)

### Auxiliary segmentation candidates

- [PRAD-10K (arXiv 2504.07760)](https://arxiv.org/abs/2504.07760)
- [PRAD-10K (MICCAI 2025 paper PDF)](https://papers.miccai.org/miccai-2025/paper/0247_paper.pdf)
- [PRAD GitHub (Nankai)](https://github.com/nkicsl/PRAD)

### Panoramic (out of modality scope but referenced in Wimalasiri Table 1)

- [Chang 2020 hybrid (Sci Rep)](https://www.nature.com/articles/s41598-020-64509-z)
- [Ryu 2023 (Applied Sciences)](https://www.mdpi.com/2076-3417/13/9/5261)
- [Uzun Saylan 2023 (Diagnostics)](https://www.mdpi.com/2075-4418/13/10/1800)
- [Kurt-Bayrakdar 2024 (BMC Oral Health)](https://link.springer.com/article/10.1186/s12903-024-03896-5)
- [Krois 2019 (Sci Rep)](https://www.nature.com/articles/s41598-019-44839-3)

### HUNT4 / AI-Dentify

- [Pérez de Frutos et al. 2024 (BMC Oral Health)](https://link.springer.com/article/10.1186/s12903-024-04120-0)
- [AI-Dentify arXiv 2310.00354](https://arxiv.org/abs/2310.00354)

### Survey / catalog references

- [Uribe et al. 2024 — Publicly Available Dental Image Datasets (J Dent Res)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11633071/)
- [sergiouribe/dental_datasets_itu (GitHub catalog)](https://github.com/sergiouribe/dental_datasets_itu/blob/main/AI_Dental_Datasets_List.md)
- [Tufts Dental Database](https://tdd.ece.tufts.edu/)

### Systematic reviews of the perio-DL field

- [Applicability scoping review (Oral Radiol 2025)](https://link.springer.com/article/10.1007/s11282-025-00839-w)
- [Periodontal bone loss systematic review (Dent J 2025)](https://www.mdpi.com/2304-6767/13/9/413)
- [Frontiers panoramic perio review 2024](https://www.frontiersin.org/journals/dental-medicine/articles/10.3389/fdmed.2024.1509361/full)
- [Detection of perio bone loss via ML/DL meta-analysis (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11979759/)
