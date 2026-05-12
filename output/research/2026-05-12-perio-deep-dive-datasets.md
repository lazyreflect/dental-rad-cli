# Perio / Bone-Loss Dataset Deep-Dive — Data-Repository Survey

**Author:** research-session Claude (cold start, data-repository focus)
**Date:** 2026-05-12
**Status:** Research only — no code written. No shipping decisions made.
**Companion:** `2026-05-12-perio-deep-dive-academic.md` covers the academic-paper
angle. This document is the **aggressive data-repository survey** —
Roboflow / Zenodo / Mendeley / HuggingFace / Kaggle / DatasetNinja / GitHub —
asking the Renielaz question: *what is actually downloadable today, and what
does the schema actually contain?*
**Trigger:** CEJ keypoint head collapses on inference because DenPAR v3 ships
CEJ as a flat 2-D point list with no per-tooth grouping; only ~42% of teeth
receive clean 2-keypoint supervision after the heuristic adapter. Same shape
of failure as caries had with Renielaz on Roboflow (description-bullet
"classes" corrupted the YAML). The cure for monoculture-on-one-dataset is the
same in both cases: aggressive cross-repo dataset survey, **probe-before-trust**.

Confidence convention: `[verified]` = file metadata or page contents read this
session; `[probe pending]` = inferred from descriptions, not file-level
verified; `[gated]` = exists but access blocked (login / IRB / institutional);
`[blocked-fetch]` = could not retrieve page this session (Cloudflare /
redirect / 403). Roboflow Universe pages were blocked from direct fetch by
Cloudflare anti-bot — Roboflow entries are `[probe pending]` and require a
follow-up Roboflow-SDK probe per `scripts/_probe_roboflow_*.py`.

---

## 1. TL;DR

**Primary candidate — `perio-KPT` (Zenodo 14711842 / 17272200).** [verified-metadata, gated-files]
Direct schema match to what we need (per-tooth grouped CEJ-mesial / CEJ-distal
keypoints in YOLOv8-pose format with visibility flags). 192 IOPA + 3,588
panoramic auxiliary + 15 external validation images, 11 keypoint classes per
tooth bbox, 5 bbox classes including ARR (alveolar ridge resorption) and PLS
(periodontal ligament space). **Blocker:** Zenodo files are restricted-access
(login + university affiliation request) AND license is **CC-BY-NC-SA 2.0
Generic** — non-commercial-research only and viral share-alike. Personal-use
fit is plausibly compatible; redistribution and any forward open-source ship
is not. Action: file the Zenodo access request today; license tier-check with
Joseph before downloading. This is the **same dataset** described in the
companion academic-survey doc; this section reports its file-repository state.

**Primary fallback — `Dataset for Automating Dental Condition Detection on
Panoramic Radiographs` (Zenodo 15487430, also mirrored at HF
`ismaelportog/Panoramic_Radiographs_for_Dental_Condition`).** [verified, open]
1,628 train + 180 external-validation panoramic radiographs, **CC-BY 4.0,
open download**, YOLO format, 14 conditions INCLUDING `BON` (Bone Resorption)
and `FUR` (Furcation Lesion). Modality mismatch (panoramic vs the IOPA/BW
chairside intent) — **not a v0 fit** but useful as evidence the
class-of-interest is shippable as a downstream task, and as future
out-of-distribution generalization sanity-check once IOPA training is solid.

**Secondary fallback — `BRAR-anchored multimodal dataset` (Figshare
30155974.v3).** [probe pending, likely open]
1,104 patients, panoramic radiographs + tooth-level BRAR scores (3 severity
levels: BL / RL / Age ratio). Modality mismatch (panoramic), and annotations
are **classification-grade not segmentation-grade** (per-tooth BRAR score,
not per-tooth bone-level polylines or CEJ keypoints). Useful as
evaluation-tier label benchmark, not training source for the CEJ head.

**Rejections — fully cataloged in §4:**
- PRAD-10K (Zenodo / nkicsl GitHub): gated + CC-BY-NC + no CEJ in 9-class list.
- DentalX (Surrey, IRB internal): no public download.
- Tufts Dental Database: panoramic-only, gated access.
- DENTEX (HuggingFace): panoramic-only, no bone loss / CEJ.
- AI-Dentify / HUNT4: not public.
- DenPAR v3: this is what we already use; documented as the upstream source.
- Mendeley `kx52tk2ddj`, `yt8f2zzfpt`, `ccw5mvg69r`, `7xgzy69fw2`:
  modality or annotation mismatch (panoramic, binary, osteoporosis-only).
- HuggingFace dataset surface: no annotated radiograph dataset with CEJ /
  bone-loss in the searchable index (verified by listing all dental-named HF
  datasets; the relevant one is the Zenodo 15487430 mirror).
- DatasetNinja DentalAI: 4-class (tooth/caries/cavity/crack), no bone loss.
- Roboflow `training-horizontal-bone-loss`: panoramic, 136 images, mild/mod/
  severe classification only — too small + wrong modality + no keypoints.

**Bottom line.** The data-repository survey **confirms** the academic-survey
finding: **perio-KPT is the only public dataset whose annotation shape solves
the per-tooth-grouped-CEJ problem.** All other repository candidates are
either modality-mismatched (panoramic), annotation-mismatched (bbox/mask but
no per-tooth keypoint pairing), or schema-mismatched (binary / 4-class /
osteoporosis). The downstream problem reduces to **resolving perio-KPT's
license + access friction** OR **shipping the DenPAR-only adapter fix** from
the academic doc's fallback path.

---

## 2. Constraints recap

| Constraint                | Requirement                                          |
|---------------------------|------------------------------------------------------|
| License                   | CC-BY 4.0 or more permissive (CC0 / Public Domain / MIT / Apache-2). CC-BY-NC acceptable for personal-use only with explicit license-tier confirmation. |
| Modality                  | Bitewing OR periapical (PRIMARY). Panoramic only acceptable as auxiliary signal or for evaluation. |
| Size                      | ≥500 images preferred. Smaller acceptable if annotation quality compensates. |
| Accessibility             | Downloadable today (HTTP / Zenodo API / Roboflow SDK / HF datasets). NOT IRB-pending, NOT institutional-only, NOT email-application-required for v0.  |
| Annotation                | Per-tooth keypoints (CEJ-mesial, CEJ-distal at minimum) OR per-tooth bone-level polylines OR pixel-precise bone-line masks aligned to teeth. NOT image-level classification. NOT bbox-only. |
| Probe-before-trust        | Mandatory — Renielaz lesson. Inspect `data.yaml` / annotations.json before adopting. |

---

## 3. Promising candidates

### 3.1 `perio-KPT` — Periodontal Keypoint and Object Detection Dataset

| Property               | Value                                                       |
|------------------------|-------------------------------------------------------------|
| Host                   | Zenodo                                                      |
| v1 record              | https://zenodo.org/records/14711842 (Jan 21, 2025)         |
| v2 record              | https://zenodo.org/records/17272200 (Oct 5, 2025; mod Mar 22 2026) |
| Concept DOI            | `10.5281/zenodo.14711841`                                   |
| License                | **CC-BY-NC-SA 2.0 Generic** (Zenodo metadata) [verified]    |
| Access status          | **`access_right: restricted`** — login + university affiliation form [verified-via-Zenodo-API] |
| Code repo              | https://github.com/Banksylel/Bone-Loss-Keypoint-Detection-Code (MIT-licensed code) [verified] |
| Paper                  | Banks et al. 2025, arXiv 2503.13477 ([HTML](https://arxiv.org/html/2503.13477v1)) [verified] |
| v1 images              | 193 intraoral periapical radiographs                        |
| v2 images              | 192 IOPA (1 corrupted removed) + 3,588 panoramic-derived auxiliary + 15 external validation = 3,795 total |
| Teeth (v1)             | 582 (386 single-root + 160 double-root + 34 triple-root)    |
| Keypoints (v1)         | 3,520 total                                                 |
| Modality               | Intraoral periapical (PRIMARY); v2 adds panoramic-derived auxiliary |
| Format                 | YOLOv8-pose `.txt` + `.jpg` (per-tooth bbox carries an 11-keypoint array with visibility flags 0/1/2) |
| Bbox classes (5)       | 0=Single Root, 1=Double Root, 2=Triple Root, 3=Alveolar Ridge Resorption (ARR), 4=Periodontal Ligament Space (PLS) |
| Keypoint classes (11)  | Mesial/distal CEJ, mesial/distal BL, mesial/distal RL, furcation entrance (FA), furcation BL-mesial (FBL-m), furcation BL-distal (FBL-d), ARR landmark — visibility 0 (untrained), 1 (partially), 2 (fully visible) |
| Folder structure       | `Baseline/` (all 193 + rotational bbox) and `Experiment/` (5-fold cross-validation) [verified-from-page] |
| Train/val/test         | v2 uses 5-fold CV; holdout test set separately structured   |

**Probe results.**
- Zenodo API confirms `access_right: restricted`, license `cc-by-nc-sa-2.0` (Generic), v1.0 published 2025-01-21. [verified]
- The arXiv paper says license CC-BY in passing; the Zenodo metadata says CC-BY-NC-SA 2.0. **Internal inconsistency.** Trust the Zenodo metadata as canonical (it's the legal artifact).
- Code repo `Banksylel/Bone-Loss-Keypoint-Detection-Code` is MIT-licensed and is open; the dataset behind it is the gated piece. [verified]
- The companion arXiv paper (2503.13477) explicitly says: "Dataset: https://bit.ly/4hJ3aE7" which redirects to a Dropbox `BONE_LOSS_KPT_UPLOAD_FINAL.zip` link. [verified-via-redirect] This Dropbox copy is **probably** the original supplementary release before the formal Zenodo gating; status uncertain (could be open Dropbox today and gated tomorrow at maintainers' discretion). Worth probing once for current accessibility but not relying on for reproducibility.

**License unpacking (this is the critical decision input).**
- **CC-BY-NC-SA 2.0** = Attribution + NonCommercial + ShareAlike. Non-commercial use only. Any derivative must be released under the same license.
- "Non-commercial" per Creative Commons FAQ is **"not primarily intended for or directed toward commercial advantage or monetary compensation."** Personal-use, hobbyist, two-dentist-household use plausibly qualifies; any path toward selling the tool or shipping it under a non-NC license to office #47 does not. This is a Joseph-confirm checkpoint, not a Claude-decide one.
- **ShareAlike viral effect:** if the trained weights derive from perio-KPT, those weights and their downstream products plausibly inherit NC-SA. A clean v1.0 path that wants permissive licensing later cannot ship perio-KPT-derived weights without re-deriving from a permissive corpus.

**Fit-to-use-case.**
- Annotation shape: **exact match** for the v0 CEJ/BL keypoint pipeline. Per-tooth grouping is enforced by YOLO-pose's bbox-anchored keypoint array; mesial-distal pairing is the file format, not a heuristic.
- Image count: 192 IOPA is small but the annotation density (3,520 keypoints / 582 teeth) is high. Adequate as a supervisory signal for the CEJ head where DenPAR's ~420 cleanly-pairable teeth is the current effective n.
- Modality: matches the IOPA chairside intent.
- Code path: Banks et al.'s training repo accepts the dataset directly; integration cost is low if the access blocker resolves.

**Action items.**
1. File the Zenodo access request for v2 (record 17272200). Affiliation: list `dental-rad-cli` open-source project + personal-use dentist context. Estimated 14-day round-trip per Zenodo norms.
2. In parallel, probe the `bit.ly/4hJ3aE7` Dropbox link to see if it's still openly downloadable (the original supplementary). This does not resolve the license question but provides a probe of annotation-format correctness even before formal access.
3. License-tier check with Joseph: is CC-BY-NC-SA acceptable for personal use today, with the explicit knowledge that the weights derived from it inherit NC-SA and therefore can't ship to a third party?
4. If license is acceptable: download v2, probe the actual `.txt` annotation file format (confirm 11-keypoint per-tooth array order matches `[0=CEJ-m, 1=CEJ-d, 2=BL-m, 3=BL-d, ...]` claim), and integrate as a second supervisory signal alongside DenPAR.

---

### 3.2 `Dataset for Automating Dental Condition Detection on Panoramic Radiographs` — Zenodo 15487430

| Property               | Value                                                       |
|------------------------|-------------------------------------------------------------|
| Host                   | Zenodo                                                      |
| Record                 | https://zenodo.org/records/15487430                         |
| License                | **CC-BY 4.0** [verified]                                     |
| Access status          | **`access_right: open`** — direct download [verified]        |
| File                   | `panoramic_radiography_yolo_dataset_14_classes.zip` (1.4 GB) [verified] |
| HF mirror              | https://huggingface.co/datasets/ismaelportog/Panoramic_Radiographs_for_Dental_Condition (27,884 rows; 1.42 GB) [verified] |
| Train / val / test     | 18,600 / 4,950 / 4,360 rows on HF mirror; 1,628 train + 180 external-validation on Zenodo (HF mirror appears to be augmented) [verified] |
| Modality               | Panoramic (single-modality, Vatech PCH-2500 device)         |
| Format                 | YOLO bbox `.txt` annotations                                |
| Classes (14)           | IMP (Implant), PRR (Prosthetic Restoration), OBT (Obturation), END (Endodontic Treatment), CAR (Carious Lesion), **BON (Bone Resorption)**, IMT (Impacted Tooth), API (Apical Periodontitis), ROT (Root Fragment), **FUR (Furcation Lesion)**, APS (Apical Surgery), ROR (Root Resorption), ORD (Orthodontic Device), SRD (Surgical Device) |
| Last update            | May 22, 2025                                                |

**Probe results.**
- Zenodo metadata confirms `access_right: open`, license `cc-by-4.0`. [verified]
- HF mirror confirms ~28k-row size and YOLO bbox label format. [verified]
- **Bone Resorption (`BON`) is a bbox class, not a per-tooth keypoint or per-tooth bone-level polyline.** That's the annotation-shape gap relative to our use case: a bbox over a region of bone loss is not a CEJ-mesial / CEJ-distal / ABCL-mesial / ABCL-distal landmark set.
- **Modality is panoramic.** Chairside dental-rad-cli intent is IOPA + BW.

**Fit-to-use-case.** Two roles:
- **Not a v0 training source** for the CEJ head (modality mismatch + annotation-shape mismatch).
- **Useful as an evaluation tier** once an IOPA-trained model is shipped — can the model detect bone resorption regions on panoramic radiographs the same office shoots? Useful as out-of-distribution generalization sanity-check.
- Furcation lesion is an annotated class — the only public dataset surveyed with a dedicated furcation class. Future v1+ feature.

**Action items.**
- Hold as evaluation source. No v0 integration. Note as the strongest CC-BY 4.0 panoramic-tier candidate when a panoramic feature ships.

---

### 3.3 `BRAR-anchored multimodal dataset` — Figshare 30155974.v3

| Property               | Value                                                       |
|------------------------|-------------------------------------------------------------|
| Host                   | Figshare                                                    |
| Record                 | https://figshare.com/articles/dataset/BRAR-anchored_multimodal_dataset/30155974/3 (DOI `10.6084/m9.figshare.30155974`) |
| License                | [probe pending] — Figshare default is CC-BY; not directly verified this session (page fetch failed 403) |
| Access status          | [probe pending — likely open per author description] "freely accessible to researchers worldwide" |
| Patient count          | 1,104                                                       |
| Modality               | Panoramic radiographs (JPG) + CSV demographics + tooth-level scores |
| Format                 | JPG images + CSV labels                                     |
| Annotation             | **BRAR = (BL/RL)/Age** — per-tooth Bone Resorption Age Ratio, calibrated to chronological age. Grades into 3 severity levels. |
| Validators             | Three trained periodontists                                 |
| Source                 | Shanghai Stomatological Hospital                             |
| Published              | 2026 (Sci Data 13:89)                                       |

**Probe results.**
- Description states "uploaded to Figshare, making it freely accessible to researchers worldwide." [from-paper]
- Direct fetch of the Figshare record returned 403 this session — Figshare may rate-limit unauthenticated fetches. Re-probe via authenticated Figshare API once access is desired.
- The annotation is **per-tooth classification** (BRAR severity grade), not per-tooth keypoints or polylines.
- BRAR formula uses the same CEJ-to-alveolar-crest distance our pipeline computes, so the *measurement protocol* is aligned — but the dataset ships the *output* of that measurement (the score), not the *inputs* (CEJ coords + alveolar-crest coords).

**Fit-to-use-case.** Two roles:
- **Not a v0 training source** for the CEJ head (no keypoint annotations + modality mismatch).
- **Useful for downstream BRAR validation** — once our pipeline computes per-tooth bone loss, we could re-score panoramic radiographs and compare against BRAR ground truth. Validation-tier signal.

**Action items.**
- Probe-pending. File a follow-up if/when panoramic feature ships or BRAR-validation makes sense.

---

## 4. Rejected candidates (with reason)

### 4.1 PRAD-10K — `nkicsl/PRAD`

| Property         | Value                                                          |
|------------------|----------------------------------------------------------------|
| Host             | GitHub `nkicsl/PRAD` + linked institutional download           |
| License          | **CC-BY-NC** (per GitHub repo) [verified]                       |
| Access           | **Gated** — formal email application, ~14 working days [verified] |
| Total images     | 10,000 periapical                                              |
| Approved download | 5,000 images + masks (subset of full corpus)                  |
| Annotation       | Pixel-level segmentation                                       |
| 9 classes        | Tooth, Alveolar Bone, Pulp, Root Canal Filling, Denture Crown, Dental Fillings, Implant, Orthodontic Devices, Apical Periodontitis [verified] |
| Paper            | MICCAI 2025, arXiv 2504.07760                                  |

**Why rejected.**
- **CEJ is NOT among the 9 annotated classes.** The "Alveolar Bone" class is a full-region segmentation mask, not a CEJ-to-ABCL line per tooth. The supervision signal we need (CEJ-mesial / CEJ-distal as paired landmarks per tooth) does not exist in this dataset.
- Gated email application = >2 week round-trip.
- CC-BY-NC blocks any commercial use.
- *Salvage value:* the "Alveolar Bone" pixel mask could be useful as a weak prior for tooth-vs-bone region segmentation — but not as the per-tooth keypoint supervision we need.

### 4.2 DentalX — Surrey internal

| Property      | Value                                                            |
|---------------|------------------------------------------------------------------|
| Host          | None public — Surrey internal                                    |
| License       | Not specified [verified absent]                                   |
| Access        | **No public download** — University of Surrey Ethics gating       |
| Images        | 10,121 detection (PA+BW) + 4,544 segmentation (1,556 PA + 3,988 BW) |
| Annotation    | 20 disease types incl. "four stages of bone loss" + 6 anatomy types |
| Code          | https://github.com/zhiqin1998/DentYOLX (code only, no data)       |

**Why rejected.** No public download. Internal/proprietary corpus from 3
dental practices. The "four stages of bone loss" is the most directly
disease-aligned ontology in the entire survey but **none of it ships
publicly.** Worth a corresponding-author email if v1.0 horizon expands;
hopeless for v0.

### 4.3 Tufts Dental Database — `tdd.ece.tufts.edu`

| Property      | Value                                                          |
|---------------|----------------------------------------------------------------|
| Host          | Tufts University institutional portal                          |
| License       | Institutional (researcher-only)                                |
| Access        | **Request form gated** [verified — page promises form access only] |
| Images        | 1,000 panoramic                                                |
| Modality      | Panoramic                                                      |
| Annotation    | Abnormality masks + teeth masks + eye-tracker gaze maps + radiographic text descriptions |

**Why rejected.** Wrong modality (panoramic) + gated access + no per-tooth
CEJ keypoints. Useful for completely different research direction
(eye-tracking + report generation) — irrelevant to our chairside pipeline.

### 4.4 DENTEX — `ibrahimhamamci/DENTEX` (HuggingFace + Grand Challenge)

| Property      | Value                                                          |
|---------------|----------------------------------------------------------------|
| Host          | HuggingFace + dentex.grand-challenge.org                       |
| License       | CC-BY (per HF mirror) [probe pending]                           |
| Access        | **Open** — HF dataset loader works                              |
| Images        | 1,005 panoramic (705 train / 50 val / 250 test)                |
| Annotation    | Bbox + tooth enumeration (FDI) + 4 diagnosis classes: caries, deep caries, periapical lesions, impacted teeth |

**Why rejected.** Panoramic-only. **Bone loss is NOT among the 4 diagnosis
classes.** No CEJ keypoints. Misses our use case on both modality and
annotation axes. Already cataloged in the caries deep-dive as a panoramic-tier
secondary candidate for caries; same status here for perio.

### 4.5 AI-Dentify / HUNT4

| Property      | Value                                                          |
|---------------|----------------------------------------------------------------|
| Host          | NTNU HUNT Research Centre — not redistributed                  |
| License       | Not applicable — restricted to Regional Ethical Committee + protocol approval |
| Access        | **Gated months to years** — REC approval                        |
| Images        | 13,887 bitewings                                               |
| Annotation    | 4 caries classes (enamel / dentine / secondary / unknown)      |

**Why rejected.** Caries-only annotation + IRB-gated access. Cataloged in the
caries deep-dive; included here for completeness because Norway-tier dataset
might in theory carry periodontal labels — it doesn't.

### 4.6 DenPAR v3 — Zenodo 16645076 (the current upstream)

| Property      | Value                                                          |
|---------------|----------------------------------------------------------------|
| Host          | Zenodo                                                         |
| Record        | https://zenodo.org/records/16645076 (v3) ; concept `10.5281/zenodo.13998618` |
| License       | **CC-BY 4.0** [verified — Zenodo API]                          |
| Access        | **`access_right: open`** [verified]                            |
| File          | `DenPAR Radiographs Dataset.zip` (141 MB)                      |
| Images        | 1,000 IOPA                                                     |
| Annotation    | Tooth-segmentation masks + CEJ points + apex points + alveolar crestal bone level lines + per-tooth metadata (age, sex, FDI). Format: PNG masks + COCO-style JSON (per published description). [probe pending — actual JSON schema not opened this session] |

**Why "rejected" here is misleading — it's our current upstream.**
- DenPAR v3 IS the pipeline's current training corpus.
- The reported failure mode is **annotation-shape**: CEJ points appear to ship
  as a flat 2-D point list per image (not as a COCO-keypoints array bound to
  per-tooth bboxes), forcing the adapter to do heuristic bbox-containment +
  nearest-center pairing, which produces clean supervision for only ~42% of
  teeth.
- **Critical probe outstanding:** open the actual `annotations.json` (or
  whatever file lives inside the 141 MB zip) and confirm whether CEJ
  coordinates are stored as COCO `keypoints: [x1,y1,v1,x2,y2,v2,...]` per
  tooth (in which case the current adapter is wrong and there's a free fix)
  OR as a separate flat `points.json` (in which case the academic doc's
  Path C — rewrite the CEJ assignment adapter — is the right move).
- Worth running the same `scripts/_probe_*` style script we used for Renielaz
  before declaring the adapter the bug. The Renielaz lesson applies in
  reverse: don't assume the upstream is correctly shaped without inspecting.

### 4.7 Mendeley dataset family

| Mendeley DOI         | Title                                          | Why rejected                                      |
|----------------------|------------------------------------------------|---------------------------------------------------|
| `yt8f2zzfpt`         | Segmented dental periapical X-ray for periapical disease | 929 PA, **CC-BY-NC**, binary healthy/diseased only — no granular bone loss / CEJ |
| `kx52tk2ddj`         | Panoramic radiographs with periapical lesions  | Panoramic-only; periapical-lesion classification, no bone-level keypoints |
| `ccw5mvg69r`         | OPG for Kennedy classification                 | Panoramic; periodontally-compromised-tooth is one of 3 classes but annotation form unclear; not keypoint-based |
| `7xgzy69fw2`         | Dental periapical for osteoporosis             | 13 subjects only; osteoporosis classification (3 classes), not periodontal |
| `4fbdxs7s7w`         | Dental caries in bitewing radiographs (100 BW) | 100 BW, caries-only (already cataloged in caries deep-dive) |
| `c4hhrkxytw`         | Dental OPG XRAY                                | Panoramic, multi-purpose |
| `hxt48yk462`         | Panoramic Dental X-rays With Segmented Mandibles | 232 panoramic, mandible segmentation only (no perio) |
| `mdvs6mjgf2`         | Annotated OPG for fillings/prostheses/etc.     | Panoramic, restorative-only |
| `9d8mcyp284`         | Panoramic for apicoectomy diagnosis            | Panoramic, surgery-need label only |
| `73n3kz2k4k`         | Panoramic Dental Xray Dataset (107+60 imgs)    | Panoramic, tooth segmentation |

None of these provide per-tooth grouped CEJ / bone-level keypoints on IOPA or
BW modality. Mendeley is well-stocked for panoramic and for caries, sparse for
periodontal-on-intraoral.

### 4.8 HuggingFace dental datasets

Sweep of all HF datasets matching `dental / tooth / radiograph / xray / perio`
returned 16 entries; the only radiograph-annotation dataset is the Zenodo
15487430 mirror (`ismaelportog`, already covered as fallback). Others are:
- `lambdaeranga/dental-radiology` — 700 CBCT VQA samples (text-anchored, not bbox/keypoint, license unspecified) [verified]
- `lambdaeranga/dental-samples` — 3,288 intraoral photos with gingivitis-severity text captions; no bone loss [verified]
- `darmasrmz/radiograph_dental` — 1,630 panoramic with 17 classes including Bone Resorption bbox; MIT license; but panoramic + no CEJ [verified]
- `tamnvcc/dental_caries_detection_v1` — caries-only [verified]
- `naazimsnh02/dentalgemma-*` — LLM-text-pair datasets, not vision corpus
- `JBJoyce/DENTAL_CLICK` — **audio clicks** dataset (mislabeled name); not vision at all
- `electricsheepafrica/oral-health-dental-disease` — 30k rows, tabular WHO data, not radiographs
- `reza362/dental-xray-caries` — 20 rows, caries-only
- `sudhakark4227/dental-xray-dataset` — 48 downloads, minimal metadata
- `Wildstash/dental-treatment-planning-2.5k`, `jonathankang/dental_QA`,
  `TachyHealth/ADA_Dental_Code_to_SBS_V2`, `shangzx/...action-recognition`,
  `AbFiras/Dental_Jaw_Captions_Data` — non-vision or non-radiograph
- `naazimsnh02/dentalgemma-vqa` — VQA, not vision-annotation

**Net:** HuggingFace surface is sparse for annotated periodontal radiograph
datasets. The relevant entry (`ismaelportog`) is a panoramic mirror already
covered. No further HF candidates.

### 4.9 DatasetNinja

- `datasetninja.com/dentalai` — 2,495 images, 4 classes (tooth/caries/cavity/crack), CC-BY 4.0 [verified]. **No bone loss class.** Already cataloged in caries deep-dive as DentalAI/Germanov BW source for the binary fallback path.
- No other dental dataset on DatasetNinja with bone loss annotations.

### 4.10 Roboflow Universe

Roboflow Universe pages are protected by Cloudflare anti-bot and could not be
direct-probed this session; entries below are from search-snippet evidence and
must be probed via the Roboflow SDK before adoption (the Renielaz lesson is
in scope here above all places).

| Project                                                  | Workspace                | Reported size | Modality (claimed) | Classes (snippet) | Why rejected (so far) |
|----------------------------------------------------------|--------------------------|---------------|--------------------|-------------------|----------------------|
| `training-horizontal-bone-loss`                          | `periodontal-bone-loss`  | 136 images    | Panoramic          | Mild / Moderate / Severe horizontal bone loss (3 classes) | Panoramic + tiny + image-level severity only — no per-tooth keypoints |
| `periodontal-bone-loss-detection`                        | `periodontal-bone-loss`  | unverified    | unverified         | Bone Loss / Bone Loss Crown / various | [probe pending] — likely bbox/classification, modality unclear |
| `periodontal-bone-loss` (alt slug, model v8)             | `pbl-zev2s`              | unverified    | unverified         | unverified        | [probe pending] |
| Periodontitis-class projects (multiple)                  | various                  | unverified    | mixed              | unverified        | [probe pending] — most appear panoramic |

**Action.** The Renielaz lesson says *probe before trust*. Each Roboflow
candidate must be inspected with the Roboflow SDK to fetch `data.yaml` +
sample annotation files before adoption. Schedule as a 30-minute follow-up:
- `roboflow.Workspace("periodontal-bone-loss").project("periodontal-bone-loss-detection").version(<latest>).download()` (or via the existing `scripts/_probe_roboflow_*.py` retargeted)
- Same for `pbl-zev2s/periodontal-bone-loss`
- Same for top-3 Roboflow Universe search hits for `class:periodontitis` and `class:bone-loss`

Until probed, **none of the Roboflow Universe candidates can be relied on
for v0 training.** Pre-probe verdict is "almost certainly modality- or
annotation-mismatched given snippet evidence" but the verdict is not
defensible until the probe runs.

### 4.11 Other surveyed sources that returned nothing usable

- **Kaggle:** searched periodontal / bone-loss / dental — only panoramic-tier datasets surfaced (`daverattan/dental-xrary-tfrecords`, `lokisilvres/dental-disease-panoramic-detection-dataset`, `truthisneverlinear/dentex-challenge-2023`). All panoramic, no per-tooth CEJ/bone-level keypoints.
- **OpenML:** no dental imaging datasets matching the constraint set.
- **DDS-Hub (NIH/NIDCR head & neck imaging):** institutional-aggregator; access is per-study-request, modality varies, not v0-tractable.
- **NHANES dental survey:** clinical periodontal probing data only; no radiograph corpus tied to the survey is publicly redistributed.
- **HUNT4:** see §4.5.
- **Frontiers 2024 Lin et al. (140 PA, CEJ + ALC polygons, RMSE < 0.09):** access-pending corresponding-author request; no public Zenodo / Mendeley deposit. Documented in the academic-survey companion doc.
- **Lee et al. 2025 (Columbia, 550 BW):** not public.

---

## 5. Probe results — what I was able to inspect this session

| Source                              | Probed via                                  | Result                                                                                                |
|-------------------------------------|---------------------------------------------|-------------------------------------------------------------------------------------------------------|
| `zenodo.org/records/14711842` (perio-KPT v1) | Zenodo API + page                         | `access_right: restricted`, license `cc-by-nc-sa-2.0`, v1.0 [verified]                                |
| `zenodo.org/records/17272200` (perio-KPT v2) | Page                                        | Restricted, 37.4 GB, 192 IOPA + 3,588 panoramic auxiliary + 15 ext validation [verified]              |
| `zenodo.org/records/16645076` (DenPAR v3)    | Zenodo API + page                         | `access_right: open`, license `cc-by-4.0`, 141 MB zip. Internal JSON schema [probe pending — open the zip]. |
| `zenodo.org/records/15487430` (panoramic 14-class) | Page                                  | Open, CC-BY 4.0, YOLO 14 classes incl. `BON`+`FUR` [verified]                                         |
| `huggingface.co/datasets/ismaelportog/...`  | HF page                                   | 27,884 rows, mirror of Zenodo 15487430, YOLO bbox [verified]                                          |
| `bit.ly/4hJ3aE7` (Banks supplementary)     | URL redirect resolution                   | Redirects to public Dropbox `BONE_LOSS_KPT_UPLOAD_FINAL.zip` [verified-link, contents-unverified]     |
| `github.com/Banksylel/Bone-Loss-Keypoint-Detection-Code` | Page                          | MIT license, YOLOv8-pose code, README references Zenodo 17272200 for data [verified]                  |
| `github.com/nkicsl/PRAD`                   | Page                                      | CC-BY-NC, gated, 5,000 of 10,000 images post-approval, 9 classes incl. Alveolar Bone (segmentation) but NOT CEJ [verified] |
| `nature.com/articles/s41597-025-06400-y` (BRAR paper) | Mixed (page fetch failed; snippets) | 1,104 patients, panoramic, BRAR scores per tooth, Figshare 30155974, "freely accessible" claim [probe pending] |
| `nature.com/articles/s41597-024-04306-9` (multi-modal STS) | Fetch failed                       | STS-Tooth: 4,000 panoramic + 148.4k CBCT, Zenodo 10.5281/zenodo.10597292 [from-search-snippets]       |
| `huggingface.co/datasets?search=dental`   | Page                                       | 16 dental-named HF datasets; none beyond the ismaelportog mirror are radiograph-annotation with bone loss [verified] |
| `data.mendeley.com/datasets/...` (5 records) | Page (where reachable)                  | All either wrong modality, wrong annotation type, or CC-BY-NC [verified per row]                      |
| `universe.roboflow.com/...`                | **Cloudflare-blocked** all fetches         | Cannot probe without SDK or browser auth; must follow up with `roboflow` Python SDK [blocked-fetch]   |

**Probe outstanding.** Reading the DenPAR v3 zip's actual annotation JSON
schema this session would have eliminated half the ambiguity in §4.6 — that
read is the highest-value 5-minute task before the next decision. Treat as
the first action item below.

---

## 6. Decision tree

```
                       ┌─────────────────────────────────────┐
                       │  Step 0 (5 min): open DenPAR v3 zip │
                       │  inspect actual annotation JSON     │
                       │  schema for CEJ keypoints           │
                       └──────────────┬──────────────────────┘
                                      │
                  ┌───────────────────┴────────────────────┐
                  │                                        │
        CEJ ships as per-tooth                   CEJ ships as flat 2-D
        COCO keypoints array                     point list (current
        (current adapter is wrong)               assumption)
                  │                                        │
                  ▼                                        ▼
       ─────────────────────────              ──────────────────────
       Path A1: fix the adapter               Step 1: file Zenodo
       to consume DenPAR's native             access request for
       per-tooth schema directly.             perio-KPT v2 (17272200)
       v0 reships in hours.                   AND license-tier check
       perio-KPT becomes optional             with Joseph re NC-SA
       v1+ augmentation.                                   │
       ─────────────────────────                ┌──────────┴──────────┐
                                                │                     │
                                          License accepted        License rejected
                                          + access granted        OR access denied
                                                │                     │
                                                ▼                     ▼
                                   Path A2: ship dual-corpus    Path B: implement
                                   training (DenPAR + perio-     the academic-survey
                                   KPT). Per-tooth supervision   companion doc's
                                   density ~doubles.             "Adapter Fix"
                                                                 fallback on DenPAR
                                                                 alone (§ Fallback
                                                                 of academic doc).
```

The Step-0 probe is the highest-leverage 5 minutes in this entire decision
tree. Two branches collapse based on its result.

---

## 7. Recommendation

**Order of operations.**

1. **Step 0 (5 min):** unzip DenPAR v3, look at the actual annotation JSON
   schema for the CEJ keypoints. The two paths in the decision tree differ
   dramatically based on this single probe result.

2. **Step 1 (parallel, ~5 min):** file the Zenodo access request for
   perio-KPT v2 (record 17272200). Round-trip is ~14 days; starting the
   clock now is cheap regardless of which decision branch fires.

3. **Step 1b (parallel, ~10 min):** probe `bit.ly/4hJ3aE7` →
   `BONE_LOSS_KPT_UPLOAD_FINAL.zip`. If the Dropbox link is openly
   downloadable, inspect the annotation `.txt` format (verify the
   11-keypoint-per-tooth claim with mesial-CEJ-first ordering) **without
   training on it**. This is schema-verification, not data-acquisition;
   the formal Zenodo access is still the canonical channel.

4. **Step 2 (Joseph-confirm, async):** CC-BY-NC-SA 2.0 license-tier check
   for perio-KPT. Personal-use plausibly compatible; weights inherit
   NC-SA viral effect. Decision input for whether to integrate perio-KPT
   even after access is granted.

5. **Step 3 (conditional on Step 0):**
   - If Step 0 reveals DenPAR ships per-tooth COCO keypoints → fix
     adapter, ship v0.5, declare perio-KPT a v1+ augmentation candidate.
   - If Step 0 confirms flat-point-list assumption → execute the
     academic-survey doc's "Adapter Fix" path (cluster flat points
     per-image by tooth bbox, reject teeth without exactly 2 CEJ points,
     train only on clean ~42% subset).

6. **Step 4 (long-horizon, no action this week):** when panoramic
   modality ships in v1+, integrate Zenodo 15487430 (CC-BY 4.0
   panoramic with 14-class `BON`+`FUR`) and BRAR Figshare 30155974
   (if license clears).

**Explicitly do not.**
- Don't pay for or sign IRB protocols for AI-Dentify/HUNT4, Tufts, DentalX,
  or PRAD-10K. None of them solve the per-tooth CEJ supervision problem
  the v0 actually has.
- Don't adopt any Roboflow Universe periodontal dataset without running
  `scripts/_probe_roboflow_*.py` first. The Renielaz failure mode is real.
- Don't trust the snippet evidence on the Roboflow `training-horizontal-bone-loss`
  dataset's 136-image / panoramic / mild-mod-severe shape until SDK-probed.
- Don't hand-label periapical or bitewing landmarks for v0. Time cost
  exceeds the adapter-fix path by 10x.

---

## 8. Open follow-ups

| ID  | Item                                                                                   | Owner   | Trigger                                                                |
|-----|----------------------------------------------------------------------------------------|---------|------------------------------------------------------------------------|
| D1  | Open DenPAR v3 zip; inspect actual CEJ keypoint annotation JSON schema                 | Pickles | Before any adapter-rewrite work                                        |
| D2  | File Zenodo access request for perio-KPT v2 (record 17272200)                          | Joseph  | This week — 14-day round-trip                                          |
| D3  | License-tier check: is CC-BY-NC-SA 2.0 acceptable for personal-use weights?            | Joseph  | Before downloading perio-KPT if access granted                         |
| D4  | Run `scripts/_probe_roboflow_*.py` against `periodontal-bone-loss/*` and `pbl-zev2s/*` | Pickles | Before any Roboflow Universe adoption decision                         |
| D5  | Probe `bit.ly/4hJ3aE7` Dropbox supplementary for schema-verification (read-only)       | Pickles | While Zenodo access pending; no training use                           |
| D6  | Probe Figshare 30155974.v3 via authenticated Figshare API for license confirmation     | Pickles | When panoramic-tier feature is on the horizon                          |
| D7  | If access denied or license rejected, execute academic-doc "Adapter Fix" path on DenPAR | Pickles | After D1 + D2 + D3 resolve                                            |
| D8  | Email corresponding authors of inaccessible CEJ papers (Lin 2024, Lee 2025) for data   | Joseph  | If both perio-KPT and adapter-fix paths fail                          |
| D9  | Cross-reference findings here against academic-survey companion doc; reconcile         | Joseph  | When making integration decision                                       |
| D10 | Hold Zenodo 15487430 (panoramic 14-class) as OOD evaluation source                     | Pickles | After v0.5 IOPA training ships                                         |

---

## 9. Sources

### Datasets evaluated

- [perio-KPT v1 — Zenodo 14711842](https://zenodo.org/records/14711842)
- [perio-KPT v2 — Zenodo 17272200](https://zenodo.org/records/17272200)
- [DenPAR v3 — Zenodo 16645076](https://zenodo.org/records/16645076)
- [Dataset for Automating Dental Condition Detection on Panoramic Radiographs — Zenodo 15487430](https://zenodo.org/records/15487430)
- [HF mirror of Zenodo 15487430 — ismaelportog/Panoramic_Radiographs_for_Dental_Condition](https://huggingface.co/datasets/ismaelportog/Panoramic_Radiographs_for_Dental_Condition)
- [BRAR-anchored multimodal dataset — Figshare 30155974.v3](https://doi.org/10.6084/m9.figshare.30155974.v3)
- [BRAR paper — Sci Data 13:89 2026](https://www.nature.com/articles/s41597-025-06400-y)
- [PRAD-10K — GitHub nkicsl/PRAD](https://github.com/nkicsl/PRAD)
- [PRAD-10K paper — MICCAI 2025 / arXiv 2504.07760](https://arxiv.org/abs/2504.07760)
- [Tufts Dental Database](https://tdd.ece.tufts.edu/)
- [DENTEX — HuggingFace](https://huggingface.co/datasets/ibrahimhamamci/DENTEX)
- [DentalX (DentYOLX code only) — GitHub zhiqin1998/DentYOLX](https://github.com/zhiqin1998/DentYOLX)
- [Mendeley `yt8f2zzfpt` — segmented periapical](https://data.mendeley.com/preview/yt8f2zzfpt)
- [Mendeley `kx52tk2ddj` — panoramic periapical lesions](https://data.mendeley.com/datasets/kx52tk2ddj/3)
- [Mendeley `ccw5mvg69r` — OPG Kennedy classification](https://data.mendeley.com/datasets/ccw5mvg69r/1)
- [Mendeley `7xgzy69fw2` — periapical osteoporosis](https://data.mendeley.com/datasets/7xgzy69fw2/1)
- [Mendeley `hxt48yk462` — panoramic mandible segmentation](https://data.mendeley.com/datasets/hxt48yk462/2)
- [Mendeley `mdvs6mjgf2` — OPG fillings/prostheses](https://data.mendeley.com/datasets/mdvs6mjgf2/1)
- [DatasetNinja DentalAI](https://datasetninja.com/dentalai)
- [Roboflow `periodontal-bone-loss/training-horizontal-bone-loss`](https://universe.roboflow.com/periodontal-bone-loss/training-horizontal-bone-loss)
- [Roboflow `periodontal-bone-loss/periodontal-bone-loss-detection`](https://universe.roboflow.com/periodontal-bone-loss/periodontal-bone-loss-detection)
- [Roboflow `pbl-zev2s/periodontal-bone-loss`](https://universe.roboflow.com/pbl-zev2s/periodontal-bone-loss)

### Code repositories

- [Banks et al. perio-KPT training code — GitHub Banksylel/Bone-Loss-Keypoint-Detection-Code](https://github.com/Banksylel/Bone-Loss-Keypoint-Detection-Code) — MIT-licensed
- [Wimalasiri DenPAR code (codes-only) — GitHub chathurawimalasiri/Intraoral-periapical-radiography-codes](https://github.com/chathurawimalasiri/Intraoral-periapical-radiography-codes)
- [Wimalasiri 2026 paper code — GitHub chathurawimalasiri/analysis-in-detecting-alveolar-bone-loss](https://github.com/chathurawimalasiri/analysis-in-detecting-alveolar-bone-loss)

### Papers

- [Banks et al. 2025 — Periodontal Bone Loss Analysis via Keypoint Detection (arXiv 2503.13477)](https://arxiv.org/html/2503.13477v1)
- [Wimalasiri et al. 2026 — AI-assisted radiographic analysis (arXiv 2506.20522 / Sci Rep)](https://arxiv.org/html/2506.20522v1)
- [Rasnayaka et al. 2025 — DenPAR paper (Sci Data)](https://www.nature.com/articles/s41597-025-05906-9)
- [PRAD-10K paper — arXiv 2504.07760](https://arxiv.org/abs/2504.07760)
- [Uribe et al. 2024 systematic review of public dental datasets](https://pmc.ncbi.nlm.nih.gov/articles/PMC11633071/)
- [Uribe et al. dental-datasets-itu curated list](https://github.com/sergiouribe/dental_datasets_itu/blob/main/AI_Dental_Datasets_List.md)

### Tooling references

- [Roboflow Python SDK](https://docs.roboflow.com/python)
- [Zenodo API documentation](https://developers.zenodo.org/)
- [Figshare REST API](https://docs.figshare.com/)
- [HuggingFace datasets library](https://huggingface.co/docs/datasets/index)

---

## 10. Schema-correctness probe template (Renielaz-lesson application)

For every adopted dataset, run the following probe **before** integrating
into the training pipeline:

1. Inspect the actual annotation file (`data.yaml` / `annotations.json` /
   `labels/*.txt`). Confirm the class list / field schema matches the
   description.
2. Spot-check 10 random sample annotations. Confirm:
   - Class strings are real dental terms, not description bullets, URLs, or
     LLM-generated filler (the Renielaz failure mode).
   - Per-tooth keypoints (if applicable) have the expected length and order.
   - Visibility flags (YOLO pose) or COCO `keypoints` field is populated
     correctly, not left as zeros.
3. Sample 5 random images. Confirm:
   - Modality matches description (cone-shape IOPA vs wing-shape BW vs full
     arch panoramic — these are visually unambiguous).
   - Annotations visibly align with the radiographic content.
4. Cross-reference the class list against the dataset's accompanying paper
   (where one exists). Description-paper mismatch is a red flag.

This probe is identical in form to the one that exposed Renielaz; running it
preemptively on every candidate is the lesson made operational. The cost of
the probe is ~5 minutes per dataset; the cost of skipping it is days of
debugging on contaminated data.
