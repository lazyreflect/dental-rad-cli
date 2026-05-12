# Perio / Bone-Loss Deep-Dive — Synthesis

**Date:** 2026-05-12
**Status:** Research only — no shipping decisions, no hand-labeling proposals.

This document synthesizes three parallel deep-dives:

- [`2026-05-12-perio-deep-dive-academic.md`](2026-05-12-perio-deep-dive-academic.md) — academic paper survey starting from Wimalasiri's Table 1 prior-work list
- [`2026-05-12-perio-deep-dive-datasets.md`](2026-05-12-perio-deep-dive-datasets.md) — public-dataset-repository survey (Roboflow / Zenodo / Mendeley / HuggingFace / Kaggle / DatasetNinja / GitHub)
- [`2026-05-12-perio-deep-dive-bw-and-arch.md`](2026-05-12-perio-deep-dive-bw-and-arch.md) — bitewing-specific literature + architectural alternatives

Trigger: the bone-loss head trained from Wimalasiri's pipeline on DenPAR v3 collapses CEJ keypoints on 35–60% of teeth at inference, hallucinates apex predictions on bitewings (apex cut off by definition), and is structurally PA-only. Wimalasiri is the sole source we're building on — a comparable aggressive survey of alternatives is what we did for caries and skipped for perio.

---

## TL;DR

**One concrete primary candidate emerged: Banks et al. 2025 "perio-KPT"** ([arxiv 2503.13477](https://arxiv.org/abs/2503.13477), [GitHub MIT](https://github.com/Banksylel/Bone-Loss-Keypoint-Detection-Code), [Zenodo v1](https://zenodo.org/records/14711842) / [v2](https://zenodo.org/records/17272200)). All three agents independently surfaced this as the standout. It ships:

- **192 IOPA + 3,588 panoramic auxiliary** with **per-tooth-grouped keypoints in YOLOv8-pose format with visibility flags** — exactly the annotation shape DenPAR v3 lacks
- Keypoint classes: CEJ-mesial / CEJ-distal / BL-mesial / BL-distal / RL-mesial / RL-distal / ARR (alveolar ridge resorption) / PLS (periodontal ligament space)
- **Rotated/oriented tooth bboxes** — addresses the tilt failure mode flagged earlier
- 3 tooth root-class labels (single / double / triple)
- **MIT-licensed code with pretrained weights** (Google Drive link in the repo)
- **Heuristic post-processing that snaps keypoints to tooth boundary** — directly addresses the "keypoint floats off tooth" failure mode

Blockers (verified): Zenodo access is `restricted` (login + institutional affiliation request required). Paper claims CC-BY 4.0 but Zenodo metadata shows CC-BY-NC-SA 2.0 Generic — license inconsistency to resolve before integrating. The MIT-licensed code itself is unblocked and reusable today.

**Convergent architectural recommendation across two independent agents:** if perio-KPT access is denied/delayed, pivot the CEJ head from paired-keypoint regression to **thin-line polyline segmentation** (the same approach v3 already uses for bone-crest). Both the academic survey (Lee/Kabir 2022, [arxiv 2109.12115](https://arxiv.org/abs/2109.12115)) and the BW+architectures survey (Family A — unit change to mm CEJ→bone-crest distance) converged on segmentation-not-points as the right pivot.

**Confirmed dead-ends:**

- HUNT4 / AI-Dentify (13,887 BWs) — **bone-loss labels do NOT exist**; only caries. Norwegian ethics-gated regardless. Stop wishing for it.
- Chen 2023 (8,000 PAs) — never released
- Lee 2025 Columbia (550 BWs), Ameli 2024 Alberta (2,582 PAs), Lin 2024 Taiwan (281 PAs), Alqaderi 2026 Tufts (1,063 PA+BW) — all private, author-request only
- All panoramic candidates — out of intraoral scope
- Wimalasiri's `%` bone-loss formula on BW — confirmed structurally invalid (apex required)

**Wimalasiri code has no LICENSE file** — legally ambiguous to reuse. The cleanroom-from-brief approach we've been doing is the right path.

---

## What changes vs the pre-survey conversation

Before this survey, the conversation was: "the CEJ head collapses, retrain with a better adapter on DenPAR v3." Each of the three failure modes I'd identified — CEJ collapse, apex hallucination on BW, tilt-induced bbox failure — survives an adapter rewrite alone. The survey surfaces three things that change the framing:

**1. Banks et al. 2025 already built the per-tooth-grouped keypoint head we need.** Their training code is MIT, the heuristic post-processing for keypoint-snap-to-tooth is published, and the only blocker for the data is a Zenodo access request. The "build it ourselves from DenPAR-v3 loose lists" path is no longer the only path.

**2. The Wimalasiri % formula doesn't transfer to BW. Period.** The literature has three families of solutions (mm-unit, polynomial-arch-fit, classifier-only). All sidestep the apex denominator. So the "fix CEJ on PA, ship to BW" path was never structurally viable.

**3. The CEJ-as-polyline-segmentation architecture is convergently recommended.** Two independent agents reached this from different starting points (academic survey via Lee/Kabir 2022; BW+arch survey via Lee/Columbia + the bone-polyline-already-shipping observation). Same conclusion: segmentation > keypoint for CEJ on this data shape.

---

## Convergent findings (all three spawns)

| Finding | Agents | Confidence |
|---|---|---|
| Banks et al. 2025 / perio-KPT is the standout alternative | All 3 | High |
| MIT-licensed companion code at `Banksylel/Bone-Loss-Keypoint-Detection-Code` | All 3 | Verified |
| Zenodo access is restricted (login + affiliation request) | Datasets + Academic | Verified via API |
| License inconsistency: paper claims CC-BY, Zenodo says CC-BY-NC-SA 2.0 Generic | All 3 | Verified |
| CEJ-as-line-segmentation is the right pivot if perio-KPT denied | Academic + BW/Arch | High |
| HUNT4 has NO bone-loss labels (caries only) | All 3 | Verified |
| No fully open + permissive + downloadable BW bone-loss dataset exists today | All 3 | High |
| Wimalasiri % formula is PA-specific, cannot port to BW | BW/Arch + Academic | High |

---

## Unique findings per spawn

### From the academic survey

- **Wimalasiri's GitHub repo has NO LICENSE file** — strict copyright default. Cleanroom reimplementation (what we're already doing) is the only legally clean reuse path.
- **DenPAR Sci Data 2025 companion paper** ([nature.com/articles/s41597-025-05906-9](https://www.nature.com/articles/s41597-025-05906-9)) may document the CEJ annotation methodology more carefully than the dataset itself. Worth re-reading — possibly the per-tooth grouping IS in there and our adapter is reading the wrong field.
- **Lee/Kabir 2022 architecture** — three segmentation networks (bone-area, tooth, CEJ-band). DSC > 0.91, RBL stage AUC 0.89-0.90. Data not released but architecture template applicable to DenPAR v3 alone.

### From the dataset survey

- **Zenodo 15487430** — 1,628 + 180 panoramic radiographs, **14 classes including BON (Bone Resorption) and FUR (Furcation Lesion)**, **CC-BY 4.0**, fully open. Wrong modality for v0 IOPA pipeline but useful as out-of-distribution evaluation if/when a panoramic path is added.
- **BRAR-anchored Figshare 30155974.v3** — 1,104-patient panoramic with per-tooth BRAR (Bone Resorption Ascending Rate) scores. Classification-grade only, no CEJ keypoints. Panoramic.
- **Roboflow Universe pages were Cloudflare-blocked** from direct fetch this session. Three candidate dataset names surfaced but unverified: `periodontal-bone-loss`, `pbl-zev2s`, `training-horizontal-bone-loss`. Require the existing `scripts/_probe_roboflow_*.py` SDK-based probe to verify — same lesson from Renielaz.
- **Banks's supplementary Dropbox link** (`bit.ly/4hJ3aE7` → `BONE_LOSS_KPT_UPLOAD_FINAL.zip`) — possibly openly downloadable. Worth probing for schema verification without consuming it for training (the Zenodo channel remains the canonical access path).

### From the BW + architectures survey

- **Three families of solutions to no-apex-on-BW:**
  - **Family A (Denti.AI / commercial path):** mm CEJ→bone-crest distance, AAP thresholds. Architecture stays keypoint-based; only the math at the head changes.
  - **Family B (Lee/Columbia 5-network):** polynomial fit across all teeth in one arch as the implicit reference axis. No apex needed.
  - **Family C (Erturk 2024):** YOLOv8m-cls image-level classifier emits AAP stage directly. No geometry. Eigen-CAM for explainability.
- **Hybrid architecture recommended:** Family A as primary measurement head, Family B's arch polynomial as a consistency check, Family C as a defensive sidecar (when geometry says "incomputable," sidecar still emits stage).
- **GeoSapiens** ([arxiv 2507.04710](https://arxiv.org/abs/2507.04710)) — Sapiens foundation model + LoRA + 3-patient few-shot = 93% SDR@2mm on dental landmarks (CBCT). MIT-licensed. Few-shot transferability is interesting if office-data labeling ever becomes feasible.
- **finetune-SAM** ([mazurowski-lab](https://github.com/mazurowski-lab/finetune-SAM)) — Apache 2.0 generic medical SAM/SAM2 adapter framework. No dental BW reference impl but adaptable.

---

## Confirmed dead-ends (consolidated)

| Dataset | Reason | Source |
|---|---|---|
| Chen 2023 (8,000 PA) | Never released, single-institution retrospective | Academic |
| HUNT4 / AI-Dentify (13,887 BW) | Caries-only labels; Norwegian ethics gate is moot | All 3 |
| Lee 2025 Columbia (550 BW) | Private, author-request only | Academic + Datasets |
| Ameli 2024 Alberta (2,582 PA) | Private, author-request only | Academic |
| Alqaderi 2026 Tufts (1,063 PA+BW) | Single-institution Axium EHR, redistribution unlikely | Academic + BW/Arch |
| Lin 2024 Taiwan (~281 PA) | Private, methodology reference only | Academic |
| AlGhaihab 2025 UNC (39 PA+BW) | Too small + private | Academic + BW/Arch |
| Erturk 2024 (Necmettin Erbakan) | Private, classifier-only methodology reference | BW/Arch |
| Akarsu 2026 (1,197 BWs) | No code/data release | BW/Arch |
| Tsoromokos 2022 (ACTA, 446 PA mand) | Private; sober baseline (ICC 0.601) | Academic |
| Tufts Dental Database | Panoramic-only, modality mismatch | All 3 |
| DENTEX (MICCAI 2023) | Panoramic, no bone-loss class | Academic + Datasets |
| All other panoramic candidates (Bayrakdar, Sunnetci, Ryu, Uzun Saylan, Chang, Kurt-Bayrakdar, Krois) | Out of intraoral modality scope | Academic |
| DentalAI / DatasetNinja | Mixed modality, binary caries only | Datasets |
| 10 surveyed Mendeley records | Modality / annotation mismatch | Datasets |
| 16 HuggingFace dental dataset sweep | No relevant new candidate | Datasets |

---

## Open candidates with action paths

### Tier 1 — Banks et al. 2025 / perio-KPT

**Path:** Zenodo access request (login + institutional affiliation form) for record [17272200](https://zenodo.org/records/17272200) (v2, 37.4 GB) and/or [14711842](https://zenodo.org/records/14711842) (v1, 33 GB). Round-trip ~14 days.

**Parallel:** Examine the MIT-licensed companion code at [Banksylel/Bone-Loss-Keypoint-Detection-Code](https://github.com/Banksylel/Bone-Loss-Keypoint-Detection-Code) immediately — no access barrier on the code. Specifically:

- Training config and ultralytics customizations for YOLOv8-pose
- Heuristic tooth-segmentation post-processing logic (this is the load-bearing technique for keypoint-snap-to-tooth)
- Annotation schema (verify the per-tooth grouping format)
- HRNet / DeepPose / RTMPose comparison harness — gives a benchmark to replicate

**Decision input needed before downloading:** the license inconsistency. Paper says CC-BY 4.0, Zenodo says CC-BY-NC-SA 2.0 Generic. CC-BY-NC-SA 2.0 means:
- Non-commercial use only (compatible with personal scope)
- Share-alike viral effect on derivative works (any downstream model weights inherit NC-SA terms)
- Cannot redistribute trained weights under MIT

Resolution paths: (a) email Banks et al. to clarify which license applies, (b) train on it under NC-SA terms and keep derived weights local-only.

### Tier 2 — Adapter fix on DenPAR v3 alone

If Tier 1 is blocked / declined / license-incompatible:

**The fix:** rewrite the CEJ adapter to enforce strict per-tooth pairing. Cluster v3's flat point list per-image by tooth bbox using anatomical priors (CEJ in upper-third of bbox, vertical-midline distance, bone-line-y-anchor from v3's polylines). Hungarian assignment so each bbox claims ≤2 points and each point goes to ≤1 bbox. Reject teeth that can't find exactly 2 valid CEJ points after assignment — train only on the clean ~42% subset with `visibility=0` on the rejected ones.

This was the "Path B+" we were already considering. The survey confirms it as a legitimate fallback rather than the only path.

### Tier 3 — Architectural pivot: CEJ as polyline segmentation

If Tier 1 blocked AND Tier 2 doesn't drop collapse rate sufficiently:

**Replace the keypoint head with a thin-line segmentation head** following Lee/Kabir 2022. Derive mesial/distal CEJ points from polyline endpoints intersected with tooth bbox. Symmetric with v3's existing bone-crest polyline head — drop-in reuse of the segmentation pipeline.

This works on DenPAR v3 alone (no new data needed); the loose 2-D point list can be densified into a short line segment per cluster.

### Tier 4 (panoramic, deferred) — Zenodo 15487430

**Path:** open download, CC-BY 4.0, no access barriers. Hold for OOD evaluation when panoramic modality ever comes online. Not actionable for IOPA/BW work.

### Tier 5 (auxiliary segmentation pretraining) — PRAD-10K

**Path:** application to aics@nankai.edu.cn, ~14 working days. CC-BY-NC 4.0. 10,000 PAs with pixel-level segmentation of 9 anatomical classes including alveolar bone region. Wrong shape for CEJ keypoints but **strong auxiliary signal for tooth + bone segmentation heads** — could improve the segmentation-only architecture path (Tier 3).

### Tier 6 (BW dataset existence check) — Roboflow Universe SDK probe

**Path:** run `scripts/_probe_roboflow_*.py` against `periodontal-bone-loss/*` and `pbl-zev2s/*` candidate names. Cloudflare blocked direct page fetches in this session; the SDK route uses the Roboflow API and bypasses the block.

Roboflow data quality is unverified until probed — Renielaz lesson applies. Probably won't surface usable BW datasets but cheap to verify.

---

## Decision tree

```
                     ┌─────────────────────────────────────────┐
                     │ Decide: pursue perio-KPT integration?   │
                     │ (file access request + license check)   │
                     └──────────────────┬──────────────────────┘
                                        │
                       ┌────────────────┴───────────────┐
                       │                                │
              Pursue (file request,                Skip / decline
              ~14-day round-trip)
                       │                                │
                       ▼                                ▼
        ┌──────────────────────────┐    ┌──────────────────────────┐
        │ Examine MIT-licensed     │    │ Tier 2 — DenPAR adapter   │
        │ companion code today.    │    │ fix (filter to 2-CEJ-pt   │
        │ Train heuristic post-    │    │ teeth, Hungarian assign,  │
        │ processing on DenPAR.    │    │ visibility=0 elsewhere)   │
        │ Wait for Zenodo access.  │    └────────────┬─────────────┘
        └────────────┬─────────────┘                 │
                     │                               │
        Access granted │  Access denied/             Drop in collapse rate?
                       │  license incompatible
                       ▼                  ▼               │
            ┌──────────────────┐  ┌────────────────┐   Yes — refine adapter
            │ Co-train CEJ     │  │ Tier 3 —        │   No — go to Tier 3
            │ head on DenPAR + │  │ CEJ-as-polyline-│
            │ perio-KPT.       │  │ segmentation    │
            │ Use Banks's      │  │ (Lee/Kabir 2022 │
            │ keypoint-snap-   │  │ template).       │
            │ to-tooth post-   │  │ Works on DenPAR │
            │ processing.      │  │ v3 alone.        │
            └──────────────────┘  └─────────────────┘
                                            │
                                            │
                                    ┌───────┴────────┐
                                    │ For BW mode:   │
                                    │ separate pipe- │
                                    │ line. CEJ +    │
                                    │ bone polyline  │
                                    │ segmentation + │
                                    │ mm calibration │
                                    │ + Erturk-cls   │
                                    │ sidecar.       │
                                    └────────────────┘
```

---

## Action set (research-only, no shipping decisions)

The three spawns proposed slightly different "next actions." Reconciling:

1. **Examine Banks's MIT code immediately.** Zero access barriers. Reveals whether their heuristic post-processing alone could be transplanted onto the existing DenPAR-trained model, AND reveals the exact annotation schema we'd want to replicate in our own adapter.

2. **File Zenodo access request for perio-KPT v2** (record 17272200). 5 minutes. 14-day round-trip. Independent of any other decision.

3. **Probe DenPAR's full annotation schema (re-verify).** The dataset spawn proposed unzipping DenPAR v3 and inspecting the actual CEJ JSON shape. We've already done this work (loose 2-D point lists, confirmed by adapter docstring and corpus analysis). Re-confirm one more time before retraining; possibly read the **DenPAR Sci Data 2025 companion paper** (Rasnayaka et al.) which may document a per-tooth grouping convention we've missed.

4. **Read the Banks paper text for the keypoint-to-tooth-snap algorithm.** The "snap predicted keypoint to nearest tooth segmentation mask boundary" is the post-processing technique that addresses the CEJ-floats-off-tooth problem. This is a pure inference-time change with no retraining required.

5. **(Optional, cheap)** SDK-probe Roboflow Universe candidates `periodontal-bone-loss/*` and `pbl-zev2s/*`. Renielaz lesson — never trust description alone.

6. **(Optional, longer-horizon)** PRAD-10K access request to aics@nankai.edu.cn if tooth + bone segmentation heads need more pretraining data.

7. **(Optional, deferred)** Hold Zenodo 15487430 (panoramic, CC-BY 4.0, BON + FUR classes) for if/when panoramic modality is added to the pipeline.

---

## What this survey does NOT resolve

- Whether the resulting pipeline should be **PA-only**, **BW-only**, **both**, or **modality-detected-and-routed**. That's a scope question, not a research-availability question.
- Whether to do hand-labeling — explicitly out of scope per current instruction; the survey is consistent with that.
- Whether the architecture pivots (keypoint → polyline segmentation) should happen at the same time as the data swap (DenPAR → DenPAR+perio-KPT) or be sequenced.
- Licensing compatibility between perio-KPT's CC-BY-NC-SA 2.0 and any future weight release — needs the author email AND a license-tier decision before integration.

---

## Reference companion documents

- [`2026-05-12-perio-deep-dive-academic.md`](2026-05-12-perio-deep-dive-academic.md) (~550 lines) — academic paper survey
- [`2026-05-12-perio-deep-dive-datasets.md`](2026-05-12-perio-deep-dive-datasets.md) (~620 lines) — public-dataset-repository survey
- [`2026-05-12-perio-deep-dive-bw-and-arch.md`](2026-05-12-perio-deep-dive-bw-and-arch.md) (~780 lines) — BW literature + architectural alternatives

Cross-reference against [`2026-05-11-caries-v0.5-paths-deep-dive.md`](2026-05-11-caries-v0.5-paths-deep-dive.md) which is the template — same intensity, same probe-before-trust discipline.

---

## Sources (consolidated; each child doc has its own full list)

**Primary candidate:**

- [Banks et al. 2025 arxiv 2503.13477](https://arxiv.org/abs/2503.13477)
- [Banks GitHub MIT](https://github.com/Banksylel/Bone-Loss-Keypoint-Detection-Code)
- [perio-KPT v1 Zenodo 14711842](https://zenodo.org/records/14711842)
- [perio-KPT v2 Zenodo 17272200](https://zenodo.org/records/17272200)

**Current upstream:**

- [Wimalasiri et al. 2026 arxiv 2506.20522](https://arxiv.org/abs/2506.20522)
- [DenPAR v3 Zenodo 16645076](https://zenodo.org/records/16645076)
- [DenPAR Scientific Data 2025 companion paper](https://www.nature.com/articles/s41597-025-05906-9)

**Architectural alternatives:**

- [Lee/Kabir/Jiang 2022 arxiv 2109.12115](https://arxiv.org/abs/2109.12115) — CEJ as thin-line segmentation
- [GeoSapiens arxiv 2507.04710](https://arxiv.org/abs/2507.04710) — Sapiens foundation model + LoRA + few-shot
- [finetune-SAM (mazurowski-lab)](https://github.com/mazurowski-lab/finetune-SAM) — SAM/SAM2 medical adapter

**Useful auxiliary datasets:**

- [PRAD-10K (arxiv 2504.07760)](https://arxiv.org/abs/2504.07760) — 10K PAs, segmentation pretraining
- [Zenodo 15487430](https://zenodo.org/records/15487430) — panoramic, BON+FUR classes, CC-BY 4.0
- [Figshare 30155974.v3](https://figshare.com/articles/dataset/Multimodal_dataset_for_dental_imaging/30155974) — BRAR-anchored panoramic

**Survey + catalog references:**

- [Uribe et al. 2024 — Publicly Available Dental Image Datasets (J Dent Res, PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11633071/)
- [sergiouribe/dental_datasets_itu (GitHub catalog)](https://github.com/sergiouribe/dental_datasets_itu)

**Confirmed inaccessible:**

- [HUNT4 / AI-Dentify BMC](https://link.springer.com/article/10.1186/s12903-024-04120-0)
- [Lee 2025 Columbia BMC](https://link.springer.com/article/10.1186/s12903-025-05677-0)
- [Ameli 2024 Alberta Frontiers](https://www.frontiersin.org/journals/dental-medicine/articles/10.3389/fdmed.2024.1479380/full)
- [Lin 2024 Taiwan MDPI](https://www.mdpi.com/2075-4418/14/15/1687)
- [Alqaderi 2026 Tufts medRxiv (403)](https://www.medrxiv.org/content/10.64898/2026.04.12.26350726v1.full.pdf)
