# State of the Work — dental-rad-cli — 2026-05-13

**Audience:** the next session picking up this project (also an external review reader). Read this end-to-end before any code.

**Revisions:** v2 (2026-05-12 evening) incorporates the Joseph-conversation reframes: Family A as primary math head, apex head deletion, mm calibration via tooth-class priors, parallel `dental-tooth-numbering` substrate work, Joseph annotation deferred, Q5 dissolved into post-retrain empirical review. Earlier external-review corrections (y-band clustering, 30-px buffer, new metrics) carried forward unchanged.

## TL;DR

We have two measured baselines on held-out test data. The CEJ keypoint head fails on ~30% of teeth. The decided next move is an architectural pivot from Keypoint R-CNN to **polyline segmentation** for the CEJ landmark, on pickles (RTX 4090) over Tailscale SSH. The polyline supervision is built via **y-band clustering** of the loose DenPAR v3 CEJ point list (NOT bbox-anchored pairing — see "Three structural corrections" below).

| Metric | Value | Lower/Higher better | Notes |
|---|---|---|---|
| `cej_collapse_rate` | **0.3051** (pickles CUDA) / 0.3071 (Mac MPS) | lower | Bimodal failure: ~30% of teeth get collapsed prediction (<10 px between mesial and distal); the rest are ~150-250 px apart |
| `caries_map50` | **0.6478** (Baasils val n=58) | higher | At-par with Salehizeinabadi 2025; deep AP50 = 0.85 beats their 0.80 |
| `caries_map50_95` | 0.3858 | higher | |

The cross-host drift (0.3051 vs 0.3071) is fp rounding noise. Pin **pickles-CUDA** as the canonical eval host going forward — eval runs in 9 sec there vs 48 sec on Mac MPS vs 219 sec on pickles CPU.

**Architectural reframes locked in v2 (Joseph conversation 2026-05-12):**

- **Family A (apex-free mm CEJ→bone-crest math) is the primary bone-loss math head for BOTH PA and BW.** Clinical rationale: apex frequently misses on PAs due to operator positioning; retakes increase radiation dose. Apex-free math means diagnostic works on the radiograph you already took. Single regime simpler than dual-mode.
- **Apex head deleted from the pipeline.** No longer load-bearing. Removes one source of bone-loss-math error (apex predictions hugging bbox-top → ±20% bias on % math) and one model from the architecture diagram.
- **Px→mm calibration** becomes a new small dependency. Per-tooth-class anatomical priors (published mean tooth dimensions). v0 uses image-region tag or coarse aspect-ratio heuristic; v0.5+ consumes proper FDI numbering from the parallel substrate work.
- **Parallel substrate work in flight:** `~/repos/work/dental-tooth-numbering/` (new repo). Phase 1 (research + architecture jam) running in a sister Claude session. Will provide FDI/Universal tooth numbering as a workspace substrate consumed by dental-rad-cli (immediate), NoteBrusher, DDE, curve-genie, eob-ingest. Does NOT block polyline pivot — independent training, data, eval.
- **Joseph annotation deferred for v0.** Eval anchors entirely to DenPAR test split. Office-distribution validation revisited after polyline architecture is locked.

**First action for next session:** write `scripts/eval_cej_polyline.py` with the three real metrics (per-site y-error, CEJ-band IoU, polyline-degenerate rate), report in mm using per-tooth-class calibration. Run on current Keypoint R-CNN outputs to establish the baseline number that the polyline pivot will be measured against. ~200 lines.

## Three structural corrections from external review (load-bearing — carried from v1)

The v1 draft of this handoff had three flaws an external Claude reviewer caught. The corrections shape the polyline pivot. **All three are baked into this revision.**

### Correction 1 — adapter must use y-band clustering, not bbox containment

The v0 draft said: *"1 CEJ pt on a tooth → bbox-anchored extrapolation using the median CEJ-y-fraction prior."* That reintroduces the bbox-pairing problem through the back door — it requires knowing which tooth bbox owns each CEJ point, which is exactly the noisy heuristic we wanted to escape.

**Corrected approach:** For each image, take the flat `CEJ_Points` list as-is. Cluster points by y-proximity (within ~30 px y of each other) AND reasonable x-proximity (within ~200 px x of nearest neighbor, to avoid linking across image gaps). Connect each cluster's points in x-order into a polyline. Buffer to a band. **No bbox lookup at training time.** Per-tooth assignment happens once at inference, deterministically, when the predicted polyline intersects each tooth bbox's left/right edges.

This is the only adapter shape that delivers the "polyline pivot is data-bottleneck-free" claim. The previous draft's version was structurally a regression to the noisy-pairing regime under a new name.

Anatomical justification (Joseph-confirmed via separate conversation): CEJ is anatomically a **continuous band across all teeth in an arch**, not isolated per-tooth landmarks. Multi-tooth PA → typically ONE polyline across all teeth. BW showing both arches → TWO polylines (upper + lower) clustered separately by y. **CEJ visibility is symmetric on BW vs PA** (Joseph confirmed 2026-05-12) — no modality-specific weighting needed in training.

### Correction 2 — buffer to 30 px, not 2 px

The v0 draft specified "2-px line." That's sub-pixel at YOLOv8x-seg's proto resolution. The proto basis is `imgsz / 4` — at `imgsz=640` it's `160×160`, so a 2-px line in image space is `0.5 px` in proto space. The model literally can't represent it reliably.

Our existing bone-segmentation precedent in `src/dental_rad_cli/data/denpar_adapter.py:483` uses `_BONE_STRIP_HALF_WIDTH = 15` (30 px total band) and trains cleanly. **Use the same constant for CEJ:** `_CEJ_STRIP_HALF_WIDTH = 15`.

At inference time, if a thin curve is needed for endpoint extraction, skeletonize the predicted band (`cv2.ximgproc.thinning` or `scipy.ndimage.skeletonize`). Standard image processing.

### Correction 3 — `cej_collapse_rate` becomes trivial after polyline; need new primary metrics

The v0 draft said "collapse rate stays valid because two endpoints by construction = no collapse." That's true and useless. After polyline post-process intersects the predicted band with `bbox.x1` and `bbox.x2`, the endpoint separation is **roughly the bbox width** (80-300 px typical). Collapse threshold is 10 px. So `cej_collapse_rate` drops to ~0 by construction, regardless of polyline anatomical accuracy. The polyline could be 50 px coronal of the true CEJ and `cej_collapse_rate` would still be ~0.

The metric is "low by construction, not by quality." Headline metric needs to change.

**Primary metrics for the polyline pivot:**

| Metric | Definition | Target |
|---|---|---|
| **per-site y-error (mm)** (PRIMARY) | absolute y-error between predicted CEJ endpoint and ground-truth CEJ point, at `x=bbox.x1` (mesial) and `x=bbox.x2` (distal), in mm via per-tooth-class calibration. Computed only on teeth where ground-truth has BOTH CEJ points (~42% of DenPAR test). | median < 1.0 mm; p90 < 3.0 mm (anchored against AlGhaihab 2025 BW MAE 0.499 mm and AAP threshold of 2 mm) |
| **per-site y-error (px)** (sibling) | same as above in raw pixels. | median < 15 px; p90 < 40 px |
| **CEJ-band pixel IoU** | predicted CEJ mask vs dilated ground-truth band (same 30-px buffer as training). | > 0.50 (loose, matches Lee/Kabir 2022 territory) |
| **polyline-degenerate rate** | fraction of teeth where the predicted band fails to cross both vertical bbox edges. | < 5% |
| `cej_collapse_rate` (demoted) | as before. | ~0 — used as sanity check, not headline |

**These metrics need to be built BEFORE the retrain**, so we have a baseline number on the current Keypoint R-CNN outputs to compare against. The current 30.71% / 30.51% collapse rate gives no per-site y-error number — that requires recomputing against the same ground truth using the keypoint predictions instead of polyline endpoints.

Path: new eval script `scripts/eval_cej_polyline.py` (sibling to existing `eval_keypoint_cej.py` so the original baseline number stays stable). Roughly 200 lines.

## Architectural reframes — v2 detail

### Family A as primary math head for both modalities

**Decision:** mm CEJ→bone-crest with AAP/EFP thresholds (≥2 mm mild, ≥4 mm moderate, ≥6 mm severe) becomes the primary bone-loss math head for both PA and BW. Wimalasiri's `%` formula is retired.

**Clinical rationale (Joseph):** "Not needing the apex will be much more useful clinically. It gets missed sometimes with PA radiographs. Having to retake x-rays exposes the patient to more radiation. If we can accomplish what we need with just bitewings and anterior PAs then even less radiation."

**Engineering rationale:** Single math regime is simpler to ship, debug, and explain. AlGhaihab/Denti.AI 2025's dual-mode (% on PA, mm on BW) was data-availability-driven, not clinical-preference-driven. Apex-free is product-superior across the board.

**Published anchor:** AlGhaihab 2025 reports MAE 0.499 mm on BW with mm CEJ→bone-crest formulation. That's the performance ceiling target for any open replication; our v0 target is "approach 0.5 mm MAE" while staying clinically conservative on stage cutoffs.

### Apex head deleted

The apex prediction head is no longer load-bearing in the bone-loss math. It can be deleted from the pipeline entirely, or kept as a sidecar that's ignored downstream. Either way it stops contaminating output.

Deletion is the cleaner move. Reduces failure modes: apex predictions currently hug bbox-top edges, off by 30-100 px from real root tips, biasing the % math by ±20% even on clean PAs. Removing the head removes the bias.

### Pixel→mm calibration via per-tooth-class anatomical priors

Family A math needs mm, not pixels. Calibration approach:

- **Per-tooth anatomical prior.** Published mean tooth dimensions (e.g., max central incisor ≈ 22 mm long, max first molar MD width ≈ 10.5 mm). Per tooth: `px/mm = bbox_height_px / published_mean_height_mm`. Average across teeth in image for a stable per-image scale.

**Dependency on tooth-class identity:**

| Granularity needed | Method | When |
|---|---|---|
| Image-region (anterior PA / posterior BW left / etc.) | Mount metadata or user-supplied at inference | **v0 hack** — sufficient for first ship |
| Tooth-class (incisor / canine / premolar / molar) | Bbox aspect-ratio + position heuristic | v0 sanity check |
| Full FDI/Universal (#1-#32) | `dental-tooth-numbering` substrate (parallel work) | **v0.5+ — clean answer** |

v0 ships with image-region tag (chairside operator already knows the view from the PMS series notation). v0.5 swaps in proper FDI from the substrate work below.

### Parallel substrate work — `dental-tooth-numbering`

**New repo at `~/repos/work/dental-tooth-numbering/`** (does not exist yet at this writing; being stood up in a sister Claude session). Phase 1 (research + architecture jam with Joseph) is in flight.

**Why a workspace substrate, not a dental-rad-cli module:** Tooth numbering is the **join key** between visual data (radiographs, intraoral photos) and structured data (PMS records, claims, EOBs, treatment plans). Five validated downstream consumers in the workspace:

1. **dental-rad-cli** — mm calibration via per-tooth-class priors (immediate use case); per-tooth findings reporting; longitudinal comparison across visits
2. **NoteBrusher** — auto-fill tooth numbers in clinical notes from radiograph instead of operator dictation
3. **dental-decision-engine (DDE)** — every decision class references teeth numerically (denial triage, frequency validation, PA qualification)
4. **curve-genie** — Curve PMS stores everything tooth-numbered; round-trips need accurate numbering
5. **eob-ingest** — EOBs cite specific tooth numbers per line item; cross-reference with detected teeth enables auto-repair correlation

**Strict dependency direction:** `dental-tooth-numbering` is upstream substrate. Consumers (rad-cli, NoteBrusher, DDE, curve-genie, eob-ingest) import from it. It never imports from any consumer. CI-enforced.

**Polyline pivot does NOT block on this work.** Independent training, data, eval. v0 of polyline pivot uses the image-region calibration hack; v0.5 reads FDI from the substrate once it ships. Both projects can land in either order.

**Architectural decision deferred** to the sister session's interactive jam with Joseph. Four candidate paths (YOLO multi-class 32-way, two-stage detect-then-classify, detection + ordering heuristic, mount-aware two-stage). Don't pre-commit from this session.

## Where we are

**Repo state at HEAD** (`82ade54` on `main` at v1 write time; v2 doc commit follows):

- Eval scripts (`scripts/eval.sh`, `scripts/eval_keypoint_cej.py`, `scripts/eval_caries.py`) wired and working on both hosts. The CEJ eval picks CUDA when available, then MPS, then CPU.
- Native-resolution annotated renderer (no more side-by-side panel that halved usable resolution). Caries bboxes are drawn now; previously only mentioned in the text banner.
- Autoresearch unit at `autoresearch/cej-collapse/` (Karpathy-style training loop). **SIDELINED, not the path.** It's there if needed later; polyline pivot is the immediate work.
- Perio research synthesis at `output/research/2026-05-12-perio-deep-dive.md` and three child docs.
- Lessons-learned doc at `output/2026-05-12-lessons-learned-26hr-session.md` — read this before doing forensic conversation on individual images.

**Pickles state:**
- Tailscale IP `100.92.66.113`, SSH user `13038` (admin group), Tailscale-interface-only firewall rule `SSH-Tailscale`.
- Repo at `C:\Users\13038\repos\dental-rad-cli`, currently at `main` (`ee5d4a6` synced; pull again for `82ade54`+ before next training).
- Python 3.12 venv with torch 2.11.0+cu128 — see "Pickles gotchas" below.
- All trained weights present, baseline reproduces (`cej_collapse_rate = 0.3051` on CUDA).
- RTX 4090, 24 GB VRAM, driver 595.79.

**Mac state:**
- Working directory `~/repos/work/dental-rad-cli`.
- `.venv` set up via `uv sync` on M4 Max with MPS torch.
- DenPAR v3 unzipped at `data/denpar/Dataset/`.
- Baasils caries data prepared at `data/prepared/yolo_caries/` (download via `scripts/download_caries_data.sh` requires `ROBOFLOW_API_KEY` in `.env`).

## The decided direction — polyline pivot for CEJ

**Why polyline, not autoresearch knob-tweaking.** The 30.71% CEJ collapse rate has a structural root cause: DenPAR v3 ships loose CEJ point lists with no per-tooth grouping. After our adapter's heuristic pairing, **only 42% of teeth have clean 2-keypoint supervision** (18% have zero CEJ pts, 32% have one). The keypoint head learns that the "right" slot is unreliable and collapses to predicting identical points.

Knob-tweaking the existing Keypoint R-CNN can plausibly drop collapse to ~15-20% (best case via Hungarian pairing + 2-CEJ-only filter + loss rebalancing). That's the structural ceiling on this architecture with this data.

**Why polyline, not Banks-architecture-on-our-data.** Banks et al. 2025's perio-KPT pipeline (MIT-licensed, [arxiv 2503.13477](https://arxiv.org/abs/2503.13477)) reports 0.91+ PRCK on validation. Their architecture is YOLOv8-pose with visibility flags + heuristic snap-to-tooth-boundary. **Their architecture is replicable today**, but it still requires per-tooth-grouped keypoint training labels (same shape as Keypoint R-CNN's input). On DenPAR v3 + our adapter heuristic pairing, applying Banks's architecture hits the same data-bottleneck ceiling — the model can't differentiate keypoint slots when 50% of training teeth have unreliable supervision. **The bottleneck is the data structure, not the architecture.**

Polyline segmentation reframes the problem to be **data-bottleneck-free**:
- CEJ becomes a thin pixel band, predicted via semantic segmentation
- Training requires answering only "is this pixel CEJ band?" per pixel — a binary classification problem, not a per-tooth-slot assignment problem
- We derive the binary mask from the raw CEJ point list via y-band clustering (Correction 1) — no per-tooth grouping needed at training time
- Post-process at inference: intersect predicted polyline with tooth bbox left/right edges → 2 endpoints, **by construction**
- Collapse becomes structurally impossible
- The data sparsity ceiling softens because 1-CEJ-pt teeth contribute weak signal to their y-band cluster (a partial line is still useful), and 0-CEJ-pt teeth contribute nothing but at least don't poison the training signal

Two independent agents in the perio research synthesis converged on this recommendation (Lee/Kabir 2022 via academic survey; symmetric-with-existing-bone-polyline-head via BW survey). Published reference: [Lee/Kabir/Jiang 2022, arxiv 2109.12115](https://arxiv.org/abs/2109.12115) — three segmentation networks (bone, tooth, CEJ as thin band), DSC > 0.91, RBL stage AUC 0.89-0.90.

## Plan, sequenced

1. **Build polyline eval scaffold** (~200 lines, ~1 hour). New script `scripts/eval_cej_polyline.py`. Three metrics: per-site y-error (mm + px), CEJ-band IoU, polyline-degenerate rate. Includes a `px_to_mm` helper that uses per-tooth-class anatomical priors with a v0 image-region fallback.

2. **Run baseline on current Keypoint R-CNN** (~5 min). Produces the y-error mm number that the polyline pivot will be compared against. Output: `results/baseline-keypoint-cej-mm-eval/` with per-image and summary statistics.

3. **Build polyline adapter** (~150 lines, ~1 hour). `build_yolo_cej_polyline_dataset()` extending `src/dental_rad_cli/data/denpar_adapter.py`. Y-band clustering of flat CEJ_Points list. 30-px buffer. YOLOv8x-seg output format. Write to `data/prepared/yolo_cej_polyline/`.

4. **Extend training pipeline** (~30 lines). Add `target="cej"` branch to `src/dental_rad_cli/training/segmentation.py`. Same hyperparameters as bone segmentation.

5. **Train on pickles** (~30-90 min training time on RTX 4090). Reuse existing training stack. Save to `weights/segmentation_cej.pt`.

6. **Inference post-process** (~100 lines, ~30 min). Skeletonize predicted band → fit smooth curve → intersect with tooth bbox edges → emit mesial and distal endpoints. Wire into `analyze.py` parallel to the existing keypoint path (gated by flag during validation; default after).

7. **Eval polyline outputs** (~5 min). Run the new eval script on polyline predictions. Produces the comparison number.

8. **Build Family A math head** (~100 lines). Combine predicted CEJ polyline + existing bone-crest polyline → perpendicular mm distance per tooth within tooth bbox → AAP stage (≥2 / ≥4 / ≥6 mm thresholds). Apex head deleted; `severity.py` rewritten to consume mm instead of %.

9. **Joseph spot-checks ~20 rendered outputs** (~10-15 min). Clinical sanity judgment. Threshold for "OK / a bit off / wrong" emerges empirically. Q5 (tolerance) discovered from this review, not pre-locked.

10. Decision: ship polyline + Family A as default; iterate; or revisit architecture.

**Compute coordination with the parallel `dental-tooth-numbering` session:** Both train on pickles. 24 GB VRAM, both jobs ~5-8 GB peak each — concurrent is technically OK but risks slowdown or OOM. First-come-first-serve. Both jobs are short (<60 min); informal serialization is fine. Cloud option available if needed (see `scripts/cloud/` for GCP recipe).

## Why no Joseph annotation in v0

The v1 plan called for Joseph to scrub + annotate 30-50 office radiographs as a clinician-validated test corpus. **Deferred for v0.** Rationale:

- DenPAR v3's held-out test split (~200 PAs) is sufficient for v0 architectural validation. We're answering "does polyline beat keypoint?" — a relative question with the same test set on both sides.
- Office-distribution validation matters but is a v0.5 concern. Cross-distribution generalization should be measured after the architecture is proven on its training distribution.
- Joseph's clinical-time investment compounds more when the architecture is locked. Eyeballing 20 outputs after retrain (step 9) extracts most of the clinical value at 5% of the time cost.
- The one clinical input that was actually load-bearing — **CEJ visibility on BW vs PA symmetry** — Joseph confirmed (symmetric, same visibility on both modalities). No further upfront clinical input needed.
- Q5 (mm tolerance threshold) was originally framed as a hard prerequisite. It dissolves into the post-retrain review step: Joseph spot-checks ~20 outputs, ratings of "OK / a bit off / wrong" empirically reveal the y-error threshold that flips the rating. Discovered, not predicted.

Office-data revisit triggers (any one fires → start the office-data path):
- Polyline architecture locks AND ships v0 on DenPAR
- A second consumer (NoteBrusher chairside use) demands office-distribution validation
- Joseph spot-check (step 9) reveals systematic miscalibration that DenPAR test split alone can't explain

## Outdated stack components beyond Keypoint R-CNN

| Component | Status | Notes |
|---|---|---|
| Keypoint R-CNN (CEJ head) | **Replacing** | Polyline pivot in flight |
| Keypoint R-CNN (apex head) | **DELETING (v2)** | Family A math head doesn't need apex; head deletion is part of v2 plan step 8 |
| Keypoint R-CNN (bone head) | Stays (uses polyline-derived bone crest) | Unchanged |
| Wimalasiri % bone-loss formula | **DELETING (v2)** | Replaced by Family A mm formula across both modalities |
| 7 separate trained models architecture | Outdated systemically | 2024+ pattern is one foundation backbone (SAM2 / DINOv2 / Sapiens) + N lightweight LoRA adapters. Multi-week pivot. Revisit after polyline lands. |
| DenPAR v3 adapter heuristic (CEJ side) | Outdated, dataset-forced | Polyline pivot dissolves most of it. Bone-crest side stays. |
| YOLOv8x / YOLOv9e (tooth detection) | Partial — defer | YOLOv11/v12 exist with ~1-3% mAP gain. Tooth detector has molar-misclassified-as-single-rooted bug; YOLOv11 swap might help, low-risk experiment. |
| YOLOv8x-seg (tooth + bone segmentation) | Partial — defer | SAM2 + adapter is the bitter-lesson alternative. Working acceptably today. |
| YOLOv8s (caries) | Defer | Caries head matches/beats published baselines. Don't fix what isn't broken. |
| Hand-engineered rule layer | Philosophically outdated, practically keep | Bitter lesson predicts learned. But explainability matters chairside. Defer. |
| CLAHE preprocessing (clip=40) | Not really outdated | 1994 algorithm; Wimalasiri's paper specifies clip=40 as load-bearing. Foundation models would re-evaluate. |
| Schema, matplotlib rendering, eval methodology | Not outdated | Fit-for-purpose. |

**Highest-leverage modernization beyond v2 changes:**
1. Polyline segmentation pivot for CEJ (decided, immediate work)
2. `dental-tooth-numbering` substrate (parallel work, sister session)
3. YOLOv11 swap for tooth detection + segmentation as quick experiment (~1 hour, low risk)
4. Foundation backbone + LoRA adapters across all heads (multi-week, defer until polyline lands)

## Pickles access + gotchas

**SSH from this Mac:**
```bash
ssh 13038@100.92.66.113
# or via Tailscale hostname:
ssh 13038@pickles
```

**Critical: don't use `uv run python` on pickles.** It triggers a dependency re-sync on every invocation, which downgrades torch to CPU-only (because `pyproject.toml` lists `torch>=2.5.0` without specifying the CUDA index). Always use the venv's Python directly:

```powershell
cd C:\Users\13038\repos\dental-rad-cli
.venv\Scripts\python.exe scripts\eval_keypoint_cej.py
```

The long-term fix is adding `[tool.uv.sources]` to `pyproject.toml` to pin torch/torchvision to a CUDA index. Not done yet.

**If venv torch gets downgraded to CPU**, restore CUDA with:

```powershell
cd C:\Users\13038\repos\dental-rad-cli
.venv\Scripts\python.exe -m pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

`cu128` matches driver 595.79. Adjust if the driver changes.

**Other pickles notes:**
- Admin user `13038` means SSH keys live at `C:\ProgramData\ssh\administrators_authorized_keys`, NOT `C:\Users\13038\.ssh\authorized_keys`. sshd silently ignores the per-user file for admin-group users.
- Firewall rule `SSH-Tailscale` scopes port 22 to the Tailscale interface only. Don't open `OpenSSH-Server-In-TCP`.
- The system Python on pickles is a Microsoft Store stub that errors when invoked bare. Use `.venv\Scripts\python.exe` or install Python 3.12 separately.
- 24 GB VRAM is plenty for our workload. Don't worry about batch sizes blowing memory.

## Important docs to read

In rough priority order:

1. **`output/2026-05-12-lessons-learned-26hr-session.md`** — 12 named failure patterns from the previous build sprint. The first action of any future "is the model good?" conversation is to compute the held-out metric, not analyze images.
2. **`output/research/2026-05-12-perio-deep-dive.md`** + child docs (especially `2026-05-12-perio-deep-dive-bw-and-arch.md` for Family A/B/C/D detail) — synthesis of the perio architecture/data landscape.
3. **`docs/methodology-brief.md`** — cleanroom reference for the Wimalasiri reimplementation.
4. **`src/dental_rad_cli/data/denpar_adapter.py`** — existing adapter, especially `_bone_polygons_from_polylines()` (geometric template for the polyline adapter).
5. **`scripts/eval.sh`** — how to measure. Run this after any model retrain.

## Flagged but not the immediate focus

Real issues, documented so they're not lost, but not the polyline-pivot path's responsibility:

- **Tooth detector misclassifies molars as single-rooted** on most BWs and several PAs. Separate retrain or YOLOv11 swap. v0.5 of `dental-tooth-numbering` substrate may produce a fix (if it ships a 32-class detector that retires our current 1-class).
- **Apex predictions hug bbox-top edge.** Mitigated by v2 plan step 8 (apex head deletion). After deletion, this stops being a contaminant.
- **The autoresearch unit at `autoresearch/cej-collapse/`** is parked. Could be useful later to optimize polyline hyperparameters once architecture is locked.
- **GCP cloud GPU setup** under `scripts/cloud/` is partially functional but unused — pickles via SSH is the host now. Available as fallback if pickles unavailable.

## Numbers to track in the next session

| Metric | Baseline (current Keypoint R-CNN) | Polyline pivot target | Published anchor |
|---|---|---|---|
| **per-site y-error median (mm)** (PRIMARY) | TBD — compute on current outputs first | < 1.0 mm | AlGhaihab 2025 MAE 0.499 mm (BW) |
| **per-site y-error p90 (mm)** | TBD | < 3.0 mm | AAP threshold = 2 mm |
| **per-site y-error median (px)** (sibling) | TBD | < 15 px | |
| **per-site y-error p90 (px)** | TBD | < 40 px | |
| **CEJ-band pixel IoU** | TBD | > 0.50 | Lee/Kabir 2022 DSC > 0.91 (stretch) |
| **polyline-degenerate rate** | n/a (no polyline output today) | < 5% | n/a |
| `cej_collapse_rate` (sanity check) | 0.3051 | ~0 by construction | n/a |
| `caries_map50` (val n=58, control) | 0.6478 | unchanged (don't touch caries) | Salehizeinabadi 2025 |

## The concrete first action

```bash
# On Mac:
cd ~/repos/work/dental-rad-cli
# Create scripts/eval_cej_polyline.py with three real metrics:
#   - per-site y-error (mm + px) at bbox.x1 and bbox.x2 vs GT CEJ points
#   - CEJ-band pixel IoU on dilated GT band (30-px buffer)
#   - polyline-degenerate rate (fraction of teeth where predicted band
#     fails to cross both vertical bbox edges)
# The script must accept BOTH:
#   - current Keypoint R-CNN output format (for baseline measurement)
#   - future polyline output format (for after-retrain comparison)
# Uses per-tooth-class anatomical priors for px→mm conversion. v0
# fallback if class unknown: image-level scale from mean tooth height.
```

After the eval script: run it on the current Keypoint R-CNN to produce the baseline mm y-error number. THEN write the polyline adapter (step 3 of plan above).

The total work to first measurable result is probably 6-10 hours of Claude time, sequenced over a few days. No Joseph clinical time required until step 9 (~10-15 min spot-check).

---

*Written 2026-05-13 (v1) and revised 2026-05-12 evening (v2). Ground truth on these dates: `cej_collapse_rate=0.3051` (pickles CUDA), `caries_map50=0.6478` (val n=58). Both numbers are the floor — any change gets compared against these. Pickles is the canonical training/eval host; Mac is for development.*

*v1 corrections from external review (other Claude) baked in: y-band clustering not bbox-anchored pairing (Correction 1), 30-px buffer not 2-px (Correction 2), new primary metrics not `cej_collapse_rate` (Correction 3).*

*v2 reframes from Joseph conversation 2026-05-12 baked in: Family A primary (apex-free, both modalities), apex head deletion, px→mm calibration via tooth-class priors, parallel `dental-tooth-numbering` substrate work, Joseph annotation deferred, Q5 dissolved into post-retrain empirical review, BW/PA symmetry confirmed.*
