# State of the Work — dental-rad-cli — 2026-05-13

**Audience:** the next session picking up this project (also an external review reader). Read this end-to-end before any code.

## TL;DR

We have two measured baselines on held-out test data. The CEJ keypoint head fails on ~30% of teeth. The decided next move is an architectural pivot from Keypoint R-CNN to **polyline segmentation** for the CEJ landmark, on pickles (RTX 4090) over Tailscale SSH. The polyline supervision is built via **y-band clustering** of the loose DenPAR v3 CEJ point list (NOT bbox-anchored pairing — see "Adapter spec corrections" below).

| Metric | Value | Lower/Higher better | Notes |
|---|---|---|---|
| `cej_collapse_rate` | **0.3051** (pickles CUDA) / 0.3071 (Mac MPS) | lower | Bimodal failure: ~30% of teeth get collapsed prediction (<10 px between mesial and distal); the rest are ~150-250 px apart |
| `caries_map50` | **0.6478** (Baasils val n=58) | higher | At-par with Salehizeinabadi 2025; deep AP50 = 0.85 beats their 0.80 |
| `caries_map50_95` | 0.3858 | higher | |

The cross-host drift (0.3051 vs 0.3071) is fp rounding noise. Pin **pickles-CUDA** as the canonical eval host going forward — eval runs in 9 sec there vs 48 sec on Mac MPS vs 219 sec on pickles CPU.

**First action for next session:** request anatomical priors from Joseph (see §"Joseph's role" below), then write the polyline adapter using y-band clustering — NOT the bbox-anchored single-point extrapolation that the v1 draft of this doc proposed. ~150 lines. Path: extend `src/dental_rad_cli/data/denpar_adapter.py` with `build_yolo_cej_polyline_dataset()` analogous to `build_yolo_dataset()` for `bone_seg`.

## Three structural corrections from external review (load-bearing)

The previous draft of this handoff had three flaws an external Claude reviewer caught. The corrections shape the polyline pivot. **All three are baked into this revision.**

### Correction 1 — adapter must use y-band clustering, not bbox containment

The previous draft said: *"1 CEJ pt on a tooth → bbox-anchored extrapolation using the median CEJ-y-fraction prior."* That reintroduces the bbox-pairing problem through the back door — it requires knowing which tooth bbox owns each CEJ point, which is exactly the noisy heuristic we wanted to escape.

**Corrected approach:** For each image, take the flat `CEJ_Points` list as-is. Cluster points by y-proximity (within ~30 px y of each other) AND reasonable x-proximity (within ~200 px x of nearest neighbor, to avoid linking across image gaps). Connect each cluster's points in x-order into a polyline. Buffer to a band. **No bbox lookup at training time.** Per-tooth assignment happens once at inference, deterministically, when the predicted polyline intersects each tooth bbox's left/right edges.

This is the only adapter shape that delivers the "polyline pivot is data-bottleneck-free" claim. The previous draft's version was structurally a regression to the noisy-pairing regime under a new name.

Anatomical justification: CEJ is anatomically a **continuous band across all teeth in an arch**, not isolated per-tooth landmarks. Multi-tooth PA → typically ONE polyline across all teeth. BW showing both arches → TWO polylines (upper + lower) clustered separately by y.

### Correction 2 — buffer to 30 px, not 2 px

The previous draft specified "2-px line." That's sub-pixel at YOLOv8x-seg's proto resolution. The proto basis is `imgsz / 4` — at `imgsz=640` it's `160×160`, so a 2-px line in image space is `0.5 px` in proto space. The model literally can't represent it reliably.

Our existing bone-segmentation precedent in `src/dental_rad_cli/data/denpar_adapter.py:483` uses `_BONE_STRIP_HALF_WIDTH = 15` (30 px total band) and trains cleanly. **Use the same constant for CEJ:** `_CEJ_STRIP_HALF_WIDTH = 15`.

At inference time, if a thin curve is needed for endpoint extraction, skeletonize the predicted band (`cv2.ximgproc.thinning` or `scipy.ndimage.skeletonize`). Standard image processing.

### Correction 3 — `cej_collapse_rate` becomes trivial after polyline; need new primary metrics

The previous draft said "collapse rate stays valid because two endpoints by construction = no collapse." That's true and useless. After polyline post-process intersects the predicted band with `bbox.x1` and `bbox.x2`, the endpoint separation is **roughly the bbox width** (80-300 px typical). Collapse threshold is 10 px. So `cej_collapse_rate` drops to ~0 by construction, regardless of polyline anatomical accuracy. The polyline could be 50 px coronal of the true CEJ and `cej_collapse_rate` would still be ~0.

The metric is "low by construction, not by quality." Headline metric needs to change.

**Primary metrics for the polyline pivot:**

| Metric | Definition | Target |
|---|---|---|
| **per-site y-error (px)** | absolute y-error between predicted CEJ endpoint and ground-truth CEJ point, at `x=bbox.x1` (mesial) and `x=bbox.x2` (distal). Computed only on teeth where ground-truth has BOTH CEJ points (~42% of DenPAR test). | median < 15 px; p90 < 40 px |
| **CEJ-band pixel IoU** | predicted CEJ mask vs dilated ground-truth band (same 30-px buffer as training). | > 0.50 (loose, matches Lee/Kabir 2022 territory) |
| **polyline-degenerate rate** | fraction of teeth where the predicted band fails to cross both vertical bbox edges. | < 5% |
| `cej_collapse_rate` (demoted) | as before. | ~0 — used as sanity check, not headline |

**These metrics need to be built BEFORE the retrain**, so we have a baseline number on the current Keypoint R-CNN outputs to compare against. The current 30.71% / 30.51% collapse rate gives no per-site y-error number — that requires recomputing against the same ground truth using the keypoint predictions instead of polyline endpoints.

Path: new eval script `scripts/eval_cej_polyline.py` (sibling to existing `eval_keypoint_cej.py` so the original baseline number stays stable). Roughly 200 lines.

## Where we are

**Repo state at HEAD** (`82ade54` on `main` at write time):

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

### Parallel ask — Banks Zenodo access

File the perio-KPT [Zenodo access request](https://zenodo.org/records/17272200) anyway. 5 minutes, free, ~14-day round trip. If approved, we have either (a) replicate Banks's setup with their data, or (b) train polyline on cleaner mask labels derived from their per-tooth-grouped points. The polyline work proceeds in parallel on DenPAR v3 and finishes long before the request answer. They're complementary, not competing.

Address the license-inconsistency question in the request: paper says CC-BY 4.0, Zenodo metadata says CC-BY-NC-SA 2.0 Generic. Ask Banks et al. (Surrey University + KCL + UNMSM Peru + KGMU India) to clarify which applies.

## Plan, sequenced

1. **Joseph: anatomical priors conversation** (~30 min, see §"Joseph's role" below). Output: a handful of clinical rules-of-thumb that shape adapter and post-process heuristics.

2. **Joseph: scrub + annotate 30-50 test cases** (~60-90 min). Output: clinician-validated CEJ polyline ground truth on Joseph's office BWs/PAs.

3. **Claude: build annotation harness** (~150 lines, ~30 min). Frontier-LLM-vision pre-annotates; Joseph reviews and edits. PHI stays local.

4. **Claude: build polyline metric scaffold** (~200 lines, ~1 hour). New eval script `scripts/eval_cej_polyline.py` computing per-site y-error + IoU + degenerate-rate. Run on current Keypoint R-CNN outputs to establish baselines BEFORE retrain.

5. **Claude: build polyline adapter** (~150 lines, ~1 hour). y-band clustering of flat CEJ_Points list. 30-px buffer. YOLOv8x-seg output format. Write to `data/prepared/yolo_cej_polyline/`.

6. **Claude: training on pickles** (~30-90 min training time on RTX 4090). Reuse existing `src/dental_rad_cli/training/segmentation.py` with CEJ as a new 1-class label.

7. **Claude: inference post-process** (~100 lines, ~30 min). Skeletonize predicted band → fit smooth curve → intersect with tooth bbox edges → emit mesial and distal endpoints. Wire into `analyze.py` parallel to the existing keypoint path (gated by flag).

8. **Eval on DenPAR test split + Joseph's annotated cases.** Compare new metrics against keypoint baseline.

9. **Joseph: review model outputs** (~15 min). Clinician-grounded accuracy judgment per tooth.

10. Decision: ship polyline as CEJ default, or iterate.

**Target outcome:** drop median per-site y-error on the test split from whatever-Keypoint-R-CNN-produces to under 15 px median, 40 px p90. Stretch goal: match Lee/Kabir 2022's inferred DSC > 0.91 on the CEJ band.

## Joseph's role — leverage clinical + frontier-LLM-vision

You (Joseph) are a licensed dentist with frontier vision-LLM access. Three asks, ranked by leverage:

### Highest leverage — clinician-validated test corpus (60-90 min one-time)

The biggest single contribution: **30-50 cases from your own PMS, scrubbed, with CEJ polylines marked.** This becomes the ground-truth test corpus that anchors every polyline-pivot eval to clinical reality. DenPAR v3's 200 test PAs are useful but they're someone else's annotators on someone else's distribution; a small high-quality set from your office is the metric that actually matters chairside.

Composition:
- 20-30 bitewings (chairside use case)
- 10-20 periapicals (matches Wimalasiri's training distribution, validates cross-modality)
- Mix of healthy and visible bone loss (don't only sample dramatic cases)
- Include ≥3-5 with restorations (forces handling radiopaque crowns)
- Include ≥3-5 tilted or non-ideal positioning (real-world quality)

**Workflow — frontier LLM does the heavy lifting:**

Per image:
1. Drop scrubbed PNG into a Claude (or other frontier vision LLM) conversation
2. LLM overlays rough CEJ polyline per visible tooth + outputs coordinates
3. You review, fix wrong ones (drag/type adjustments)
4. Final polyline coords saved to JSON sibling file at `data/office-eval/{bw,pa}NN.cej.json`

Claude builds the annotation harness — ~150 lines, sends image to vision LLM with structured prompt, renders response at full resolution, supports edits, writes JSON. Time per image with this loop: 1-2 min. **30-50 cases = 60-90 min of clinical time.**

### Medium leverage — review model outputs after each retrain (10-15 min per round)

After we train the polyline model and run on your test set, **review annotated PNGs and flag clinically right vs wrong findings.** Not pixel-level — judgments like "this tooth's CEJ placement is OK / off by ~3 mm coronal / off by ~5 mm apical / completely wrong." This gives a clinician-grounded accuracy number that complements pixel IoU. Pixel IoU might say 0.78 but if the polyline is consistently 5 mm coronal of true CEJ, the bone-loss percentage systematically under-reports. Your eye catches that; the pixel metric doesn't.

Claude builds a simple review UI (Streamlit or just annotated PNGs + numbered checklist). Walks through one tooth at a time. ~15 min per retrain.

### Anatomical priors conversation (~30 min, one-time)

This is the part that wasn't fully explained earlier. Concrete questions where your clinical knowledge directly shapes the model:

**Q1 — Single-CEJ-point recovery.** When a tooth has only 1 CEJ point labeled (32% of teeth in DenPAR v3), where anatomically should the other one be?
- Naive answer: "at the opposite side of the bbox at the same y" — assumes CEJ is perfectly horizontal across a tooth
- Real anatomy might be: "the CEJ scallops — slightly more coronal at the mesial than the distal on molars; the slope direction depends on tooth type"
- Why it matters: in the y-band clustering version of the adapter, single CEJ points get connected to their y-band neighbors. If neighbors don't exist, do we extrapolate, or drop the tooth? Anatomical rules tell us which is safer.

**Q2 — CEJ continuity across teeth in an arch.** Is the CEJ band roughly horizontal across an arch, or does it follow a curve (e.g., higher anterior, lower posterior)?
- Affects the y-band clustering window — if the CEJ has significant arch-level curvature, a too-tight y-band breaks the natural cluster
- Affects how we judge whether a predicted polyline is "right" — should it be straight or curved?

**Q3 — BW vs PA visibility.** Does CEJ show up the same way on both modalities, or is it harder to see on one?
- Affects training data weighting and modality-specific handling
- Affects expectations for the BW use case (currently the no-apex math problem dominates, but if CEJ is harder to see on BW the polyline model will also suffer)

**Q4 — Multi-rooted molar detection rules.** We have a separate problem where the tooth detector calls molars "single-rooted" most of the time. Is there a fast clinical rule for spotting a multi-rooted molar from a 2D radiograph that we could encode?
- E.g., "any tooth distal to the second premolar with visible furcation = multi-rooted"
- E.g., "any tooth whose bbox aspect ratio is wider than 0.8 and is in the molar position = multi-rooted"
- Doesn't fix the tooth detector but tells us if we're missing an obvious feature

**Q5 — Measurement-error tolerance for clinical usefulness.** What's your tolerance? If the polyline is within ±2 mm of true CEJ at typical pixel scale, is that good enough to make a Stage I vs II call? ±5 mm?
- Defines the "we're done" success threshold for the polyline pivot
- Anchors the per-site y-error metric: at DenPAR's typical resolution, 1 mm ≈ 10-15 px. So ±2 mm tolerance translates to ~20-30 px y-error tolerance.

We have this conversation BEFORE the polyline adapter is written, so the answers shape implementation decisions. ~30 min of clinical time.

### Lower leverage (defer)

- **Hand-labeling thousands of cases for from-scratch retraining** — not yet. We don't have the active-learning UI to make this efficient. After polyline pivot lands and we have a working baseline, this becomes the path; not before.
- **Writing chairside note templates** — defer. The note-draft layer is easy to iterate once perception works.
- **Curating multi-office datasets** — defer to v0.6+ per TODOS.md office-data ladder.

### PHI / HIPAA boundaries (brief)

Standard pattern:
- Scrub PHI before any image leaves your local machine (use `docs/phi-scrub-recipe.md`)
- Frontier LLM calls go through a BAA endpoint (same pattern NoteBrusher uses)
- Test corpus lives at `~/tenant-data/dental-rad-eval/` or on pickles, **never** in the git repo. Workspace CLAUDE.md hard rule.
- Polyline coordinate JSONs are derived artifacts — also stay local until scrub review.
- Trained weights are fine to ship publicly (no PHI in weights). Test corpus is not.

## Outdated stack components beyond Keypoint R-CNN

Categorized so a future session knows what's load-bearing-modern vs technically-dated-but-defer.

| Component | Status | Notes |
|---|---|---|
| Keypoint R-CNN (3 heads for CEJ/bone/apex) | **Outdated, replace** | Polyline pivot does CEJ; bone-crest already uses polyline; apex is the remaining keypoint use |
| 7 separate trained models architecture | **Outdated systemically** | 2024+ pattern is one foundation backbone (SAM2 / DINOv2 / Sapiens) + N lightweight LoRA adapters. Multi-week pivot. Revisit after polyline lands. |
| DenPAR v3 adapter heuristic | **Outdated, dataset-forced** | Polyline pivot mostly dissolves it. Banks 2025 perio-KPT (gated) dissolves it entirely. |
| YOLOv8x / YOLOv9e (tooth detection) | **Partial — defer** | YOLOv11/v12 exist with ~1-3% mAP gain. RT-DETR is the bitter-lesson alternative. Tooth detector has molar-misclassified-as-single-rooted bug; YOLOv11 swap might help, low-risk experiment. |
| YOLOv8x-seg (tooth + bone segmentation) | **Partial — defer** | SAM2 + adapter is the bitter-lesson alternative. Working acceptably today. |
| YOLOv8s (caries) | **Defer** | Caries head matches/beats published baselines. Don't fix what isn't broken. |
| Hand-engineered rule layer | **Philosophically outdated, practically keep** | Bitter lesson predicts learned. But explainability matters chairside. Defer. |
| CLAHE preprocessing (clip=40) | **Not really outdated** | 1994 algorithm; Wimalasiri's paper specifies clip=40 as load-bearing. Foundation models would re-evaluate. |
| Schema, matplotlib rendering, eval methodology | Not outdated | Fit-for-purpose. |

**Highest-leverage modernization beyond Keypoint R-CNN replacement:**
1. Polyline segmentation pivot for CEJ (decided, immediate work)
2. YOLOv11 swap for tooth detection + segmentation as quick experiment (~1 hour, low risk)
3. Foundation backbone + LoRA adapters across all heads (multi-week, defer until polyline lands)

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
2. **`output/research/2026-05-12-perio-deep-dive.md`** — synthesis of the perio architecture/data landscape. Banks 2025 perio-KPT, Lee/Kabir 2022 polyline architecture, what's confirmed unavailable.
3. **`docs/methodology-brief.md`** — cleanroom reference for the Wimalasiri reimplementation.
4. **`src/dental_rad_cli/data/denpar_adapter.py`** — existing adapter, especially `_bone_polygons_from_polylines()` (geometric template for the polyline adapter).
5. **`scripts/eval.sh`** — how to measure. Run this after any model retrain.

## Flagged but not the immediate focus

Real issues, documented so they're not lost, but not the polyline pivot's responsibility:

- **Tooth detector misclassifies molars as single-rooted** on most BWs and several PAs. Separate retrain or YOLOv11 swap.
- **Apex predictions hug bbox-top edge** rather than landing on actual root tips. Same sparse-supervision pattern as CEJ. Polyline pivot doesn't fix this; apex would need its own architectural treatment. Bone-loss math accuracy is bounded by apex accuracy too.
- **BW-mode bone-loss math.** Wimalasiri's % formula assumes apex in frame. Bitewings don't have apex. Literature handles this three ways: (A) switch unit to mm CEJ→bone-crest, (B) polynomial arch fit as implicit reference, (C) image-level classifier. We'd want a separate BW pipeline; currently all bone-loss output on BW is structurally wrong.
- **The autoresearch unit at `autoresearch/cej-collapse/`** is parked. Could be useful later to optimize polyline hyperparameters once architecture is locked.
- **GCP cloud GPU setup** under `scripts/cloud/` is partially functional but unused — pickles via SSH is the host now.
- **Codex feedback on clinical image mining via Curve API** — relevant for v0.6+ office-data path, not the polyline pivot.

## Numbers to track in the next session

| Metric | Baseline (current) | Polyline pivot target |
|---|---|---|
| `cej_collapse_rate` (pickles CUDA, sanity check) | 0.3051 | ~0 by construction |
| **per-site y-error median (px)** (PRIMARY) | TBD — compute on current Keypoint R-CNN outputs first | < 15 |
| **per-site y-error p90 (px)** | TBD | < 40 |
| **CEJ-band pixel IoU** | TBD — compute on current outputs first | > 0.50 |
| **polyline-degenerate rate** | n/a (no polyline output today) | < 5% |
| `caries_map50` (val n=58) | 0.6478 | unchanged (don't touch caries) |

## The concrete first action

Before any adapter code: **anatomical priors conversation with Joseph** (30 min, see §"Joseph's role" → Q1-Q5 above).

Then in this order:
1. Build the annotation harness (Claude, ~30 min)
2. Joseph scrubs + annotates 30-50 cases (60-90 min)
3. Build the polyline metric scaffold (`scripts/eval_cej_polyline.py`, ~1 hour) — compute baseline numbers on current Keypoint R-CNN outputs
4. Build the polyline adapter (y-band clustering, 30-px buffer, ~1 hour)
5. Train on pickles (RTX 4090, ~30-90 min)
6. Build inference post-process + skeletonization
7. Eval on DenPAR test split + Joseph's annotated cases
8. Joseph reviews outputs
9. Decision: ship polyline as default or iterate

The total work to first measurable result is probably 8-12 hours of Claude time + 2-3 hours of Joseph clinical time, sequenced over a few days.

---

*Written 2026-05-13. Ground truth on this date: cej_collapse_rate=0.3051 (pickles CUDA), caries_map50=0.6478 (val n=58). Both numbers are the floor — any change gets compared against these. Pickles is the canonical training/eval host; Mac is for development.*

*Corrections from external review (other Claude) baked in: y-band clustering not bbox-anchored pairing (Correction 1), 30-px buffer not 2-px (Correction 2), new primary metrics not cej_collapse_rate (Correction 3).*
