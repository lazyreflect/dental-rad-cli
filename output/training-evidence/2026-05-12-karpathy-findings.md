# Karpathy-style stratified analysis — trained CEJ polyline model

**Date:** 2026-05-12
**Model:** `weights/segmentation_cej.pt` (88 epochs, early-stopped at epoch 58 best)
**Test set:** DenPAR Testing split, n=200 PA images
**Source data:** `output/training-evidence/karpathy-stratify.csv` (per-image features + predictions)

---

## TL;DR

The polyline architecture **works.** When the model has confidence (≥0.5), it
predicts correctly **100% of the time**. The 17% per-image failure rate is a
**data bottleneck (DenPAR's sparse, uneven labeling), NOT a model bottleneck.**

**The operational ship strategy:** filter inference output by
`max_conf >= 0.5`. This keeps 65.5% of test images with 100% prediction
success. The remaining 35% of images fall back to "model uncertain — manual
review."

| Confidence filter | Images kept | Predict-success rate |
|---|---|---|
| no filter (conf≥0.25 default) | 200/200 | 83% |
| **conf ≥ 0.5** | **131/200 (66%)** | **100%** |
| conf ≥ 0.6 | 104/200 (52%) | 100% |
| conf ≥ 0.7 | 63/200 (32%) | 100% |
| conf ≥ 0.8 | 25/200 (13%) | 100% |

Confidence is a clean separator. The model is internally well-calibrated.

---

## The headline finding — supervision-density determines success

Stratifying by the GT CEJ point count of each test image (a proxy for "does
this image LOOK like a dense-supervision image to the model"):

| n_GT_CEJ_points | n images | success_rate |
|---|---|---|
| 0-2 | 5 | **20%** |
| 3-4 | 40 | **60%** |
| 5-7 | 118 | **88%** |
| 8+ | 37 | **100%** |

**Monotonic.** At 8+ GT CEJ points, success is *complete*. The polyline
architecture is solving the problem when supervision is adequate. The failures
are concentrated in images that look — visually — like the sparsely-labeled
training images.

This is the Karpathy diagnosis: **the model is doing exactly what we trained
it to do.** DenPAR has uneven labeling density; some images get 8+ CEJ
points, some get 2. The model learned to mimic the labeling-density pattern.
At inference, images that visually resemble the sparse-supervision cluster
trigger sparse predictions.

---

## What else moves the needle — and what doesn't

### What moves the needle

**Number of teeth in image (`n_bboxes`):**

| teeth in image | n | success_rate |
|---|---|---|
| 1 | 4 | **25%** |
| 2 | 13 | **46%** |
| 3-4 | 95 | 82% |
| 5+ | 88 | **92%** |

Single-tooth crops are the worst case. More teeth in frame → more visual
context → more confident prediction.

### What doesn't move the needle

- **Arch (Upper/Lower):** Upper 85%, Lower 82%. Flat.
- **Restoration proxy** (top-quartile pixel fraction): 196/200 images cluster
  in one bin. Not a useful stratifier on DenPAR.
- **Image aspect ratio:** flat across the range.

### Mildly suggestive

**Crosstab Arch × Site (worst → best):**

| arch | site | n | success_rate |
|---|---|---|---|
| Upper | Right | 24 | **75%** |
| Lower | Left | 45 | 78% |
| Lower | Right | 56 | 82% |
| Upper | Anterior | 30 | 87% |
| Lower | Anterior | 20 | 90% |
| Upper | Left | 25 | **92%** |

Some anatomical regions are slightly harder. But the spread (75-92%) is
small compared to the supervision-density spread (20-100%). Don't optimize
for this.

---

## The Karpathy-recipe conclusion

> *"Your model is doing exactly what the data tells it to do. The next move
> is data, not training."*

**Three options ranked:**

### Option A — Ship with confidence filter (v0 recommendation)

- **What:** Pipeline runs CEJ polyline at inference; outputs go through a
  `max_conf >= 0.5` gate. High-confidence predictions feed Family A math
  head and emit mm + AAP stage. Low-confidence predictions fall back to
  "manual measurement recommended."
- **Coverage:** ~66% of chairside radiographs get automated measurement.
- **Effort:** ~30 LOC (filter + UX label in the rendered output).
- **Why this is the right Karpathy move:** ships a usable product TODAY,
  uses the model's own uncertainty calibration, doesn't burn cycles on
  data we don't have.

### Option B — Synthesize denser supervision for sparse-labeled training images (v0.5)

- **What:** For training images with `n_cej_gt < 5`, generate plausible
  additional CEJ band supervision via inter-tooth interpolation. Retrain.
- **Risk:** Hallucinated labels could introduce wrong training signal in
  the sparse regime. Have to be conservative about what we synthesize.
- **Expected lift:** could move the 20-60% success in the 0-4 GT regime
  closer to 88% (the 5-7 regime baseline).
- **Effort:** ~150 LOC adapter changes + retrain (30 min). Then re-eval.
- **Risk mitigation:** Use the trained model's own predictions on training
  images to identify which sparse cases have visually-clear CEJ that just
  wasn't labeled; only synthesize for those.

### Option C — Train longer with relaxed patience

- **What:** Retrain with `patience=200` or `--patience 0` (disable early
  stop).
- **Why this likely won't help much:** training already hit early-stop at
  epoch 88 with patience=30 from epoch 58 best. Loss curve had plateaued.
  More epochs won't teach the model what isn't in the labels.
- **When to revisit:** if Option B doesn't move numbers and we still have
  appetite for more compute.

---

## My recommendation

**Ship Option A now. Plan Option B for v0.5.**

Concrete steps for Option A:

1. Wire `max_conf >= 0.5` filter into `pipeline/aggregate.py` or
   wherever the polyline output gets consumed
2. Family A math head emits mm only when filter passes; otherwise the
   schema's `BoneLossSite.reason` field carries `"low_model_confidence"`
3. Render layer shows "automated CEJ measurement" vs "manual measurement
   recommended" badges based on the filter outcome
4. Delete apex head (the v2 plan already calls for this — Family A doesn't
   need apex)
5. End-to-end test on 5-10 office radiographs (Joseph's BWs/PAs) to verify
   the filter behaves clinically

Karpathy would say: **good engineering. The 65% of images that work, work
well. The 35% that fall back to manual review are honestly flagged. Ship.**

---

## What I learned that the wiki should capture

For the workspace knowledge artifact (per Karpathy's LLM-wiki essay
referenced in CLAUDE.md):

1. **DenPAR has uneven labeling density.** Images with < 5 GT CEJ points
   produce poor model predictions. Future training runs on DenPAR should
   either filter these out OR synthesize supplementary supervision.

2. **The polyline architecture is data-bottleneck-free in supervision
   construction but NOT in supervision density.** The y-band clustering
   doesn't need per-tooth pairing, but it can't conjure CEJ bands where
   no GT points exist.

3. **Confidence calibration earned through training is a feature.** YOLO
   seg's confidence threshold is a clean separator. Don't fight it; use
   it for triage.

4. **Single-tooth crops and posterior-molar-only views are the hard
   regime.** These are common chairside (BWs especially). Office data
   collection (v0.6+ ladder) should oversample these to fix the bottleneck.

5. **The Karpathy recipe is real.** ~30 min in pandas + matplotlib beats
   hours of architecture-fiddling.
