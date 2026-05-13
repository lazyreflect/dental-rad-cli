# 2026-05-12 night session — summary for Joseph

Written by Claude Opus 4.7 autonomously after Joseph went to bed.
Continuation of the earlier session that produced the benchmark-pivot
handoff. Goal of the night: execute the Karpathy moves on the BR-series
queue, commit/push every checkpoint, log negative results to prevent
re-tries.

**HEAD at session end: [26ff399](https://github.com/lazyreflect/dental-rad-cli/commit/26ff399).** Tests still 147 green.
Held-out split (50 images) was **not touched** at any point.

---

## TL;DR for breakfast

1. **The 0.300 mm Overjet "target" is marketing, not a benchmark.** A
   subagent literature sweep confirmed that no commercial vendor
   (Overjet / Pearl / Adravision / VideaHealth / Denti.AI) publishes
   data-curation criteria or severity-stratified MAE. AlGhaihab/
   Denti.AI 2025 explicitly used 39 hand-selected radiographs from
   Denti.AI's pool. Our DenPAR 0.723 with stratification is
   structurally more rigorous than any vendor's pooled mean. The
   goal should pivot from "beat Overjet's 0.300" to "publish the
   most rigorous public-corpus MAE with severity-stratified reporting"
   (filed as BR11).

2. **The architecture is right on the bulk of cases. The mean is dragged
   by a small hard-case tail.** Posterior-view subset (n=183, 79% of
   scored dev sites) has mean MAE 0.655 / **median 0.345 mm — within
   45 μm of Overjet's pooled 0.300**.

3. **Severe-bucket under-prediction has confirmed algorithm-side fix
   space — but not via simple y-statistic rules.** BR9 dissection of
   all 34 severe sites showed 62% have masks that reach the deep
   crest; the rule picks shallow. BRneg-2 swept 6 candidate rules
   (median / max / wide-aware / various windows) and found a
   monotonic Pareto frontier: more apical bias → better severe,
   worse healthy. **No single y-stat wins both.** Filed as BR10:
   richer representation needed (morphology, intensity, learned head).

4. **Production rule unchanged.** `min_y_half` (current default) is
   the overall-best on the dev surface and was retained. Two
   negative results recorded in BRneg-series so future-Claude
   doesn't relitigate.

5. **The pipeline barely beats a constant-prediction baseline on the
   healthy bucket.** Predict "1 mm for everyone" → healthy bucket MAE
   0.445; production → 0.451. **The pipeline earns its keep on
   non-healthy cases, not on healthy.** This reframes what the
   pipeline is *for*.

---

## Honest performance picture on dev

Dev split = 150 images, 231 scored sites. Held-out split = 50 images,
**not touched**.

### Headline

| Cut | n | mean | median | p90 |
|---|---|---|---|---|
| All scored sites | 231 | 0.723 | 0.402 | 1.706 |
| **'Honest-visible' subset** (gt<6 AND not anterior×severe) | 205 | **0.543** | **0.345** | 1.331 |
| **Posterior-view subset** | 183 | **0.655** | **0.345** | 1.475 |
| Anterior-view subset | 48 | 0.982 | 0.634 | 2.190 |

### By severity (all sites)

| Bucket | n | mean | median | max |
|---|---|---|---|---|
| healthy (gt<2) | 129 | 0.451 | 0.293 | 2.979 |
| mild (2-4) | 58 | 0.656 | 0.334 | 2.897 |
| moderate (4-6) | 25 | 0.904 | 0.655 | 3.847 |
| severe (6-8) | 11 | 1.973 | 1.706 | 5.245 |
| extreme (≥8) | 8 | 3.310 | 2.048 | 8.252 |

### By arch (DenPAR metadata)

| Arch | n | mean | median |
|---|---|---|---|
| Lower (mandibular) | 155 | 0.776 | 0.402 |
| Upper (maxillary) | 76 | 0.615 | 0.404 |

Extreme-bucket sites are all Lower-arch. Severe bucket: 6 Lower, 5 Upper.

---

## Commits made tonight

| Hash | Subject | What it adds |
|---|---|---|
| [cb61218](https://github.com/lazyreflect/dental-rad-cli/commit/cb61218) | feat(eval): lock held-out split | 150 dev / 50 held-out + gating |
| [bfd3918](https://github.com/lazyreflect/dental-rad-cli/commit/bfd3918) | feat(diagnostics): contact-sheet + 907 dissection + stratification | scripts + finding: heavy-tailed errors, 89% honest-visible mean 0.543 |
| [073aa24](https://github.com/lazyreflect/dental-rad-cli/commit/073aa24) | docs(eval): BRneg-1 record | CEJ-x sampling reverted (metric-gaming, not real fix) |
| [b1dac5f](https://github.com/lazyreflect/dental-rad-cli/commit/b1dac5f) | feat(diagnostics): BR9 bone-mask extent | 62% of severe sites have masks reaching deep — algorithm fix-space confirmed |
| [26ff399](https://github.com/lazyreflect/dental-rad-cli/commit/26ff399) | feat(eval): BR4 + BR6 + BRneg-2 sweep | dumb baseline + arch/site stratification + 6-rule landmark sweep negative result |

---

## Open BR-items (sequenced by your call)

| ID | What | Joseph cost | Compounding value |
|---|---|---|---|
| BR3 | Hand-curate ~30 dev GT sites (rate clinically correct vs suspect) | 30-45 min | Tightens the GT-noise vs model-error split; bridge to BR10 |
| BR5 | `root_class` accuracy validation | 0 (auto) | Enables tooth-class-aware calibration; bounded |
| BR7 | SAM2 + LoRA timeboxed spike | 1 week author time, 0 Joseph | The bitter-lesson check; could obsolete the architecture ladder |
| BR8 | DenPAR patient-level audit | 30 min — read companion paper / contact authors? | Critical: if dev/held-out share patients, the lock is leaky |
| BR10 | Richer-representation landmark inference (morphology / intensity / learned head) | multi-day | Only path to fix the severe bucket per BRneg-2 |
| BR11 | Reframe product narrative away from "beat Overjet 0.3" | 1-2 hr writing | Aligns external comms with the honest numbers |

---

## What Karpathy would say

The night's discipline held: predict before measuring, ship measured
wins, log negative results, don't argue with the data. The two
biggest learnings are structural:

1. **The architecture is more competitive than the headline 0.723
   number suggests.** Strip out 11% of the data (anterior+severe
   suspects) and we're at 0.543 mean / 0.345 median — Pearl-PA-class
   accuracy on a public dataset they don't even test on.

2. **The remaining gap to Overjet's claim is half marketing, half
   real.** The marketing half: their corpus is curated, ours is
   not. The real half: the severe-perio under-prediction needs
   richer features than mask + y-statistic, OR explicit
   uncertainty calibration for hidden-defect cases (radiographic
   sensitivity ceiling of 0.22 for 3-walled defects per
   [PMC12182397](https://pmc.ncbi.nlm.nih.gov/articles/PMC12182397/)).

3. **The bitter lesson is sitting in the room.** Seven architecture-
   ladder commits across the project have produced incremental gains
   on a corpus that may already be near the radiographic-information
   ceiling for the current method. A timeboxed SAM2 + LoRA spike
   (BR7) is the experiment that determines whether the ladder is
   exhausted or just well-tuned.

---

## Recommended next session start

1. Read this doc (5 min).
2. Decide which of BR3 / BR5 / BR7 / BR8 / BR10 / BR11 to prioritize.
   - My call if forced: **BR8 first** (patient-level audit — cheap,
     blocks held-out reporting), **BR11 next** (reframe narrative
     while the work is fresh), **BR7 last** (the big spike).
3. Don't touch held-out yet (still locked at 50 imgs untouched).
4. If you want to validate any specific severe-bucket case from
   tonight, the dissection PNGs are in
   `output/diagnostics/severe-sites/` (gitignored; regenerate via
   `scripts/dissect_severe_sites.py`).

The architectural ladder is approximately at its single-rule ceiling
on DenPAR. The interesting work going forward is either (a)
richer-representation landmark inference (BR10), or (b) different
foundation (BR7), or (c) different framing (BR11). All three are
substantial; none is "fix one more thing in family_a.py."
