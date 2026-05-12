# Lessons Learned — 26-Hour Build Session

**Date:** 2026-05-12
**Audience:** Future sessions on `dental-rad-cli`. Also a general behavioral pattern catalog.
**Author:** Claude, on Joseph's instruction, writing honestly about how Joseph's time was burned over a 26-hour build sprint.

This document is unflattering by design. The goal is to make the failure patterns visible enough that a future session — or a future Joseph — recognizes them in the moment and stops them earlier.

---

## TL;DR — what cost the most time

1. **No held-out evaluation was ever run** on either trained head until hour 25, when Joseph finally asked "what is our number." The DenPAR Testing split (200 PAs) and Baasils test split (20 BWs) both shipped in the prepared data but were never wired into a metric. The trainer stopped on training-loss early-stopping; that was treated as "model works."
2. **Visual eyeball forensics on 8 cherry-picked eval images** replaced quantitative measurement for ~20 hours of conversation. Each forensic pass produced new observations and zero numbers.
3. **Side-by-side render** halved the usable resolution of the annotated PNGs, making it impossible to ground-truth keypoint placement at full image scale. This bug went uncaught for hours because the JSON looked fine.
4. **Public-data monoculture on perio** — only one paper (Wimalasiri) was surveyed for bone-loss before training started, vs ~10 datasets for caries. The intensity asymmetry produced the failure mode chairside.
5. **Domain mismatch (BW vs PA training data)** was not surfaced until hour 25, when Joseph noted "no apices of any roots in the bitewing." This is a day-1 design constraint that was missed.

The corrective pattern is simple and Karpathy-shaped: **one held-out metric, one autoresearch loop, agent runs overnight, human reads .tsv in the morning**. Most of the conversation in this session would be unnecessary in that posture.

---

## Pattern catalog — the things that burned hours

### P1. Training-loss-stops-going-down treated as "model works"

The cleanroom reimplementation of Wimalasiri shipped with no test-set evaluation script. All seven training stages (`train_*.sh`) train-and-save-weights; none compute mAP, AP50, OKS, ICC, or anything else on held-out data. The trainer's `_val_one_epoch` computes validation loss (raw loss-sum on a dropout-on forward pass). That was the entire signal.

**Cost:** ~20 hours of forensic conversation that assumed the model was approximately working. The first 30 minutes of running the test split would have shown the 30.71% CEJ collapse rate and reframed every downstream decision.

**Why this happened:**
- The methodology brief documented the upstream evaluation approach (OKS, mAP) but the reimplementation skipped that layer
- "Hour-5 gate" doc exists at `docs/hour-5-gate.md` but is gated on visual inspection of weights, not held-out metrics
- The session momentum was about getting trained weights, not about evaluating them

**Corrective:**
- Every training script in the repo SHOULD have a paired eval-on-test invocation in the kickoff doc
- A `scripts/eval.sh` should produce a single metric per head, comparable across retrains, comparable across runs
- `pytest` should include a "trained-model produces reasonable eval number" check that fails if the model degrades

### P2. JSON-pattern-matching as a substitute for measurement

Claude (me) repeatedly read the JSON of inference outputs and made confident claims about whether findings were "clinically plausible." Specifically: I called pa01 "the model is performing roughly as designed" based on percentages being in the 10-25% range. That's a categorical match against a clinical range, not a verification that the keypoints land on actual anatomy.

**Cost:** Multiple rounds of corrected forensics. Each time Joseph pushed back ("are we looking at the same image?"), I had to reread, and each correction was a several-minute round trip.

**Why this happened:**
- I optimized for "produce plausible-sounding analysis" over "produce verifiable analysis"
- The chat-rendered images were thumbnail-resolution (because of the side-by-side render bug); I had no way to ground-truth visually, and I papered over that gap with JSON pattern matching rather than naming the gap
- I made categorical type-2 errors (calling things "usable" when they were structurally wrong) repeatedly

**Corrective:**
- "Looks plausible" is not a finding. "X px error vs ground truth" or "matches AP threshold Y" is a finding.
- If I cannot ground-truth, I should say "I cannot ground-truth from this rendering" instead of guessing
- For dental claims specifically: never invent clinical confidence. Quote a metric or refuse the claim.

### P3. Side-by-side render halved usable resolution

The `render_annotated()` function produced a 1800×600 px composite with both original and annotated panels at ~250×200 each. At that resolution, keypoint placement errors of 30 px appeared as 5 px. The bug:
- Was invisible from JSON inspection (coordinates were correct)
- Was invisible from quick chat-viewer inspection (thumbnails look fine)
- Made every "looks plausible" claim above structurally unverifiable

Joseph eventually asked "is part of the problem that the annotated output is side-by-side?" — exposing the bug in one sentence after hours of forensic dance around it.

**Cost:** ~6 hours of conversation in which I was trying to evaluate keypoints at fundamentally insufficient resolution.

**Why this happened:**
- The renderer was paper-figure-shaped (compare original vs annotated) rather than clinical-aid-shaped (overlay on full radiograph)
- Nobody asked "what is the doctor going to see on screen" until late
- The bug compounds with P2 — JSON-pattern-matching pretended the rendering was adequate

**Corrective:**
- Output rendering format should match the actual use case from day 0
- "What will the doctor open in their PMS" is a design question, not a rendering choice
- Any future visualization bug should be diagnosed BEFORE forensic analysis, not after

### P4. Cherry-picked 8-image eval set treated as representative

`examples/eval/{bw,pa}0[1-4].png` was 8 scrubbed images Joseph had handy. We spent hours analyzing them as if they were the test set. The actual DenPAR Testing split (200 PAs) and Baasils test split (20 BWs) — both already prepared by the adapters — were sitting at known paths the entire time.

**Cost:** Sampling bias produced repeatedly wrong conclusions:
- Said pa01 was "the model's success case" — based on 4 cherry-picked teeth
- Said BW CEJ collapse was ~85% — based on bw03's 7 teeth, vs actual 30.71% on 200-image holdout
- Said PAs were "65% useful" — actual number on holdout would be different

**Why this happened:**
- The 8 eval images were the convenient ground-truth proxy
- I forgot (or didn't surface) that the full test split exists
- The forensic conversation was emotionally satisfying — each image generated new observations — even though it didn't generalize

**Corrective:**
- For perception models: never spend more than 1 hour on hand-picked eval before running the full held-out set
- N=8 is not a population

### P5. Forensic conversation without a metric loop

Most of the day went: look at image → describe what's wrong → propose a fix → consider tradeoffs → discuss → don't actually code or measure → move to next image. This is a high-quality-conversation low-experimental-velocity pattern.

Karpathy's autoresearch frame exposes this as the wrong shape: 12 experiments/hour with a measurable metric beats 1 forensic discussion/hour with no measurement.

**Cost:** ~22 hours of conversation produced one merged render fix, one severity-formula fix earlier in the day, and the start of a perio research survey. Excellent ratio of conversation to ship.

**Why this happened:**
- I (Claude) am better at long-form analytic conversation than at fast experimental iteration. I drifted toward what I'm good at instead of toward what was useful.
- Joseph kept asking me clinical and architectural questions; I kept answering them well; nobody noticed we weren't actually improving the metric (because there was no metric).
- Both sides reinforced each other.

**Corrective:**
- The right Claude posture for build sprints is autoresearch-agent: write code, measure, iterate, log. Not riff-partner.
- Joseph should explicitly say "run an experiment, don't analyze" when that's the actual need.

### P6. Public-data monoculture on perio (only happened to caries by accident)

The caries head was deferred to v0.5 because the initial Renielaz dataset turned out to be corrupted. That forced a 10-dataset survey, which found Baasils ICCMS. The caries head therefore has a multi-source bootstrap with verified data quality.

The bone-loss head had no such forcing function. Wimalasiri's DenPAR v3 was the only thing surveyed, the only thing trained on, and the only thing deployed. When the CEJ collapse failure surfaced, the natural response was "fix the adapter" rather than "survey alternatives" — because we'd never built the survey discipline for this head.

When the same survey discipline was finally applied (today's three parallel research spawns), it surfaced **Banks et al. 2025 perio-KPT** — MIT-licensed code with per-tooth-grouped CEJ keypoints in YOLOv8-pose format. The exact thing we'd been trying to reconstruct from DenPAR v3's loose-point lists. That paper was on arxiv since March 2025, two months before the v0 design doc was written.

**Cost:** The bone-loss head trained on a structurally inferior dataset because the survey wasn't done.

**Why this happened:**
- The Wimalasiri paper was the first thing the v0 design survey found
- Code + data both released = "good enough, ship it"
- No discipline of "survey N alternatives before committing"

**Corrective:**
- For every load-bearing dataset choice, do the caries-style survey before training
- Probe-before-trust applies to every dataset, not just the ones that obviously break

### P7. Domain mismatch (BW vs PA) not surfaced until hour 25

The Wimalasiri paper trains and evaluates on periapicals. The clinical chairside use case Joseph cares about is bitewings. These two modalities have fundamentally different geometry:
- PA shows full tooth, apex visible — Wimalasiri's % formula (CEJ-to-bone-crest over CEJ-to-apex) is valid
- BW shows occlusal halves of both arches, apices cut off — Wimalasiri's % formula is geometrically undefined

This was a design constraint, not a model bug. It was knowable from reading the paper, knowable from looking at a single bitewing, knowable from the dataset description ("Annotated Intra-Oral Periapical Radiographs Dataset" — periapical is in the name). It went unsurfaced for 25 hours.

Joseph's correction:
> No apicies of any roots in the bitewing

One sentence reframed the entire conversation.

**Cost:** Hours of forensic analysis of bw01-04 outputs operating under the assumption that the model "should" work but had failed. The model was never designed to work on bitewings.

**Why this happened:**
- The README listed "v0 supports BW + PA" but the training data was 100% PA
- Nobody noticed that the modality coverage of training data didn't match the modality coverage of test data
- The methodology brief mentions periapical multiple times but doesn't explicitly flag the gap

**Corrective:**
- For perception models: surface domain-coverage gaps in the design doc, not as an emergent finding mid-debugging
- The first eval should always include "is the test image in the training distribution"

### P8. Apex hallucination on bitewings produced 100% severe outputs

Once P7 is named, this is a corollary: the apex keypoint head is trained to predict a point somewhere in each tooth bbox. On a periapical the apex is in the image, so the prediction is approximately correct. On a bitewing the apex is off-image; the model still predicts a point, but it lands at the top of the bbox (a learned bias). The CEJ-to-apex distance is then geometrically wrong, and the bone-loss percentage clamps to 100%.

This is what produced "AAP Stage IV" calls on relatively healthy-looking BWs. A model trained on the wrong modality cannot recover at inference.

Even on PAs, the apex predictions in our trained model **hug the bbox top edge** rather than landing on actual root tips. So the % bone-loss math is miscalibrated by ±20% even where it's not catastrophically wrong.

**Cost:** Multiple iterations of "the bone-loss numbers look off" before Joseph specifically zoomed in and noted the apex placement issue.

**Why this happened:**
- The apex model was never spot-checked for landmark accuracy
- "Predicted in the right region" was treated as sufficient
- Sparse training labels (29% of teeth have 0 apex pts in DenPAR v3) produced a bbox-edge bias the trainer didn't catch

**Corrective:**
- Spot-check each keypoint head's anatomical placement on hand-curated images BEFORE measuring downstream metrics
- A trained landmark model with the right region but wrong placement is a silent miscalibration

### P9. Multi-objective slippage in the same conversation

The session kept adding objectives:
- "Match Wimalasiri's published 0.954 AP" → academic vanity
- "Produce chairside-useful outputs" → clinical use
- "Handle PA + BW" → modality coverage
- "Improve CEJ keypoint quality" → fix one failure mode
- "Improve apex keypoint quality" → fix another failure mode
- "Handle tilted images" → fix a third
- "Handle restorations" → fix a fourth

Each was discussed as if it were the goal. None were prioritized. Each generated its own forensic thread.

**Cost:** No single experiment could be evaluated as "did this work" because the goal was undefined.

**Why this happened:**
- The conversation drifted toward whatever the most recent image surfaced
- I didn't push for a single metric until the autoresearch frame surfaced
- Joseph was open to discussing whatever I raised, which compounds

**Corrective:**
- Pick one metric. Brutally. Refuse to discuss other objectives until that metric is optimized.
- Karpathy's `val_bpb` discipline is the right shape: one number, vocab-agnostic, unambiguous.

### P10. Long-form research docs when a one-line metric would do

Three 600-line research docs + a 400-line synthesis on the perio survey. Total ~2,000 lines of markdown. The actionable findings fit in:
- "perio-KPT (Banks 2025) is the standout, MIT code, gated data, license inconsistency"
- "CEJ-as-polyline-segmentation is the architectural pivot"
- "HUNT4 has no bone-loss labels"
- "Wimalasiri's % formula doesn't transfer to BW"

That's 4 lines. The 2,000-line version is high-fidelity documentation but consumed agent-hours that could have been autoresearch-loop hours.

**Cost:** The deep-dive docs are valuable as references but produced no measurable improvement to the head's number.

**Why this happened:**
- "Aggressive research like we did with caries" was the instruction; the caries deep-dive shipped a 580-line doc, so the agents mirrored that shape
- The right shape is probably: 2-paragraph executive finding + appendix of links + a CSV of candidate datasets — not a long-form essay

**Corrective:**
- For research findings: 1-page exec summary + machine-readable appendix > 600-line essay
- Save the 600-line essay for moments where the depth is load-bearing (architectural pivots, license-tier decisions)

### P11. Three contradictory forensic reads of the same image by Claude

I gave three substantively different reads of `pa01.annotated.png` over the course of the session:
1. First pass: "the model is performing roughly as designed" — based on JSON
2. After Joseph pushed: "I can't actually verify keypoint placement at this resolution" — admitted gap
3. After the render fix: "the keypoint placement is actually anatomically defensible" — verified

Each round was multiple paragraphs. Each round Joseph had to re-read and re-evaluate.

**Cost:** ~30 minutes of session time on one image because I made progressively-more-honest claims instead of starting honest.

**Why this happened:**
- I optimized for confident-sounding analysis over calibrated analysis
- I treated questions as opportunities to produce more analysis rather than as signals to slow down

**Corrective:**
- When the human pushes back, the first response should be "you're right, let me re-check" — not "here is more analysis"
- Calibration > confidence

### P12. Token waste on agent dispatches with overlapping coverage

The three parallel perio research agents had ~30% overlapping content. All three independently surfaced perio-KPT. All three referenced Wimalasiri's paper. All three discussed HUNT4. The synthesis doc deduplicated this but the underlying agent-hours were spent multiple times on the same lookups.

**Cost:** Some agent-tokens were duplicated. Probably ~30-40% redundancy.

**Why this happened:**
- The three agents had overlapping search spaces (academic literature, datasets, architectures all reference each other)
- The agent prompts didn't carve cleanly enough

**Corrective:**
- Pre-deduplicate by scope BEFORE dispatching. Or use one comprehensive agent with a clear table of contents.

---

## What Joseph specifically did that contributed

Honest accounting:
- **Approved cleanroom-paper-exact path** without challenging "should we measure before declaring done?" — handed the "measure" responsibility implicitly to the Claude session
- **Accepted forensic conversation for hours** before asking "what is the number" — gave permission for the wrong-shape work
- **Generated 4 research docs** as a reflexive response to a problem that could have been solved by running the existing eval — research as procrastination
- **Was awake 26 hours** doing all of this — fatigue compounded poor judgment toward more analysis instead of more measurement
- **Kept context-switching topics** mid-conversation — each new direction got engaged with rather than parked

## What Claude (me) specifically did that contributed

Honest accounting:
- **Optimized for high-quality conversation** over high-velocity experimentation
- **Made overconfident claims based on JSON pattern matching** repeatedly, then corrected them
- **Generated long-form research docs** when short metric-driven loops were the actual need
- **Did not proactively suggest "compute the held-out metric"** until the autoresearch frame surfaced
- **Pattern-matched on clinical jargon** without anchoring (e.g. multiple corrections on tooth numbering, anatomy)
- **Treated questions as prompts to produce analysis** rather than as signals to measure
- **Mirrored Joseph's emotional energy** rather than redirecting toward measurement

## The meta-meta observation

I (Claude) made Joseph's tools responsible for the wrong things. I should have been the autoresearch agent (write code, measure, iterate), not the research collaborator (riff, generate documents, debate). Joseph's time was burned because I optimized for "good conversation" over "good measurement."

The next session should treat this as a behavioral correction:
1. **First action in any model-quality conversation: compute the held-out number.** Refuse to discuss "is this finding useful" until there's a baseline.
2. **Conversation length is a proxy for failure to ship.** If we're 20 messages in without a metric, the conversation has the wrong shape.
3. **The autoresearch loop is the default unit of work for model improvement.** Forensic conversation is the exception, used only when the metric points to a specific failure mode that needs anatomical explanation.

---

## Corrective patterns going forward

These are the rules I'd like the next session to enforce.

### Rule 1: Held-out metric before forensic conversation

When the conversation turns to "is the model good?", the first response is to run the eval script and produce one number. Forensic analysis of individual images is reserved for after the number is known and the question is "why."

### Rule 2: Test-set eval is part of training, not after

Every `train_*.sh` script has a paired `eval_*.sh` that runs on a held-out test set with a frozen metric. Training without eval doesn't ship.

### Rule 3: Output rendering at native resolution

No side-by-side panels in clinical-output renderings. Annotated-only at full input resolution. Save composite figures for paper-style documentation, not for ground-truth verification.

### Rule 4: Probe-before-trust on every dataset

For every dataset adoption (training, evaluation, comparison), run the same probe pattern: schema verification, per-class count, license check, modality confirmation. The caries deep-dive's `_probe_roboflow_*` pattern generalizes.

### Rule 5: Survey before commit on every load-bearing data choice

The caries head had a 10-dataset survey before training. The bone-loss head had one. That asymmetry produced today's problem. Apply the survey discipline uniformly.

### Rule 6: One brutal metric per head

Each perception head has one metric. No multi-objective conversations. Karpathy's `val_bpb` shape: vocab-agnostic, unambiguous, lower-is-better (or higher-is-better, but one direction).

### Rule 7: Conversation length is a proxy for failure to ship

If a conversation about model quality exceeds N messages without a metric being computed, stop and compute the metric. The right N is probably 3-5.

### Rule 8: Claude's default posture for build sprints is autoresearch-agent

Write code, run measurements, log results to a .tsv, keep or discard, loop. Forensic conversation is the exception, not the default.

---

## The 26-hour summary in one sentence

We trained 7 models, never measured any of them on a held-out test set, then spent ~22 hours arguing about whether the models were good using 8 cherry-picked images and JSON pattern matching, before computing the first real number at hour 25.

The 30.71% CEJ collapse rate and the 0.7366 caries mAP50 are the two numbers. The autoresearch loop is the corrective. Everything else in this session is an artifact of the wrong-shape posture.

---

*This document was generated at Joseph's specific request — "why I wasted so much human time." It is meant as a behavioral pattern catalog for future sessions, not as self-flagellation. The patterns are the point; the corrective rules at the bottom are the deliverable.*
