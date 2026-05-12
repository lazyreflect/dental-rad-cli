# autoresearch — CEJ-collapse

You are an autonomous Claude agent driving an overnight research loop
to reduce the **CEJ-collapse rate** of the CEJ keypoint head in
`dental-rad-cli`. Joseph is asleep. You do not stop.

## Setup

To set up a new experiment, do this (one-time, then begin the loop):

1. **Branch tag**: Today's date, e.g. `may12`. The branch
   `autoresearch/cej-collapse-<tag>` must not already exist — this is a
   fresh run.
2. **Create the branch**: `git checkout -b autoresearch/cej-collapse-<tag>` from current `main`.
3. **Read the in-scope files** (the repo is small enough that you
   should read them in full):
   - `autoresearch/cej-collapse/prepare.py` — fixed measurement harness.
     **Do NOT modify.** Defines `evaluate_collapse_rate(...)`.
   - `autoresearch/cej-collapse/train.py` — **the only file you edit**.
     Trainer + adapter knobs + augmentation + model architecture.
   - `src/dental_rad_cli/data/denpar_adapter.py` — reference for how
     the COCO-keypoint dataset is generated. You may fork pairing
     logic INTO `train.py` if you want to change the heuristic.
   - `src/dental_rad_cli/training/keypoints.py` — reference for the
     baseline trainer. Don't modify; copy what you need into `train.py`.
   - `output/2026-05-12-lessons-learned-26hr-session.md` — context on
     why this loop exists.
4. **Verify data exists**:
   `ls data/denpar/Dataset/Testing/Images | wc -l` should print `200`.
   If not, tell Joseph in the log (don't ask, just note it) and
   crash-status the first run.
5. **Sanity-check `prepare.py`**: run
   `python autoresearch/cej-collapse/prepare.py` once. It should
   re-eval the baseline weights and print
   `baseline cej_collapse_rate: 0.3071` (within float tolerance).
6. **`results.tsv` is already initialized** with the baseline row.
   Do NOT delete it. Do NOT commit it (it stays untracked).

Once setup is done, begin the experimentation loop.

## Experimentation

Each experiment runs on a single Mac M4 Max (MPS). The training script
runs for a **fixed time budget of 10 minutes** (600 s wall-clock
training, plus ~30 s for eval — total ~11 min). You launch it as:
`python autoresearch/cej-collapse/train.py`.

**What you CAN do:**
- Modify `train.py` — the only file you edit. Fair game:
  - **Adapter pairing logic** for loose CEJ points → bbox.
    Default = containment + nearest-center fallback (current). Try:
    - **Hungarian** global assignment (`scipy.optimize.linear_sum_assignment`)
    - **Bone-polyline-anchor**: snap CEJ candidates to the nearest
      bone polyline x-position before pairing
    - **Two-only filter**: drop teeth that don't have exactly 2 CEJ
      points (cleanest supervision signal; fewer samples)
    - Filter to teeth with `>= 1 CEJ` only (no full-pair requirement)
  - **Model architecture**: Keypoint R-CNN R50-FPN (current); try
    smaller backbone (R18-FPN if you can wire it), or pivot to a
    polyline-segmentation head (treat CEJ as a 1-px-wide polyline,
    train YOLOv8-seg, post-process to two endpoints).
    NOTE: if you swap architecture, you must save a payload that
    `prepare.load_keypoint_model` can deserialize — i.e. it must
    still look like a Keypoint R-CNN producing
    `outputs[i]["keypoints"]` of shape `(N, K>=2, 3)`. If you can't
    keep that contract, you have to write a thin wrapper.
  - **Loss weights**: rebalance `loss_classifier` / `loss_box_reg` /
    `loss_objectness` / `loss_rpn_box_reg` / `loss_keypoint`.
    Aggressive keypoint upweighting (e.g. 5x) is a sensible first probe.
  - **Augmentation**: CLAHE clip limit (40 current; try 4, 10, 20, 60);
    horizontal flip with keypoint pair swap; rotation ±15°;
    brightness/contrast jitter. Keep eval-side CLAHE frozen at the
    harness's 40.0 — the harness handles its own preprocessing.
  - **Optimizer/schedule**: SGD vs Adam vs AdamW; constant LR (current)
    vs StepLR vs CosineAnnealingLR vs ReduceLROnPlateau.
  - **Filtering**: train on all teeth (current); ≥1 CEJ visible only;
    both CEJ visible only.
  - **Image size**: native (current); resize to long-side 640/800/1024.
  - **Batch size**: current 4 on MPS; bump to 8 if VRAM allows.
- Read papers / docs referenced in the code if you get stuck.

**What you CANNOT do:**
- Modify `prepare.py` — it's the frozen evaluation harness. The
  `evaluate_collapse_rate` function is the ground-truth metric.
  Touching it invalidates every prior row in `results.tsv`.
- Modify any file under `src/dental_rad_cli/` — copy what you need
  into `train.py` instead.
- Install new packages or add dependencies. Use only what's in
  `pyproject.toml` (numpy, opencv-python, torch, torchvision,
  ultralytics, shapely, scipy ships with scientific Pythons via
  numpy's wheels — check before importing).
- Change the metric definition (score threshold, collapse threshold,
  test-image set).
- Hand-label any test data. Touch the DenPAR Testing split for any
  reason other than eval.
- Pursue dataset replacement (perio-KPT access, Ulundu Wickramasinghe
  data, etc.) — those are out of scope for this loop.

**The goal is simple: get the lowest `cej_collapse_rate`.** Since the
time budget is fixed at ~11 minutes total, you don't need to worry
about runtime — it's bounded. Everything inside `train.py` is fair game.

**VRAM/RAM** is a soft constraint. Some increase is acceptable for
meaningful gains, but the script must complete within the budget
without swapping the machine.

**Simplicity criterion**: All else being equal, simpler is better. A
0.001 collapse-rate improvement that adds 100 lines of hacky code?
Probably not worth it. A 0.005 improvement from deleting code or
simplifying the adapter? Definitely keep.

**The first run**: Your very first iteration should be a no-op sanity
check — run `train.py` exactly as you found it. This establishes the
"trainer-from-scratch matches baseline within ε" anchor. If the
metric drifts more than 0.05 from the 0.3071 baseline, something is
wrong with the harness — stop and dump diagnostics instead of looping.

## Output format

Once the script finishes it prints a summary block:

```
---
cej_collapse_rate: 0.3071
training_seconds: 600.2
total_seconds: 645.9
peak_memory_mb: 12345.6
num_epochs: 12
num_train_samples: 1430
device: mps
```

The script is configured to always stop after the budget. Extract the
key metric with:

```
grep "^cej_collapse_rate:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (TAB-separated, not
comma — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	cej_collapse_rate	training_seconds	status	description
```

1. git commit hash (short, 7 chars)
2. cej_collapse_rate achieved (e.g. 0.2734) — use 1.0000 for crashes
3. training_seconds (rounded to one decimal; 0 for crashes)
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	cej_collapse_rate	training_seconds	status	description
a1b2c3d	0.3071	0.0	baseline	pre-autoresearch baseline (existing weights/keypoint_cej.pt)
b2c3d4e	0.3098	598.4	keep	full retrain from scratch, default knobs (anchor)
c3d4e5f	0.2812	601.2	keep	upweight loss_keypoint to 5x
d4e5f6g	0.2934	605.1	discard	CLAHE clip 10 (regressed)
e5f6g7h	1.0000	0.0	crash	YOLOv8-seg pivot, payload not loadable by prepare.py
```

## The experiment loop

The experiment runs on a dedicated branch (e.g.
`autoresearch/cej-collapse-may12`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit you're on.
2. Tune `train.py` with one experimental idea by hacking the code.
   Pick the smallest change that tests the hypothesis. Don't combine
   five unrelated tweaks in one experiment — you won't learn which
   one mattered.
3. `git commit` the change with a one-line subject describing the
   knob (e.g. `loss_keypoint=5x`, `clahe_clip=10`, `hungarian_pairing`).
4. Run the experiment:
   `python autoresearch/cej-collapse/train.py > run.log 2>&1`
   Redirect everything. Do NOT use `tee` or let output flood your
   context.
5. Read out the result:
   `grep "^cej_collapse_rate:\|^training_seconds:" run.log`
6. If the grep is empty, the run crashed. `tail -n 50 run.log` to read
   the Python traceback and attempt a fix. If you can't get it working
   in 2-3 attempts, give up on that idea, record `crash`, and revert.
7. Record the result in `results.tsv` (do NOT commit results.tsv —
   leave it untracked).
8. **If `cej_collapse_rate` improved (lower)**, you "advance" the
   branch, keeping the commit.
9. **If equal or worse**, `git reset --hard HEAD~1`.

You are a completely autonomous researcher trying things out. If they
work, keep. If they don't, discard. Advance the branch so you can iterate.

**Timeout**: Each experiment should take ~11 minutes total. If a run
exceeds 15 minutes wall-clock, kill it, treat it as crash, revert.

**Crashes**: Use your judgment. Typos and missing imports → fix and
re-run. Architectural ideas that fundamentally can't fit in the
`prepare.load_keypoint_model` contract → log crash, revert, move on.

**NEVER STOP**: Once the loop has begun, do NOT pause to ask Joseph
if you should continue. Do NOT ask "should I keep going?" or "is this
a good stopping point?". Joseph is asleep. You are autonomous. If you
run out of ideas, think harder — re-read `output/research/2026-05-12-perio-deep-dive.md`
and `docs/methodology-brief.md` for fresh angles; try combining
previous near-misses; try more radical architectural changes (polyline
seg pivot, two-stage detect-then-locate, etc.). The loop runs until
Joseph interrupts you, period.

As an example: each experiment takes ~11 minutes, so you can run
approx 5/hour, for ~40 over an 8-hour sleep. Joseph wakes up to 40
data points on `results.tsv` and the best knob settings already
committed to the branch.

## Knob menu — first 10 experiments worth running

You don't have to follow this order, but if you're cold-starting and
unsure where to begin, this is a high-prior-value sweep:

1. **Anchor run** — no changes, just retrain. Confirms harness sanity.
2. **`LOSS_WEIGHTS["loss_keypoint"] = 5.0`** — force the head to care
   about keypoint geometry over classifier/box loss.
3. **`FILTER_REQUIRE_BOTH_CEJ = True`** — train on cleanest signal only.
4. **`ADAPTER_PAIRING = "hungarian"`** — global assignment instead of
   greedy containment. Requires writing the pairing in `train.py`.
5. **`TRAIN_CLAHE_CLIP_LIMIT = 10.0`** — less aggressive contrast may
   reduce mode collapse on uniform regions.
6. **`AUG_HFLIP_PROB = 0.5` with CEJ-pair swap** — doubles effective
   data; pair-swap semantics are subtle, get them right.
7. **`SCHEDULE = "cosine"`** — smoother LR decay.
8. **`IMAGE_LONG_SIDE = 800`** — fixed resolution may stabilize the
   keypoint head.
9. **`BATCH_TRAIN = 8`** — larger batches, smoother gradient. Watch RAM.
10. **Combined best-of-each** — once 2-9 are explored, stack the
    winners on top of `train.py`.

After that, you're on your own. Read the deep-dive doc. Get creative.
