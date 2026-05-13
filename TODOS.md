# TODOS

Carry-forward items from hour-0 v0 / v0.5 work. Organized by source so
context isn't lost. Each row names a trigger condition — these are not
"do soon," they're "do when this fires."

## Benchmark-rigor follow-ups (BR-series)

After the 2026-05-12 evening pivot to "exceed Overjet benchmark" framing,
Karpathy's discipline applied to current state surfaced these. Sequenced
in priority order — do BR1 → BR2 → BR3 before tuning architecture
further.

| ID | Status | Description | Trigger |
|----|--------|-------------|---------|
| BR1 | **DONE** | Lock 50-image DenPAR held-out split (seed=42). Default `benchmark_eval.py --split=dev`; held-out gated behind `--confirm-held-out-touch` + log entry in `splits/HELD_OUT_TOUCHES.md`. Dev mean MAE on 150 imgs = 0.723 (was 0.705 on 200; honest correction after un-leaking test set). | DONE 2026-05-12 |
| BR2 | **DONE** | Contact-sheet generator + visual inspection + stratification by GT severity bucket. Found heavy-tailed error distribution; 89% "honest-visible" subset has mean 0.543 / median 0.345 mm. 907 dissection revealed algorithm-vs-segmentation failure modes. | DONE 2026-05-12 |
| BR3 | OPEN | Sample N=50 random dev sites, rate each GT placement (clinically correct / suspect-anterior / suspect-other / unclear) by visual inspection. Compute label-noise-corrected MAE. Anchors how much of the 0.723 mean is the model vs the labels. Likely requires ~30 min of Joseph time. | After BR2 (need contact-sheet to inform sampling) |
| BR4 | **DONE** | Dumb baseline floor via constant-prediction MAE on dev JSON. Constant=0 → MAE 2.540; constant=median(1.82mm) → MAE 1.477 (optimal constant-prediction MAE); production = 0.723. **Pipeline buys 0.754 mm over the constant-median floor.** Surprise: on the healthy bucket alone, "predict 1 mm for everyone" gives MAE 0.445 — basically matches pipeline's 0.451. The pipeline earns its keep on mild/moderate/severe, NOT on healthy. | DONE 2026-05-12 |
| BR5 | OPEN | `root_class` accuracy validation: tooth detector outputs single/double/unknown but never validated against FDI-derived expected. Build cross-check on DenPAR Testing (using tooth-wise masks + FDI inference) before tuning anything that depends on `root_class` (e.g. A1 calibration). ~50 LOC. | Before A1 tooth-class-aware calibration |
| BR6 | **PARTIAL DONE** | Stratify dev errors by DenPAR Arch + Site metadata (per `Characteristics of radiographs included.xlsx`). Anterior views (n=48): MAE 0.982. Posterior views (n=183): MAE 0.655 / **median 0.345 — within 45 μm of Overjet's 0.300 on the half of the corpus without incisal-edge GT mislabel risk**. Lower-arch severe+extreme buckets carry the worst absolute errors. Per-FDI stratification deferred until tooth-numbering substrate (parallel session) ships. | DONE 2026-05-12 (arch/site); per-FDI on hold for tooth-numbering |
| BR7 | OPEN | SAM2 + LoRA timeboxed spike: 1-week budget. Foundation backbone with light landmark adapter. If it gets within 100 μm of v0.7 architecture ladder with ~100 LOC, the ladder is dead and we pivot. Bitter-lesson check. | After BR3 (if model ceiling well-characterized) AND after Joseph review |
| BR8 | **DONE** | Patient-level audit. Subagent confirmed DenPAR Sci Data 2025 paper reports "440 male + 560 female patients = 1000 radiographs" (1:1 mapping implied). Local image-perceptual-hash audit (`scripts/audit_dev_held_out_leakage.py`) found ZERO near-duplicate pairs among the 200 Testing images (Hamming ≤ 8). Metadata-signature collisions are common-view-across-different-patients, not same-patient. Lock confirmed patient-clean. | DONE 2026-05-12 |
| BR9 | **DONE** | Investigated bone-mask extent at all 34 severe-bucket dev sites. **Result: 62% have `mask_reaches_deep` + 24% `no_bone_at_cej_x` + 12% `mask_short` + 3% `mask_bimodal`.** The bone segmentation IS extending to deep crest in most severe cases; the failure is the landmark-selection rule picking the shallow extreme of a mask that contains the right answer. Algorithm-side fix space confirmed exists. | DONE 2026-05-12 |
| BR10 | OPEN | **Richer-representation landmark inference.** BRneg-2 (rule sweep) confirmed no single y-statistic over bone-on-tooth-ring fixes severe without hurting healthy. The monotonic Pareto frontier (more apical → better severe, worse healthy) implies the fix needs MORE than a y-stat. Candidates: (a) bone-mask morphology — detect bimodal y distribution + branch behavior; (b) bone-pixel intensity — real bone is brighter than artifacts; (c) train a small landmark-prediction head that learns the right y from data. | Before any more bone-landmark algorithmic experiments |
| BR11 | OPEN | **Vendor benchmark framing.** Subagent literature review confirmed no commercial vendor (Overjet, Pearl, Adravision, VideaHealth, Denti.AI) publishes severity-stratified MAE or data-curation criteria. The 0.300 mm Overjet number is a pooled marketing MAE on opaque internal data; AlGhaihab/Denti.AI 2025 explicitly used 39 hand-selected radiographs. Our DenPAR 0.723 with stratification is structurally more rigorous. Goal-framing should change from "beat Overjet's 0.300 pooled mean" to "best public-corpus MAE with severity-stratified reporting." Update product narrative + docs accordingly. | Before any external comm / sales / PR mentions a benchmark number |

## Benchmark-rigor negative results (BRneg-series)

Architectural experiments tried and reverted. Keep so future-Claude doesn't redo.

| ID | Date | Description | Outcome |
|----|------|-------------|---------|
| BRneg-1 | 2026-05-12 | **Sample bone-landmark at CEJ x ± 10..30 px tolerance** instead of "most-coronal apical-to-CEJ in mesial/distal half." Mirrored GT-derivation method. Motivation: 907 middle tooth showed half-tooth rule picks adjacent-tooth contamination. | REVERTED. Dev mean 0.723 → 0.684 (-39 μm) but coverage 76.5% → 70.2% AND severe bucket got WORSE (mean 1.973 → 2.341). Mean improvement came from abstaining on 19 lower-error sites, not from fixing the hard cases. Metric gaming, not architecture improvement. The 907-style contamination isn't representative of the severe-perio failure mode in aggregate. Evidence: `output/training-evidence/benchmark-eval-dev-2026-05-12T211651.json` |
| BRneg-2 | 2026-05-12 | **Landmark-rule sweep across 6 candidates** (`min_y_half`, `median_y_half`, `max_y_half`, `median_y_at_cej_x`, `max_y_at_cej_x`, `wide_aware`). Hypothesis: a different y-statistic over the same `bone_on_tooth_ring` mask would fix severe under-prediction without regressing healthy. | NEGATIVE. Monotonic Pareto: more apical-biased rule → better severe, worse healthy. No rule wins both axes. Production rule `min_y_half` (default, best overall: 0.723 / 0.402) stays. Severe-bucket reduction needs richer features than a single y-stat — filed as BR10. Evidence: `output/training-evidence/sweep-2026-05-12T*/`. |

## Research-report follow-ups (F1-F7)

From `output/research/2026-05-11-caries-v0.5-paths-deep-dive.md` §9.

| ID | Status | Description | Trigger |
|----|--------|-------------|---------|
| F1 | OPEN | Email Salehizeinabadi (corresponding author, IJD 2025 ICCMS BW caries paper) for the underlying labeled dataset, citing the paper in the project README and offering per-class evaluation metrics back from our office's BW pool. Round-trip ~1-2 weeks. | If Baasils v0.5 training underperforms paper's reported mAP, OR if a future v0.x re-train needs more data than Baasils provides |
| F2 | OPEN | Apply for HUNT4 / AI-Dentify dataset access via NTNU Regional Ethical Committee + research protocol. 13,887 expert-annotated BWs would be the gold-standard training set. Timeline: weeks-to-months. | When `dental-rad-cli` scales beyond personal use (e.g. wife's office gets its own deployment; or v1.0 productization considered) |
| F3 | **DONE (2da68dc)** | Promote probe script from `_`-prefixed local-only to `scripts/diagnostics/probe_caries_dataset.py`. Parameterized via env vars so it can vet any future Roboflow dental dataset. | Once v0.5 caries lands |
| F4 | OPEN | Build Label Studio Docker-compose for pickles. Self-hosted annotation environment with ML-backend pre-annotation from a binary detector. Enables active-learning bootstrap on Joseph's own BWs. | If Baasils-trained model accuracy is insufficient on Joseph's BWs at v0.5, requiring §6.2 active-learning fallback |
| F5 | OPEN | Pull Mendeley dataset `4fbdxs7s7w` (100 BWs, 8 annotators) as an **external** held-out test set. Different annotator pool means it measures generalization, not just held-out performance on Baasils' distribution. | After v0.5 ships and bone-loss accuracy gate passes — used as a generalization sanity-check |
| F6 | OPEN | Draft `docs/annotation-policy.md` documenting Joseph's ICCMS calibration: visual guide reference (Salehizeinabadi 2025 supplementary or ICCMS atlas), 20-BW calibration set, edge-case rules ("ambiguous RA3 vs RB4 → annotate as RA3 with confidence flag"). Total prep ~1 hour. | Before any §6.2 active-learning bootstrap session begins. Or if Joseph wants to hand-label more BWs for a fine-tune corpus |
| F7 | OPEN | Revisit DentoMorph-LDM-style synthetic data generation. Train a diffusion model that produces realistic caries-bearing BWs from a seed corpus, then use synthetic augmentation for the deep tier (RC6) where real samples are scarce. | When real-data ICCMS-labeled corpus reaches ~200 BWs (Baasils + Joseph's own labels) — seed size for a usable LDM |

## Tech-debt items flagged during hour-0

| ID | Status | Description | Trigger |
|----|--------|-------------|---------|
| T1 | OPEN | `denpar_adapter.build_yolo_dataset` writes an absolute `path:` into each `dataset.yaml`. Brittle: moving `data/prepared/` breaks every YAML. Switch to relative path or let Ultralytics auto-resolve from the YAML's own location. | Anytime `data/prepared/` would need to move (e.g. multi-machine training, or pickles disk reorg) |
| T2 | OPEN | Add a 5-line smoke test that asserts `prepare_datasets.{sh,ps1}` output dir names match what `train_*.sh` scripts expect. Catches the path-divergence bug class that shipped at 0cfa2fd and required emergency fix at 36a9d95. | Before any future addition of a new training stage or data-prep target |
| T3 | OPEN | Promote pickles-Claude's local `_bootstrap_build_prepared.py` (PowerShell here-string workaround for `prepare_datasets.ps1`) into a permanent helper, OR fix `prepare_datasets.ps1` to handle the here-string limitation cleanly on PowerShell 5.1+. | Next time bootstrap re-runs on a fresh Windows host |

## Architectural revisits from Cluster 2 subagents (post-hour-5 gate)

These were defaults chosen at hour-0 with explicit caveats. Revisit
after first chairside use reveals which matter.

### Subagent B (training reimpls)

| ID | Status | Description | Trigger |
|----|--------|-------------|---------|
| B1 | OPEN | Decide LR schedule: constant lr=1e-4 (paper-exact, current) vs ReduceLROnPlateau or StepLR. Paper likely intended a scheduler but upstream committed code had it commented out. | If validation loss plateaus early in any training run, or hour-5 model accuracy falls short of paper's 87% pattern claim |
| B2 | OPEN | Apex `num_keypoints` is 1 (per current design); paper used 2 with `[0,0,0]` padding for missing root on single-rooted teeth. Revisit if apex localization underperforms. | If apex landmark accuracy is weak in hour-5 evaluation |
| B3 | OPEN | DataLoader `num_workers=4` may need to drop to 0 on Windows native (not WSL). Linux/WSL works. | If training hangs or throws multiprocessing errors on the RTX 4090 box |

### Subagent C (rule layer)

| ID | Status | Description | Trigger |
|----|--------|-------------|---------|
| C1 | OPEN | AAP Stage IV criteria: currently uses `≥3 severe sites → IV` count-proxy. AAP 2018 staging is more nuanced (extent/complexity modifiers, masticatory dysfunction) — not all observable from a single radiograph. Either refine the proxy or cap at Stage III until multi-image rollup exists. | When Joseph reviews first chairside outputs and the AAP staging reads wrong |
| C2 | OPEN | Tier boundaries at exact 15.0% / 33.0% bone loss: currently inclusive → moderate. Confirm clinical reading matches. | When Joseph encounters a boundary case in chairside use |
| C3 | OPEN | Pattern algorithm uses "any angular endpoint wins" rather than majority voting across both endpoints. Conservative — slightly over-flags vertical defects. Flip to majority if false-positive rate is too high. | If Joseph notes over-flagging of vertical defects |
| C4 | OPEN | `classify_pattern` signature includes `cej_landmarks` / `bone_crest_landmarks` as guard-only inputs (used to detect "unmeasurable") but they don't factor into the angle math. Either remove from signature or genuinely consume them. | Schema cleanup; not user-facing |

### Subagent D (orchestrator + render)

| ID | Status | Description | Trigger |
|----|--------|-------------|---------|
| D1 | DEFERRED | Render mode default: currently side-by-side (original | annotated). Add `--overlay-only` flag for tight chairside screen-real-estate. | After Joseph tries it on his op monitor; trivial to add when need is real |
| D2 | OPEN | Caries integration in `analyze.py` builds lightweight ToothFinding stubs from raw detections (only bbox is populated). If the rule layer later needs `keypoints` / `bone_loss` per-tooth for caries surface assignment refinement, this stub builder needs to inherit from the post-keypoint stage. | Only if surface-assignment accuracy is poor in chairside output |

## Workflow items

| ID | Status | Description | Trigger |
|----|--------|-------------|---------|
| W1 | OPEN | Eval-set transfer to pickles: pick one of Tailscale Drive / OpenSSH server + scp / `python -m http.server` over Tailscale. Four scrubbed BWs at `examples/eval/bw0[1-4].png` need to land on pickles before hour-5 inference. | Anytime before bone-loss training finishes (~3 hours from kickoff) |
| W2 | OPEN | Update design doc at `~/repos/work-root/.claude/worktrees/priceless-poitras-203181/output/proposals/2026-05-11-dental-rad-cli-v0-design.md` with all locked decisions: A2 paper-exact arch, MIT license, hardware (RTX 4090 + M4 Max, no Mac Mini), no Huan testing, Baasils for caries, GitHub Release weights deferred per Elon-mode, etc. | After hour-5 gate decision so design doc reflects actual perception outcome, not just plan |
| W3 | OPEN | After v0.5 caries training finishes, evaluate whether to fold the now-stale Renielaz caveat in `docs/v0.5-caries-remediation.md` into a shorter "lessons learned" section, OR keep the full forensic for institutional memory. | When v0.5 has been running cleanly for ≥1 week without surfacing new corruption-pattern concerns |

## Office-data scaling ladder (v0.6 onward)

Joseph has ~25,000 patients in his office's PMS, each with potentially
multiple FMX/BW image sets over years. Conservative estimate: ~80K-
150K unlabeled radiographs sitting in the practice's image archive.

In 2026 ML this is enough data to outperform a public-corpus baseline
*without* hand-labeling all of it. Foundation models + self-supervised
pretraining + active-learning annotation collapse the labeled-data
requirement by 10-100x compared to 2019-era supervised CV.

The ladder below converts existing PMS radiographs into trainable
signal without 125K hand-labels. Each rung has a clean trigger.

| ID | Status | Description | Trigger |
|----|--------|-------------|---------|
| D1 | OPEN | **PHI scrub pipeline at scale.** Automate DICOM-header strip + burned-in pixel-region detection (corner heuristic + small OCR pass) + randomized filename + audit log. Outputs scrubbed image pool at `~/tenant-data/dental-rad-eval/scrubbed-corpus/`. Bounded engineering: ~2-3 days. Replaces the 4-image hour-0 manual recipe. | When v0.5 chairside use validates the perception layer and Joseph wants to fine-tune to his office's BW distribution |
| D2 | OPEN | **Pull all PMS radiographs into the scrub pipeline.** Use curve-genie / dental-sdk-server to enumerate patients with images, download via Curve API or direct image-archive export, route through D1. Resulting corpus: ~80K-150K scrubbed BWs/PAs. Storage: ~40 GB at ~300 KB/image, fits on pickles' disk post-Steam cleanup. | After D1 ships |
| D3 | OPEN | **Self-supervised pretraining on the scrubbed corpus.** SimCLR / MAE / DINO-style pretraining of an image encoder on the full unlabeled pool. No labels needed; the encoder learns "what does a dental radiograph look like" representations stronger than ImageNet's. Output: a `dental-encoder.pt` checkpoint reusable as the backbone for tooth-detect / keypoint / segmentation / caries heads. ~12-48 hours of training on RTX 4090. | After D2 ships |
| D4 | OPEN | **Active-learning annotation pass.** Use the D3 encoder + the v0.5 model heads to pre-annotate the office corpus. Surface the top-N most-uncertain cases for Joseph's review in Label Studio (F4). Target: ~500-2,000 reviewed cases over a few weeks. Per-case time drops to 10-20 sec because geometry is pre-placed; Joseph only judges class assignment + corrections. | After D3 produces usable pre-annotations |
| D5 | OPEN | **Fine-tune the v0.5 model heads on the D4 corpus.** Initialize from public-data weights, fine-tune on office-labeled data. Expected accuracy lift vs v0.5 baseline: significant (the office data is in-distribution; public data is partly mismatched). This is the v1.0 quality bar — at-or-above Overjet's reported numbers on Joseph's distribution. | After D4 reaches ~500 reviewed cases |
| D6 | OPEN | **Outcome feedback loop.** Once v1.0 is deployed chairside, capture doctor accept/edit/reject events per finding into a structured log. Periodic retraining incorporates this signal — model learns "Joseph rejected this caries finding" → adjust threshold. This is the closest analogue to Overjet's carrier-feedback moat and produces the same compounding-quality effect. | After v1.0 chairside use accumulates ≥1 month of accept/edit logs |
| D7 | OPEN | **Longitudinal cohort training.** PMS holds patients over many years. Training on "same tooth across multiple FMX sets" enables temporal models (bone loss progression, caries development rate). Different head architecture; same encoder backbone from D3. v1.x territory. | After D5 ships and clinical use surfaces specific multi-image questions (e.g., "is this bone loss progressing?") |

**Strategic note.** This ladder is the alternative to a multi-tenant
federated training story. A single office at ~25K patients is enough
to compete with payer-side commercial perception (Overjet/Pearl) on
provider-side accuracy, as long as the workflow exists to turn
unlabeled radiographs into trainable signal without hand-labeling all
of them. By D5/D6 the model trained on Joseph's office data should
be at-or-above Overjet's reported accuracy on his distribution. The
federated/multi-tenant pool becomes a v2.0+ nice-to-have, not a
precondition for credibility.

This changes the workspace-mission framing slightly: tenant #2 is no
longer required for the perception layer to be load-bearing. It becomes
required for the *generalization-across-offices* claim, which is a
different (and longer-horizon) product.

## Memory items (worth surfacing to workspace-Claude)

Once dental-rad-cli stabilizes (after first chairside use), these
deserve memory-system entries in the work-root memory layer:

- New project pattern: cleanroom-reimplementation from a methodology brief (vs clone-and-port) successfully dodged the no-LICENSE legal issue from the upstream paper repo. Pattern is reusable for other unlicensed open-source ML references.
- New rule: never trust a dataset's marketing description; always probe `data.yaml` + actual per-class annotation counts via REST API before committing.
- New rule: every multi-script pipeline needs a smoke test that asserts the scripts AGREE on file paths. Path-divergence bugs are silent until runtime.
