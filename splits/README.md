# DenPAR splits — held-out lock

## What's here

| File | Size | Purpose |
|---|---|---|
| `denpar_dev.txt` | 150 stems | Decision set. Touched freely for architectural choices, hyperparameter sweeps, diagnostic inspection. |
| `denpar_held_out.txt` | 50 stems | Final-eval set. Touched **once**, at end of architectural work. Every touch logged in `HELD_OUT_TOUCHES.md`. |
| `HELD_OUT_TOUCHES.md` | append-only | Log of every held-out read with rationale. |
| `lock_held_out_split.py` | script | Reproducer (in `scripts/`). Refuses to overwrite once locked. |

## Why

Prior to 2026-05-12, every architectural-version comparison (v0.5 through
v0.7) used the full 200-image DenPAR Testing set. That set has informed
7+ architecture decisions (PCA add/revert, erosion=3/5/10, boundary ring,
mask-intersection landmarks). The mean MAE on that set is therefore
overfit to those decisions. The held-out set is the surface that
produces the honest final number.

Discipline:
- All measurement during architectural iteration runs on `--split=dev`.
- `--split=held-out` requires an explicit confirm flag AND a log entry
  in `HELD_OUT_TOUCHES.md` BEFORE the run.
- Touching held-out more than ~2-3 times across the project's lifetime
  defeats its purpose. Treat it as one-shot.

## Reproduction

```bash
.venv/bin/python scripts/lock_held_out_split.py
```

Seed = 42, `random.Random()`, sorted-stems input. Reproducible across
Python 3.x versions.

## Known caveat: image-level vs patient-level

DenPAR ships ~1000 radiographs with numeric stem IDs. **Audited
2026-05-12 (BR8): lock confirmed patient-clean.** Evidence:

1. **DenPAR Sci Data 2025 paper** ([Nature s41597-025-05906-9](https://www.nature.com/articles/s41597-025-05906-9))
   reports "440 male and 560 female patients" — 440 + 560 = 1000,
   matching the image count exactly. Strongly implies 1 radiograph
   per patient.

2. **Local dhash audit** (`scripts/audit_dev_held_out_leakage.py`):
   ZERO image perceptual-hash near-duplicates among the 200 Testing
   images (Hamming ≤ 8). Metadata-signature collisions exist (31 in
   Testing) but they reflect common PA view types across DIFFERENT
   patients (e.g., "Lower right with FDIs 45-48" is shared by 21
   different patients), not same-patient duplicates.

Treat the held-out lock as patient-clean. If a future DenPAR release
or author correspondence reveals multi-image patients, regenerate via
`scripts/lock_held_out_split.py --force` with a documented rationale.

## Pre-lock observations that crossed into held-out

These stems were visually inspected for GT placement issues PRIOR to
the split being locked (2026-05-12 worst-errors diagnosis):

| Stem | Now in | Note |
|---|---|---|
| 106 | held-out | anterior mandibular incisor; incisal-edge GT |
| 1251 | held-out | anterior mandibular incisor; incisal-edge GT |
| 473 | held-out | inspected, not characterized |
| 372 | dev | inspected, not characterized |

Their diagnostic visuals at `output/diagnostics/worst-errors/` persist.
The information from those 4 inspections IS pre-leaked into our
hypotheses about GT noise. Any GT-noise characterization work going
forward should re-sample from the dev set, NOT rely on these 4.
