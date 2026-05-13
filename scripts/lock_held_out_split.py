#!/usr/bin/env python3
"""Lock the DenPAR Testing held-out / dev split.

Run ONCE to deterministically partition the 200-image DenPAR Testing set:
  - dev (150 images): touched freely for architectural decisions
  - held-out (50 images): touched ONCE, at the end of architectural work

After the first successful run, this script refuses to overwrite the
split files. The split is committed to git as immutable.

Seed: random.Random(42). Reproducible across Python 3.x versions.

Why a held-out lock
-------------------

Every architectural-version comparison (v0.5 through v0.7) used the full
200-image Testing set. That set has informed 7+ architecture decisions,
which means the current mean MAE on it is overfit to those decisions.
The held-out set is the honest final-eval surface, untouched until a
single end-of-development read.

CAVEAT on patient-level leakage
-------------------------------

DenPAR ships ~1000 radiographs with numeric stem IDs. Whether each stem
is a unique patient or multiple shots per patient is not documented at
the level of detail verified here. This split is image-level, NOT
patient-level. If we later confirm the patient mapping and find leakage
across dev/held-out, the split must be regenerated.

Usage::

    python scripts/lock_held_out_split.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

SEED = 42
HELD_OUT_SIZE = 50
TOTAL_EXPECTED = 200


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--denpar-root", type=Path, default=Path("data/denpar"))
    ap.add_argument("--splits-dir", type=Path, default=Path("splits"))
    ap.add_argument(
        "--force", action="store_true",
        help="Overwrite existing splits. DANGEROUS — breaks the lock.",
    )
    args = ap.parse_args()

    images_dir = args.denpar_root / "Dataset" / "Testing" / "Images"
    if not images_dir.is_dir():
        print(f"ERROR: {images_dir} not found", file=sys.stderr)
        return 1

    stems = sorted(p.stem for p in images_dir.glob("*.jpg"))
    if len(stems) != TOTAL_EXPECTED:
        print(
            f"WARN: expected {TOTAL_EXPECTED} images, found {len(stems)}",
            file=sys.stderr,
        )

    dev_file = args.splits_dir / "denpar_dev.txt"
    held_out_file = args.splits_dir / "denpar_held_out.txt"

    if (dev_file.exists() or held_out_file.exists()) and not args.force:
        print(
            "ERROR: split files already exist. The split is locked once "
            "written.",
            file=sys.stderr,
        )
        print(
            "Use --force ONLY if you have explicit reason (e.g., patient-"
            "level leakage discovered) and have logged the regeneration "
            "rationale in splits/README.md.",
            file=sys.stderr,
        )
        return 1

    rng = random.Random(SEED)
    shuffled = list(stems)
    rng.shuffle(shuffled)
    held_out = sorted(shuffled[:HELD_OUT_SIZE])
    dev = sorted(shuffled[HELD_OUT_SIZE:])

    args.splits_dir.mkdir(parents=True, exist_ok=True)
    dev_file.write_text("\n".join(dev) + "\n")
    held_out_file.write_text("\n".join(held_out) + "\n")

    print(f"Locked split (seed={SEED}):")
    print(f"  dev:       {len(dev):3d} stems  → {dev_file}")
    print(f"  held-out:  {len(held_out):3d} stems  → {held_out_file}")
    print()
    print("Discipline: held-out is touched ONCE, at end of architectural")
    print(f"work. Log every touch in {args.splits_dir}/HELD_OUT_TOUCHES.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
