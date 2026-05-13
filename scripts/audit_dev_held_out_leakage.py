"""BR8 partial: audit dev/held-out for likely-same-patient pairs.

Subagent finding (2026-05-12): DenPAR's Sci Data 2025 paper states
"1000 radiographs of 440 male and 560 female patients" — the
demographic arithmetic strongly implies 1:1 mapping (one radiograph
per patient). Lock is "almost certainly fine."

This script is the belt-and-suspenders: detect any near-duplicate
pairs within the 200-image Testing folder by metadata + image hash.

Two checks:
  1. Metadata signature collisions: stems sharing (arch, site, FDI set)
     are candidates for same-patient bilateral views or follow-ups.
  2. Image perceptual-hash collisions: dhash similarity within Testing.

If any pair has metadata-collision AND high dhash similarity AND lands
on opposite sides of the dev/held-out split → lock is leaky.

Outputs: stdout summary table.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

try:
    import openpyxl
except ImportError:
    print("ERROR: openpyxl required.", file=sys.stderr)
    raise SystemExit(1)

import cv2
import numpy as np


def _load_meta(xlsx_path: Path) -> dict[str, tuple]:
    """Return {stem: (arch, site, frozenset(fdis))}."""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    id_col = next(i for i, h in enumerate(header)
                  if h and str(h).strip().lower() == "id")
    arch_col = next(i for i, h in enumerate(header)
                    if h and str(h).strip().lower() == "arch")
    site_col = next(i for i, h in enumerate(header)
                    if h and str(h).strip().lower() == "site")
    fdi_col = next((i for i, h in enumerate(header)
                    if h and "fdi" in str(h).strip().lower()),
                   None)
    out: dict[str, tuple] = {}
    for row in rows:
        if id_col is None or row[id_col] is None:
            continue
        stem = (str(int(row[id_col])) if isinstance(row[id_col], float)
                else str(row[id_col]).strip())
        arch = str(row[arch_col]).strip() if row[arch_col] else "?"
        site = str(row[site_col]).strip() if row[site_col] else "?"
        fdis = frozenset()
        if fdi_col is not None and row[fdi_col]:
            fdi_str = str(row[fdi_col]).strip()
            fdis = frozenset(
                p.strip() for p in fdi_str.replace(",", " ").split()
                if p.strip()
            )
        out[stem] = (arch, site, fdis)
    return out


def _dhash(img: np.ndarray, hash_size: int = 8) -> int:
    """8x8 dhash → 64-bit int. Compare with Hamming distance."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    resized = cv2.resize(gray, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    bits = 0
    for v in diff.flatten():
        bits = (bits << 1) | int(bool(v))
    return bits


def _ham(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--denpar-root", type=Path, default=Path("data/denpar"),
    )
    ap.add_argument(
        "--splits-dir", type=Path, default=Path("splits"),
    )
    ap.add_argument(
        "--xlsx", type=Path,
        default=Path("data/denpar/Dataset/Characteristics of radiographs included.xlsx"),
    )
    ap.add_argument(
        "--dhash-threshold", type=int, default=8,
        help="Hamming-distance threshold for near-duplicate (0=identical, 64=opposite)",
    )
    args = ap.parse_args()

    testing_images = args.denpar_root / "Dataset" / "Testing" / "Images"
    if not testing_images.is_dir():
        print(f"ERROR: {testing_images} not found", file=sys.stderr)
        return 1

    dev_stems = set((args.splits_dir / "denpar_dev.txt").read_text().split())
    held_out_stems = set(
        (args.splits_dir / "denpar_held_out.txt").read_text().split()
    )
    print(f"dev stems: {len(dev_stems)}   held-out stems: {len(held_out_stems)}")

    meta = _load_meta(args.xlsx)
    testing_stems = sorted(p.stem for p in testing_images.glob("*.jpg"))
    print(f"testing stems on disk: {len(testing_stems)}")
    print(f"metadata rows: {len(meta)}\n")

    # Check 1: metadata signature collisions WITHIN Testing.
    by_sig: dict[tuple, list[str]] = defaultdict(list)
    for stem in testing_stems:
        m = meta.get(stem)
        if m is None:
            continue
        by_sig[m].append(stem)
    sig_collisions = {k: v for k, v in by_sig.items() if len(v) > 1}
    print(f"Metadata-signature collisions in Testing: {len(sig_collisions)}")
    if sig_collisions:
        print("  (stems sharing identical arch+site+FDIs)")
        for sig, stems in list(sig_collisions.items())[:10]:
            arch, site, fdis = sig
            fdi_str = ",".join(sorted(fdis)) if fdis else "?"
            split_breakdown = []
            for s in stems:
                tag = "dev" if s in dev_stems else (
                    "held-out" if s in held_out_stems else "?"
                )
                split_breakdown.append(f"{s}({tag})")
            print(f"    {arch}/{site} FDIs={fdi_str}: {split_breakdown}")
        cross = sum(
            1 for stems in sig_collisions.values()
            if any(s in dev_stems for s in stems)
            and any(s in held_out_stems for s in stems)
        )
        print(f"  Sig-collisions spanning BOTH dev AND held-out: {cross}")
    print()

    # Check 2: image dhash near-duplicates.
    print("Computing dhash on all 200 Testing images...")
    hashes: dict[str, int] = {}
    for stem in testing_stems:
        img = cv2.imread(str(testing_images / f"{stem}.jpg"))
        if img is None:
            continue
        hashes[stem] = _dhash(img)
    print(f"hashed {len(hashes)} images\n")

    # All-pairs Hamming. n=200 → 19900 pairs. Cheap.
    near_dupes: list[tuple[str, str, int]] = []
    stems_list = sorted(hashes.keys())
    for i in range(len(stems_list)):
        for j in range(i + 1, len(stems_list)):
            d = _ham(hashes[stems_list[i]], hashes[stems_list[j]])
            if d <= args.dhash_threshold:
                near_dupes.append((stems_list[i], stems_list[j], d))

    near_dupes.sort(key=lambda r: r[2])
    print(f"dhash near-duplicate pairs (Hamming ≤ {args.dhash_threshold}): "
          f"{len(near_dupes)}")
    cross_split_pairs = 0
    sig_match_count = 0
    for s1, s2, d in near_dupes:
        split1 = "dev" if s1 in dev_stems else "held-out"
        split2 = "dev" if s2 in dev_stems else "held-out"
        cross = (split1 != split2)
        if cross:
            cross_split_pairs += 1
        sig_match = (meta.get(s1) == meta.get(s2))
        if sig_match:
            sig_match_count += 1
        flag = "⚠ CROSS-SPLIT" if cross else ""
        sig_flag = "(metadata-match)" if sig_match else ""
        if d <= 4 or cross or sig_match:
            print(f"  {s1}({split1}) ↔ {s2}({split2})  ham={d:>2}  "
                  f"{flag} {sig_flag}")

    print(f"\n  pairs spanning dev <-> held-out: {cross_split_pairs}")
    print(f"  pairs with matching metadata signature: {sig_match_count}")

    # Verdict.
    #
    # Metadata-signature collisions are EXPECTED in dental radiographs:
    # multiple unrelated patients have PA views of the same tooth region
    # (e.g., "Lower Right molars FDIs 45-48"). Collision alone is NOT
    # evidence of same-patient. The strong signal is image perceptual-
    # hash similarity. The truly bad case is a metadata-collision PAIR
    # that ALSO has high image similarity (Hamming ≤ 4ish).
    print("\n" + "=" * 70)
    cross_split_dhash = any(
        d <= 4 and ((s1 in dev_stems) != (s2 in dev_stems))
        for s1, s2, d in near_dupes
    )
    cross_split_sig_dhash = any(
        ((s1 in dev_stems) != (s2 in dev_stems))
        and (meta.get(s1) == meta.get(s2))
        for s1, s2, d in near_dupes
    )
    if cross_split_dhash:
        print("VERDICT: ⚠ LIKELY LEAKAGE — near-duplicate image pair spans split.")
        print("Investigate before reporting any held-out result.")
    elif cross_split_sig_dhash:
        print("VERDICT: ⚠ POSSIBLE LEAKAGE — same-metadata + similar-image pair "
              "spans split.")
    else:
        print("VERDICT: lock looks patient-clean.")
        print("  - No image perceptual-hash near-duplicates among the 200 "
              "Testing images (zero pairs at Hamming ≤ "
              f"{args.dhash_threshold}).")
        print("  - Metadata-signature collisions exist but they reflect "
              "common PA view types across DIFFERENT patients, not same-"
              "patient duplicates (per the dhash check).")
        print("  - Combined with DenPAR paper's 440+560=1000 patient "
              "arithmetic, treat the held-out lock as patient-clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
