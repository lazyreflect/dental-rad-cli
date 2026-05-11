"""``dental-rad-cli`` command-line entrypoint.

Usage::

    dental-rad-cli analyze IMAGE [IMAGE ...] --out results/
    dental-rad-cli analyze 'data/*.jpg' --out results/ --emit-note-draft
    dental-rad-cli analyze data/bw01.jpg --dry-run --out results/

Exit codes
----------

- ``0`` — success.
- ``2`` — weights directory or a required weight file is missing.
- ``1`` — any other error (file read failure, etc.).
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence


def _expand_image_paths(patterns: Sequence[str]) -> List[Path]:
    """Resolve positional args: direct paths AND glob patterns."""
    out: List[Path] = []
    for pattern in patterns:
        # If the argument is a literal file, use it as-is. Otherwise
        # treat as a glob pattern. This keeps ``foo.jpg`` and
        # ``data/*.jpg`` both working without an explicit flag.
        if any(ch in pattern for ch in "*?["):
            matched = sorted(glob.glob(pattern))
            out.extend(Path(p) for p in matched)
        else:
            out.append(Path(pattern))
    return out


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse tree. Exposed for unit testing."""
    parser = argparse.ArgumentParser(
        prog="dental-rad-cli",
        description="Chairside dental radiograph documentation aid.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser(
        "analyze",
        help="Run the inference pipeline on one or more radiographs.",
    )
    analyze.add_argument(
        "image_paths",
        nargs="+",
        help="Image file(s) or glob pattern(s) to analyze.",
    )
    analyze.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for JSON / PNG / note files. "
             "If omitted, results are computed but no files are written.",
    )
    analyze.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/"),
        help="Directory containing the trained model weight files.",
    )
    analyze.add_argument(
        "--no-render",
        action="store_true",
        help="Skip annotated PNG generation.",
    )
    analyze.add_argument(
        "--emit-note-draft",
        action="store_true",
        help="Also write a clinical-note text draft per image.",
    )
    analyze.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip model inference; emit a synthetic AnalysisResult. "
             "Useful for wiring tests without a GPU or trained weights.",
    )
    analyze.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable INFO-level logging.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry. Returns an exit code (0 success, 2 weights missing, 1 other)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("dental_rad_cli")

    if args.command != "analyze":
        parser.error(f"unknown command: {args.command}")
        return 2

    # Import the orchestrator lazily — keeps ``--help`` fast and avoids
    # importing torch/ultralytics for trivial CLI usage like
    # ``--dry-run`` against a glob that matches nothing.
    from dental_rad_cli.analyze import analyze as run_analyze
    from dental_rad_cli.analyze import WeightsNotFoundError

    image_paths = _expand_image_paths(args.image_paths)
    if not image_paths:
        log.error("no images matched: %s", args.image_paths)
        return 1

    render = not args.no_render

    failures = 0
    for image_path in image_paths:
        log.info("analyzing %s", image_path)
        try:
            run_analyze(
                image_path=image_path,
                weights_dir=args.weights,
                out_dir=args.out,
                emit_note_draft=args.emit_note_draft,
                render=render,
                dry_run=args.dry_run,
            )
        except WeightsNotFoundError as exc:
            print(
                f"weights/ not found ({exc}) — run scripts/download_weights.sh "
                f"or scripts/train_all.sh",
                file=sys.stderr,
            )
            return 2
        except FileNotFoundError as exc:
            log.error("file not found: %s", exc)
            failures += 1
        except Exception as exc:  # noqa: BLE001 — surface unexpected errors
            log.error("analysis failed for %s: %s", image_path, exc)
            failures += 1

    return 0 if failures == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
