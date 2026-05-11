"""Renielaz Dental Caries X-ray (Roboflow) → internal YOLO format adapter.

This adapter is the caries counterpart to :mod:`denpar_adapter` for the
tooth/keypoint pipeline. It downloads the Renielaz dataset from Roboflow
and converts its 6-class ICCMS annotation into our 3-class collapse
(``initial`` / ``moderate`` / ``deep``).

Dataset
-------
- Source: https://universe.roboflow.com/renielaz/dental-caries-x-ray
- Size: ~586 source bitewings, expanded to 1483 images at the version
  used by v0 (Roboflow's preprocessing/augmentation export multiplies
  count; default Roboflow augmentations preserve label fidelity).
- License: CC-BY 4.0
- Annotations: ICCMS 6-class polygons (RA1, RA2, RA3, RB4, RC5, RC6)

Class collapse
--------------

The ICCMS 6-tier scale is collapsed to 3 classes for v0 to keep the
deepest tier (RC6, ~pulp-near) from being starved by the corpus.
Mapping (also documented in ``docs/caries-class-mapping.md``)::

    initial   = RA1 + RA2 + RA3   (enamel through EDJ)
    moderate  = RB4 + RC5         (outer + middle dentin)
    deep      = RC6               (inner dentin / pulp-near)

The internal YOLO class indices written by :func:`build_yolo_caries_dataset`
are: ``0=initial``, ``1=moderate``, ``2=deep``.

Inference time, the trained model emits class ids 0/1/2; the inference
helper :func:`dental_rad_cli.pipeline.caries_inference.detect_caries`
maps those to the schema's ``CariesDepth`` field
(``"E1"``/``"D1"``/``"D3"`` respectively).
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Final

_LOG = logging.getLogger(__name__)

# Roboflow project + version identifiers — match the Renielaz universe page.
# The project re-exports periodically with slightly different counts and
# augmentation settings. Version 6199 (2024-01-26, 1483 images) was the
# latest known at v0 ship time. Override via env var RENIELAZ_VERSION if
# Roboflow re-uploads again (avoids a code change).
_RF_WORKSPACE: Final[str] = "renielaz"
_RF_PROJECT: Final[str] = "dental-caries-x-ray"
_RF_VERSION_DEFAULT: Final[int] = int(os.environ.get("RENIELAZ_VERSION", "6199"))
_RF_FORMAT: Final[str] = "yolov8"

# Source ICCMS class names → internal 3-class collapse.
# Lower-cased on lookup to be robust to Roboflow casing differences.
_ICCMS_TO_INTERNAL: Final[dict[str, str]] = {
    "ra1": "initial",
    "ra2": "initial",
    "ra3": "initial",
    "rb4": "moderate",
    "rc5": "moderate",
    "rc6": "deep",
}

# Internal class order (index = YOLO class id written to label files).
_INTERNAL_CLASSES: Final[tuple[str, ...]] = ("initial", "moderate", "deep")
_INTERNAL_INDEX: Final[dict[str, int]] = {
    name: i for i, name in enumerate(_INTERNAL_CLASSES)
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_renielaz(output_root: Path, api_key: str | None = None) -> Path:
    """Download the Renielaz dental-caries dataset via the Roboflow API.

    Idempotent: if the dataset already appears to be downloaded
    (``data.yaml`` present under ``output_root``), the function returns
    the existing path without re-downloading.

    Args:
        output_root: Destination directory. The Roboflow SDK creates
            a subdirectory named after the project/version; this
            function returns whichever subdirectory contains the
            extracted ``data.yaml``.
        api_key: Roboflow API key. If ``None``, falls back to the
            ``ROBOFLOW_API_KEY`` environment variable. Raises
            ``RuntimeError`` if neither is set.

    Returns:
        Path to the dataset root containing ``data.yaml`` + ``train/``
        / ``valid/`` / ``test/`` subdirectories (Roboflow YOLOv8
        export layout).
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Idempotency: scan for an existing data.yaml anywhere under
    # output_root before touching the network.
    existing = _find_existing_dataset(output_root)
    if existing is not None:
        _LOG.info("renielaz: dataset already present at %s", existing)
        return existing

    key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY not provided; set the env var or pass api_key="
        )

    # Local import — keep top-level import light + avoid hard dependency
    # on `roboflow` for unrelated unit tests.
    from roboflow import Roboflow  # type: ignore

    rf = Roboflow(api_key=key)
    project = rf.workspace(_RF_WORKSPACE).project(_RF_PROJECT)
    version = project.version(_RF_VERSION_DEFAULT)
    # Roboflow SDK writes into the current working directory by default;
    # chdir for the duration of the download so it lands under output_root.
    cwd = Path.cwd()
    try:
        os.chdir(output_root)
        dataset = version.download(_RF_FORMAT)
    finally:
        os.chdir(cwd)

    ds_path = Path(dataset.location) if hasattr(dataset, "location") else None
    if ds_path is None or not ds_path.is_dir():
        # Fallback: re-scan after download.
        ds_path = _find_existing_dataset(output_root)
        if ds_path is None:
            raise RuntimeError(
                f"roboflow download completed but no data.yaml found under {output_root}"
            )
    _LOG.info("renielaz: downloaded to %s", ds_path)
    return ds_path


def build_yolo_caries_dataset(roboflow_root: Path, output_root: Path) -> Path:
    """Convert a Roboflow YOLOv8 caries export to the internal 3-class layout.

    Reads the Roboflow export at ``roboflow_root`` (containing
    ``data.yaml`` + ``train/`` / ``valid/`` / ``test/``), and writes a
    re-mapped dataset at ``output_root`` using the internal YOLO
    directory shape used by the rest of this repo
    (``images/<split>/`` + ``labels/<split>/``).

    The class-collapse logic (RA1/RA2/RA3 → 0=initial,
    RB4/RC5 → 1=moderate, RC6 → 2=deep) is applied to every label
    file. Class ids in the Roboflow export are translated through the
    export's own ``data.yaml`` ``names:`` list, so this works regardless
    of the ordering Roboflow chose.

    Args:
        roboflow_root: Path containing the Roboflow YOLOv8 export's
            ``data.yaml`` (the value returned by :func:`download_renielaz`).
        output_root: Destination directory. Existing files are
            overwritten; idempotent re-runs are safe.

    Returns:
        Path to the generated ``data.yaml`` at ``output_root/data.yaml``.
    """
    roboflow_root = Path(roboflow_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    src_yaml = roboflow_root / "data.yaml"
    if not src_yaml.is_file():
        raise FileNotFoundError(f"roboflow data.yaml not found: {src_yaml}")

    source_class_names = _parse_yaml_names(src_yaml)
    if not source_class_names:
        raise RuntimeError(
            f"roboflow data.yaml at {src_yaml} declares no classes; cannot "
            "build collapsed dataset."
        )

    # Build per-source-id → internal-id mapping. Unknown classes raise.
    src_to_internal: dict[int, int] = {}
    for src_id, name in enumerate(source_class_names):
        key = name.strip().lower()
        if key not in _ICCMS_TO_INTERNAL:
            raise RuntimeError(
                f"unknown ICCMS class '{name}' at index {src_id} in "
                f"{src_yaml}; expected one of {sorted(_ICCMS_TO_INTERNAL)}"
            )
        internal_name = _ICCMS_TO_INTERNAL[key]
        src_to_internal[src_id] = _INTERNAL_INDEX[internal_name]

    # Roboflow YOLOv8 export uses train/ valid/ test/ subfolders, each
    # containing images/ and labels/. Normalize to our convention.
    rf_splits: dict[str, str] = {"train": "train", "valid": "val", "test": "test"}

    for sub in ("images", "labels"):
        for out_split in rf_splits.values():
            (output_root / sub / out_split).mkdir(parents=True, exist_ok=True)

    for rf_split, lc_split in rf_splits.items():
        src_split = roboflow_root / rf_split
        if not src_split.is_dir():
            _LOG.warning("renielaz: split missing in source: %s", src_split)
            continue
        src_images = src_split / "images"
        src_labels = src_split / "labels"
        out_images = output_root / "images" / lc_split
        out_labels = output_root / "labels" / lc_split

        if src_images.is_dir():
            for img in sorted(src_images.iterdir()):
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                dst = out_images / img.name
                if not dst.exists():
                    shutil.copy2(img, dst)

        if src_labels.is_dir():
            for lbl in sorted(src_labels.glob("*.txt")):
                rows: list[str] = []
                for line in lbl.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue  # malformed; YOLOv8 bbox = 5 tokens, seg = 7+
                    try:
                        src_cls = int(parts[0])
                    except ValueError:
                        continue
                    if src_cls not in src_to_internal:
                        _LOG.warning(
                            "renielaz: unknown source class id %d in %s; row skipped",
                            src_cls,
                            lbl,
                        )
                        continue
                    new_cls = src_to_internal[src_cls]
                    rows.append(" ".join([str(new_cls), *parts[1:]]))
                (out_labels / lbl.name).write_text(
                    "\n".join(rows) + ("\n" if rows else "")
                )

    return _write_caries_yaml(output_root)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_existing_dataset(root: Path) -> Path | None:
    """Locate an existing Roboflow export anywhere under ``root``.

    Returns the directory containing ``data.yaml`` (depth 0 or 1).
    """
    if (root / "data.yaml").is_file():
        return root
    for child in root.iterdir() if root.is_dir() else []:
        if child.is_dir() and (child / "data.yaml").is_file():
            return child
    return None


def _parse_yaml_names(yaml_path: Path) -> list[str]:
    """Very small YAML reader — extracts the ``names:`` list.

    Supports both the flow form (``names: ['a', 'b']``) and the block
    form (``names:\\n  - a\\n  - b``) emitted by Roboflow. Avoids a
    pyyaml dependency since the rest of this repo is stdlib-preferring.
    """
    text = yaml_path.read_text()
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        s = raw.strip()
        if s.startswith("names:"):
            after = s[len("names:"):].strip()
            # Flow form: names: ['ra1', 'ra2', ...]
            if after.startswith("[") and after.endswith("]"):
                inner = after[1:-1]
                for tok in inner.split(","):
                    tok = tok.strip().strip("'\"")
                    if tok:
                        out.append(tok)
                return out
            # Block form: names:\n  - ra1\n  - ra2 ...
            i += 1
            while i < n:
                nxt = lines[i]
                ls = nxt.strip()
                if ls.startswith("- "):
                    tok = ls[2:].strip().strip("'\"")
                    if tok:
                        out.append(tok)
                    i += 1
                    continue
                if not ls or ls.startswith("#"):
                    i += 1
                    continue
                break
            return out
        i += 1
    return out


def _write_caries_yaml(output_root: Path) -> Path:
    """Emit the internal-collapsed Ultralytics dataset YAML."""
    yaml_path = output_root / "data.yaml"
    yaml_path.write_text(
        "# Auto-generated by caries_adapter.build_yolo_caries_dataset.\n"
        "# Source: Renielaz Dental Caries X-ray (Roboflow), CC-BY 4.0.\n"
        "# Class collapse: RA1/RA2/RA3->initial, RB4/RC5->moderate, RC6->deep.\n"
        f"path: {output_root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(_INTERNAL_CLASSES)}\n"
        "names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(_INTERNAL_CLASSES)),
        encoding="utf-8",
    )
    return yaml_path
