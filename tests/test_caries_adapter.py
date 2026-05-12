"""Tests for the Renielaz → internal 3-class collapse logic.

These tests construct synthetic Roboflow YOLOv8 exports on disk and
verify that :func:`build_yolo_caries_dataset` re-maps the ICCMS class
ids correctly. No network access; no real images required (label
files are the only thing the collapse logic inspects, and we drop
empty 1x1 fixture .jpg stubs alongside them).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dental_rad_cli.data.caries_adapter import (
    _ICCMS_TO_INTERNAL,
    _INTERNAL_CLASSES,
    _INTERNAL_INDEX,
    _parse_yaml_names,
    build_yolo_caries_dataset,
)


def _write_roboflow_export(
    root: Path,
    class_names: list[str],
    split_labels: dict[str, dict[str, list[str]]],
) -> None:
    """Build a minimal synthetic Roboflow YOLOv8 export under ``root``.

    Args:
        root: Directory to populate.
        class_names: Ordered list of class names for ``data.yaml``.
        split_labels: ``{rf_split: {stem: [label_row, ...]}}`` mapping.
    """
    root.mkdir(parents=True, exist_ok=True)

    # data.yaml — block-form names list (matches Roboflow output).
    yaml_lines = [
        "path: .",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for name in class_names:
        yaml_lines.append(f"  - {name}")
    (root / "data.yaml").write_text("\n".join(yaml_lines) + "\n")

    for rf_split, files in split_labels.items():
        img_dir = root / rf_split / "images"
        lbl_dir = root / rf_split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for stem, rows in files.items():
            # 1-byte stub image — adapter only copies bytes, never decodes.
            (img_dir / f"{stem}.jpg").write_bytes(b"\xff")
            (lbl_dir / f"{stem}.txt").write_text("\n".join(rows) + "\n")


# ---------------------------------------------------------------------------
# YAML parser
# ---------------------------------------------------------------------------


def test_parse_yaml_names_block_form(tmp_path: Path) -> None:
    p = tmp_path / "data.yaml"
    p.write_text(
        "path: .\n"
        "train: train/images\n"
        "nc: 6\n"
        "names:\n"
        "  - RA1\n"
        "  - RA2\n"
        "  - RA3\n"
        "  - RB4\n"
        "  - RC5\n"
        "  - RC6\n"
    )
    assert _parse_yaml_names(p) == ["RA1", "RA2", "RA3", "RB4", "RC5", "RC6"]


def test_parse_yaml_names_flow_form(tmp_path: Path) -> None:
    p = tmp_path / "data.yaml"
    p.write_text("names: ['RA1', 'RA2', 'RC6']\n")
    assert _parse_yaml_names(p) == ["RA1", "RA2", "RC6"]


# ---------------------------------------------------------------------------
# Class-collapse end-to-end
# ---------------------------------------------------------------------------


def test_build_yolo_caries_dataset_collapses_all_six_classes(tmp_path: Path) -> None:
    rf_root = tmp_path / "caries"
    # Match the Roboflow project's likely class order; the adapter must
    # work regardless of ordering because it indexes through data.yaml.
    class_names = ["RA1", "RA2", "RA3", "RB4", "RC5", "RC6"]
    # One label row per class — verify the collapse for every source class.
    rows_train = [
        # YOLOv8 bbox: class cx cy w h
        "0 0.10 0.10 0.10 0.10",  # RA1 -> 0 initial
        "1 0.20 0.20 0.10 0.10",  # RA2 -> 0 initial
        "2 0.30 0.30 0.10 0.10",  # RA3 -> 0 initial
        "3 0.40 0.40 0.10 0.10",  # RB4 -> 1 moderate
        "4 0.50 0.50 0.10 0.10",  # RC5 -> 1 moderate
        "5 0.60 0.60 0.10 0.10",  # RC6 -> 2 deep
    ]
    _write_roboflow_export(
        rf_root,
        class_names,
        {
            "train": {"img1": rows_train},
            "valid": {"img2": ["3 0.5 0.5 0.2 0.2"]},  # RB4 -> 1 moderate
            "test": {"img3": ["5 0.5 0.5 0.2 0.2"]},  # RC6 -> 2 deep
        },
    )

    out_root = tmp_path / "prepared"
    yaml_out = build_yolo_caries_dataset(rf_root, out_root)
    assert yaml_out.is_file()
    assert yaml_out == out_root / "data.yaml"

    # data.yaml declares exactly 3 internal classes.
    text = yaml_out.read_text()
    assert "nc: 3" in text
    for name in _INTERNAL_CLASSES:
        assert name in text

    # Train labels: 3 rows -> 0, 2 rows -> 1, 1 row -> 2.
    train_label = (out_root / "labels" / "train" / "img1.txt").read_text().splitlines()
    first_ids = [int(line.split()[0]) for line in train_label]
    assert first_ids == [0, 0, 0, 1, 1, 2]

    # 'valid' split must be remapped to 'val' on the output side.
    val_label = (out_root / "labels" / "val" / "img2.txt").read_text().splitlines()
    assert [int(line.split()[0]) for line in val_label] == [1]

    test_label = (out_root / "labels" / "test" / "img3.txt").read_text().splitlines()
    assert [int(line.split()[0]) for line in test_label] == [2]

    # Images were copied through.
    assert (out_root / "images" / "train" / "img1.jpg").is_file()
    assert (out_root / "images" / "val" / "img2.jpg").is_file()
    assert (out_root / "images" / "test" / "img3.jpg").is_file()


def test_build_yolo_caries_dataset_handles_reshuffled_source_order(tmp_path: Path) -> None:
    """If Roboflow ships classes in a different order, the collapse must still hold."""
    rf_root = tmp_path / "caries"
    # Deliberately scramble.
    class_names = ["RC6", "RA1", "RB4", "RA3", "RA2", "RC5"]
    _write_roboflow_export(
        rf_root,
        class_names,
        {
            "train": {
                "img1": [
                    "0 0.5 0.5 0.1 0.1",  # RC6 at index 0 -> 2 deep
                    "1 0.5 0.5 0.1 0.1",  # RA1 at index 1 -> 0 initial
                    "2 0.5 0.5 0.1 0.1",  # RB4 at index 2 -> 1 moderate
                ],
            },
            "valid": {},
            "test": {},
        },
    )
    out_root = tmp_path / "prepared"
    build_yolo_caries_dataset(rf_root, out_root)

    rows = (out_root / "labels" / "train" / "img1.txt").read_text().splitlines()
    assert [int(line.split()[0]) for line in rows] == [2, 0, 1]


def test_build_yolo_caries_dataset_rejects_unknown_class(tmp_path: Path) -> None:
    rf_root = tmp_path / "caries"
    _write_roboflow_export(
        rf_root,
        ["RA1", "RA2", "FOO"],
        {"train": {"img1": ["0 0.5 0.5 0.1 0.1"]}, "valid": {}, "test": {}},
    )
    out_root = tmp_path / "prepared"
    with pytest.raises(RuntimeError, match="unknown ICCMS class"):
        build_yolo_caries_dataset(rf_root, out_root)


def test_iccms_mapping_covers_all_six_classes() -> None:
    """Guard: every ICCMS class must collapse into exactly one internal class."""
    assert set(_ICCMS_TO_INTERNAL) == {"ra1", "ra2", "ra3", "rb4", "rc5", "rc6"}
    # Every value must be one of the internal class names.
    assert set(_ICCMS_TO_INTERNAL.values()) == set(_INTERNAL_CLASSES)
    assert _INTERNAL_INDEX == {"initial": 0, "moderate": 1, "deep": 2}
