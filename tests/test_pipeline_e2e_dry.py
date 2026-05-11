"""End-to-end dry-run wiring test.

Exercises the full ``analyze()`` + CLI surface using ``--dry-run`` so
the test can pass without any trained weights present and without a
GPU. Verifies:

- JSON file is written with the expected top-level shape.
- Annotated PNG is written and non-empty (when render is on).
- Note draft is written when ``--emit-note-draft`` is set.
- ``--no-render`` skips the PNG.
- Missing ``weights/`` raises ``WeightsNotFoundError`` outside dry-run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from PIL import Image

from dental_rad_cli import __main__ as cli
from dental_rad_cli.analyze import WeightsNotFoundError, analyze


@pytest.fixture
def fake_image(tmp_path: Path) -> Path:
    """Write a small synthetic radiograph-shaped image to disk."""
    arr = np.random.randint(0, 255, size=(120, 160, 3), dtype=np.uint8)
    img_path = tmp_path / "bw01.jpg"
    Image.fromarray(arr).save(img_path)
    return img_path


def test_dry_run_e2e_writes_json_png_note(fake_image: Path, tmp_path: Path):
    out_dir = tmp_path / "results"
    result = analyze(
        image_path=fake_image,
        weights_dir=tmp_path / "no-such-weights",  # never read in dry-run
        out_dir=out_dir,
        emit_note_draft=True,
        render=True,
        dry_run=True,
    )

    # JSON
    json_path = out_dir / "bw01.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["schema_version"] == "0.1"
    assert payload["image"]["path"] == "bw01.jpg"
    assert payload["metadata"]["dry_run"] is True
    assert len(payload["teeth"]) >= 1
    assert "note_draft" in payload  # emit_note_draft attaches it

    # PNG
    png_path = out_dir / "bw01.annotated.png"
    assert png_path.exists()
    assert png_path.stat().st_size > 0

    # Note draft text
    note_path = out_dir / "bw01.note.txt"
    assert note_path.exists()
    note_text = note_path.read_text()
    assert "Bitewings demonstrate" in note_text

    # Return value matches written JSON shape
    assert result.metadata.dry_run is True


def test_dry_run_no_render_skips_png(fake_image: Path, tmp_path: Path):
    out_dir = tmp_path / "results"
    analyze(
        image_path=fake_image,
        weights_dir=tmp_path / "missing",
        out_dir=out_dir,
        emit_note_draft=False,
        render=False,
        dry_run=True,
    )
    assert (out_dir / "bw01.json").exists()
    assert not (out_dir / "bw01.annotated.png").exists()
    assert not (out_dir / "bw01.note.txt").exists()


def test_missing_weights_raises_outside_dry_run(fake_image: Path, tmp_path: Path):
    with pytest.raises(WeightsNotFoundError):
        analyze(
            image_path=fake_image,
            weights_dir=tmp_path / "no-such-weights",
            out_dir=None,
            dry_run=False,
        )


def test_cli_dry_run_exit_zero(fake_image: Path, tmp_path: Path):
    out_dir = tmp_path / "results"
    rc = cli.main([
        "analyze",
        str(fake_image),
        "--out", str(out_dir),
        "--dry-run",
        "--emit-note-draft",
    ])
    assert rc == 0
    assert (out_dir / "bw01.json").exists()
    assert (out_dir / "bw01.note.txt").exists()


def test_cli_missing_weights_exit_two(fake_image: Path, tmp_path: Path, capsys):
    rc = cli.main([
        "analyze",
        str(fake_image),
        "--weights", str(tmp_path / "no-such-weights"),
        "--out", str(tmp_path / "out"),
    ])
    captured = capsys.readouterr()
    assert rc == 2
    assert "weights/ not found" in captured.err
