"""Tests for the argparse surface of ``dental-rad-cli``."""

from __future__ import annotations

from pathlib import Path

import pytest

from dental_rad_cli.__main__ import build_parser


def test_analyze_minimal():
    parser = build_parser()
    args = parser.parse_args(["analyze", "data/bw01.jpg"])
    assert args.command == "analyze"
    assert args.image_paths == ["data/bw01.jpg"]
    assert args.out is None
    assert args.no_render is False
    assert args.emit_note_draft is False
    assert args.dry_run is False
    assert args.verbose is False


def test_analyze_all_flags():
    parser = build_parser()
    args = parser.parse_args([
        "analyze",
        "data/bw01.jpg",
        "data/bw02.jpg",
        "--out", "results/",
        "--no-render",
        "--emit-note-draft",
        "--dry-run",
        "--verbose",
    ])
    assert args.image_paths == ["data/bw01.jpg", "data/bw02.jpg"]
    assert args.out == Path("results/")
    assert args.no_render is True
    assert args.emit_note_draft is True
    assert args.dry_run is True
    assert args.verbose is True


def test_analyze_short_verbose():
    parser = build_parser()
    args = parser.parse_args(["analyze", "data/bw.jpg", "-v"])
    assert args.verbose is True


def test_analyze_custom_weights():
    parser = build_parser()
    args = parser.parse_args([
        "analyze", "data/bw.jpg", "--weights", "/tmp/weights",
    ])
    assert args.weights == Path("/tmp/weights")


def test_no_command_errors():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_unknown_command_errors():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["sniff", "foo"])
