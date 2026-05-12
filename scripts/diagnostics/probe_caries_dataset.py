#!/usr/bin/env python3
"""Probe a Roboflow dental-caries project for the corruption pattern
that took down the Renielaz v0 attempt.

The Renielaz ``dental-caries-x-ray`` project at hour-0 of v0 had 6
garbage classes (description-bullet markdown serialized into class
names) sitting alongside 6 real ICCMS classes, with only 205 of 2978
annotations bound to real classes. This script catches that pattern
by hitting Roboflow's REST API directly (the source of truth for
project metadata — more reliable than the SDK's download path) and
verifying:

1. Workspace + project resolve.
2. Class list contains exactly the expected ICCMS class names.
3. Per-class annotation counts pass the minimum thresholds, especially
   for the deep tier (RC6 >= 30 is the load-bearing one — Renielaz's
   13 samples was the disqualifier).
4. No high-volume "junk" classes (the Renielaz tell: description
   bullets with >50 annotations bound to them).

Run this BEFORE committing to any Roboflow dental dataset as a v0.5+
training source. Lesson logged 2026-05-11.

Usage::

    # Default: probes Baasils ICCMS (current v0.5 source).
    python scripts/diagnostics/probe_caries_dataset.py

    # Override workspace/project to vet another candidate::

    WORKSPACE=other-workspace PROJECT=other-project \\
        python scripts/diagnostics/probe_caries_dataset.py

Prerequisites:
    - ROBOFLOW_API_KEY in env, or in repo-root ``.env`` if python-dotenv
      is installed.
"""

from __future__ import annotations

import os
import sys
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
import json

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass


API_KEY = os.environ.get("ROBOFLOW_API_KEY")
if not API_KEY:
    print("ERROR: ROBOFLOW_API_KEY not set. Add to .env or export.")
    sys.exit(1)

WORKSPACE = os.environ.get("WORKSPACE", "baasils-workspace")
PROJECT = os.environ.get("PROJECT", "iccms-dental-caries-etomb")

# Expected class names. Override-able for non-ICCMS datasets.
EXPECTED_NAMES = set(os.environ.get(
    "EXPECTED_CLASSES", "RA1,RA2,RA3,RB4,RC5,RC6"
).split(","))

# Minimum sample-count thresholds per class. The deep tier (RC6) is
# load-bearing because Renielaz's 13 samples was the disqualifier.
MIN_PER_CLASS = int(os.environ.get("MIN_PER_CLASS", "30"))
JUNK_THRESHOLD = int(os.environ.get("JUNK_THRESHOLD", "50"))


def fetch_json(url: str) -> dict:
    try:
        with urlopen(url, timeout=30) as resp:
            return json.loads(resp.read())
    except (HTTPError, URLError) as e:
        print(f"ERROR: REST API call failed for {url}: {e}")
        sys.exit(1)


print(f"=== Caries dataset probe ===")
print(f"Workspace: {WORKSPACE}")
print(f"Project:   {PROJECT}")
print()

data = fetch_json(
    f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}?api_key={API_KEY}"
)
project = data.get("project") or {}
if not project:
    print("ERROR: project metadata empty — workspace/project URL wrong?")
    sys.exit(1)

print(f"Name:    {project.get('name', '?')}")
print(f"Type:    {project.get('type', '?')}")
print(f"Public:  {project.get('public')}")
print(f"License: {project.get('license', '?')}")
print(f"Images:  {project.get('images', '?')}")
print(f"Versions: {project.get('versions', '?')}")
print()

classes = project.get("classes") or {}
if not classes:
    print("VERDICT: PROBE-FAILS  no `classes` block in project metadata")
    sys.exit(2)

print("Class distribution (project metadata — source of truth):")
total = 0
for name, count in sorted(classes.items(), key=lambda kv: -kv[1]):
    expected_marker = " EXPECTED" if name in EXPECTED_NAMES else " UNEXPECTED"
    print(f"  {name:30s} {count:6d}{expected_marker}")
    total += count
print(f"  {'TOTAL':30s} {total:6d}")
print()

# --- Verdict ---
issues: list[str] = []

unexpected = [n for n in classes if n not in EXPECTED_NAMES]
if unexpected:
    issues.append(f"Unexpected class names: {unexpected}")

missing = [n for n in EXPECTED_NAMES if n not in classes]
if missing:
    issues.append(f"Missing expected classes: {missing}")

for name in EXPECTED_NAMES:
    if name in classes and classes[name] < MIN_PER_CLASS:
        issues.append(
            f"{name} has only {classes[name]} annotations (need >= {MIN_PER_CLASS})"
        )

# Renielaz-style junk class detection: high-count unexpected class.
for name, count in classes.items():
    if name not in EXPECTED_NAMES and count >= JUNK_THRESHOLD:
        issues.append(f"Junk-class pattern: {name!r} has {count} annotations")

print("=== VERDICT ===")
if issues:
    print("PROBE-FAILS")
    for issue in issues:
        print(f"  - {issue}")
    sys.exit(2)
else:
    print("PROBE-PASSES")
    print("  All expected classes present, all above sample-count floor,")
    print("  no junk-class pattern detected.")
    sys.exit(0)
