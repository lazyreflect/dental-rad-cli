#!/usr/bin/env bash
# Sweep candidate bone-landmark rules on dev split.
#
# Karpathy discipline: change → test → measure → decide. Each rule
# runs the full benchmark_eval (2.5 min) so the comparison is faithful
# to the production pipeline, not a retrospective simulation.
#
# Outputs: one JSON per rule + one stratification summary per rule.

set -euo pipefail
cd "$(dirname "$0")/.."

RULES=(
  min_y_half       # baseline (current production)
  median_y_half
  max_y_half
  median_y_at_cej_x
  max_y_at_cej_x
  wide_aware
)
# Skipping min_y_at_cej_x — that's BRneg-1, already known.

OUT_DIR="output/training-evidence/sweep-$(date +%Y-%m-%dT%H%M%S)"
mkdir -p "$OUT_DIR"

echo "Sweep output dir: $OUT_DIR"
echo "Rules: ${RULES[*]}"
echo ""

for rule in "${RULES[@]}"; do
  echo "=== rule: $rule ==="
  json_path="$OUT_DIR/benchmark-$rule.json"
  log_path="$OUT_DIR/stratify-$rule.txt"
  .venv/bin/python scripts/benchmark_eval.py --split=dev \
    --landmark-rule="$rule" \
    --out-json "$json_path" 2>&1 | tail -3
  .venv/bin/python scripts/stratify_dev_errors.py --json "$json_path" \
    > "$log_path" 2>&1
  # Extract headline line from stratify output.
  echo "--- $rule stratification (excerpt) ---"
  grep -E "TOTAL|severe|extreme|'Honest visible'" "$log_path" | head -8
  echo ""
done

echo "Sweep complete. Per-rule stratifications in $OUT_DIR/stratify-*.txt"
