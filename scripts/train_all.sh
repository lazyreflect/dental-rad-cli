#!/usr/bin/env bash
# Sequentially train all 6 models on a single RTX 4090.
# Each stage logs to logs/training-<stage>.log with both stdout + stderr.
# Set CONTINUE_ON_FAIL=1 to keep going past a failed stage; default aborts.

. "$(dirname "$0")/_common.sh"
activate_venv

: "${CONTINUE_ON_FAIL:=0}"

STAGES=(
  "tooth_detect"
  "segmentation_tooth"
  "segmentation_bone"
  "keypoint_cej"
  "keypoint_bone"
  "keypoint_apex"
)

run_stage() {
  local stage="$1"
  local script="$(dirname "$0")/train_${stage}.sh"
  local log="${LOGS_DIR}/training-${stage}.log"
  echo "==> [${stage}] starting; log: ${log}"
  if bash "${script}" >"${log}" 2>&1; then
    echo "==> [${stage}] OK"
  else
    local rc=$?
    echo "!!  [${stage}] FAILED (exit ${rc}); see ${log}" >&2
    if [ "${CONTINUE_ON_FAIL}" != "1" ]; then
      exit "${rc}"
    fi
  fi
}

for stage in "${STAGES[@]}"; do
  run_stage "${stage}"
done

echo "all stages complete; weights in ${WEIGHTS_DIR}"
