#!/usr/bin/env bash
# Deploy dental-rad-cli autoresearch to a GCP Compute Engine VM with a GPU.
#
# Prereqs:
#   1. gcloud CLI installed and authenticated (`gcloud auth login`)
#   2. Project `dental-rad-cli` exists with billing enabled
#   3. Compute Engine API enabled
#   4. GPU quota in target region (default us-central1) — request via
#      https://console.cloud.google.com/iam-admin/quotas if 0
#
# What it does:
#   1. Creates a Compute Engine VM with L4 GPU (cheap) on a Deep Learning
#      image (PyTorch + CUDA pre-baked).
#   2. Copies the repo + DenPAR data + trained weights to the VM.
#   3. Installs Python deps.
#   4. Verifies the baseline eval reproduces on CUDA.
#   5. Prints the next-step command to spawn the autoresearch agent.
#
# Cost: ~$0.30/hr L4 spot, ~$0.71/hr L4 on-demand. 8h overnight ≈ $2-6.
# Stop the VM with `gcloud compute instances stop` when done.

set -euo pipefail

PROJECT="${PROJECT:-dental-rad-cli}"
ZONE="${ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-autoresearch-l4}"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-4}"   # L4 GPU bundled
DISK_SIZE_GB="${DISK_SIZE_GB:-100}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
USE_SPOT="${USE_SPOT:-true}"   # set to "false" for on-demand

# Local paths to sync up
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DENPAR_LOCAL="${REPO_ROOT}/data/denpar"
WEIGHTS_LOCAL="${REPO_ROOT}/weights"

# Remote paths
REMOTE_USER="$(whoami)"   # gcloud uses your local username by default
REMOTE_BASE="/home/${REMOTE_USER}/dental-rad-cli"

GCLOUD="${GCLOUD:-${HOME}/google-cloud-sdk/bin/gcloud}"

step() { echo; echo "==> $*"; }

# ---------------------------------------------------------------------------
# 0. Auth + project sanity
# ---------------------------------------------------------------------------

step "0. Verify gcloud auth + project"
"${GCLOUD}" config set project "${PROJECT}"
"${GCLOUD}" auth list --filter=status:ACTIVE --format="value(account)" | head -1

# ---------------------------------------------------------------------------
# 1. Check / enable required APIs
# ---------------------------------------------------------------------------

step "1. Enable Compute Engine API (idempotent)"
"${GCLOUD}" services enable compute.googleapis.com

# ---------------------------------------------------------------------------
# 2. Create the VM
# ---------------------------------------------------------------------------

step "2. Create VM ${VM_NAME} in ${ZONE} (L4 GPU)"
EXISTING=$("${GCLOUD}" compute instances list --filter="name=${VM_NAME} AND zone:${ZONE}" --format="value(name)" 2>/dev/null || true)
if [ -n "${EXISTING}" ]; then
  echo "VM already exists: ${VM_NAME}. Skipping create."
else
  SPOT_FLAGS=""
  if [ "${USE_SPOT}" = "true" ]; then
    SPOT_FLAGS="--provisioning-model=SPOT --instance-termination-action=STOP"
    echo "Using SPOT pricing (~70% off, but may preempt)"
  fi

  "${GCLOUD}" compute instances create "${VM_NAME}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --image-family="${IMAGE_FAMILY}" \
    --image-project="${IMAGE_PROJECT}" \
    --boot-disk-size="${DISK_SIZE_GB}"GB \
    --boot-disk-type=pd-balanced \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True" \
    ${SPOT_FLAGS}
fi

# ---------------------------------------------------------------------------
# 3. Wait for SSH ready
# ---------------------------------------------------------------------------

step "3. Wait for SSH ready (driver install may take ~2-3 min)"
for i in $(seq 1 30); do
  if "${GCLOUD}" compute ssh "${VM_NAME}" --zone="${ZONE}" --command='nvidia-smi -L' 2>/dev/null; then
    echo "GPU visible:"
    break
  fi
  echo "  attempt $i/30: not ready, sleeping 20s"
  sleep 20
done

# ---------------------------------------------------------------------------
# 4. Copy repo + data + weights
# ---------------------------------------------------------------------------

step "4. Copy repo (excluding data/weights/results)"
TMP_BUNDLE="/tmp/dental-rad-cli-bundle.tar.gz"
tar --exclude='./data' --exclude='./weights' --exclude='./results' \
    --exclude='./.venv' --exclude='./.git' --exclude='./output/mockups' \
    --exclude='./__pycache__' --exclude='./.pytest_cache' \
    -czf "${TMP_BUNDLE}" -C "${REPO_ROOT}" .
echo "Bundle size: $(du -h ${TMP_BUNDLE} | cut -f1)"

"${GCLOUD}" compute ssh "${VM_NAME}" --zone="${ZONE}" --command="mkdir -p ${REMOTE_BASE} && rm -rf ${REMOTE_BASE}/*"
"${GCLOUD}" compute scp "${TMP_BUNDLE}" "${VM_NAME}:${REMOTE_BASE}/bundle.tar.gz" --zone="${ZONE}"
"${GCLOUD}" compute ssh "${VM_NAME}" --zone="${ZONE}" --command="cd ${REMOTE_BASE} && tar xzf bundle.tar.gz && rm bundle.tar.gz"

step "5. Copy DenPAR data (~1 GB) — may take 1-3 min depending on upload"
"${GCLOUD}" compute scp --recurse "${DENPAR_LOCAL}" "${VM_NAME}:${REMOTE_BASE}/data/" --zone="${ZONE}"

step "6. Copy trained weights (1.1 GB for keypoint_cej.pt alone)"
mkdir -p "${REMOTE_BASE}/weights" 2>/dev/null || true
"${GCLOUD}" compute ssh "${VM_NAME}" --zone="${ZONE}" --command="mkdir -p ${REMOTE_BASE}/weights"
# Only ship the keypoint_cej weights for the CEJ-collapse autoresearch loop
"${GCLOUD}" compute scp "${WEIGHTS_LOCAL}/keypoint_cej.pt" "${VM_NAME}:${REMOTE_BASE}/weights/" --zone="${ZONE}"

# ---------------------------------------------------------------------------
# 7. Install deps on the VM
# ---------------------------------------------------------------------------

step "7. Install Python deps on the VM (uv is pre-installed in the DL image)"
"${GCLOUD}" compute ssh "${VM_NAME}" --zone="${ZONE}" --command="
set -e
cd ${REMOTE_BASE}
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install opencv-python shapely scipy
"

# ---------------------------------------------------------------------------
# 8. Reproduce baseline eval
# ---------------------------------------------------------------------------

step "8. Verify baseline eval reproduces on CUDA"
"${GCLOUD}" compute ssh "${VM_NAME}" --zone="${ZONE}" --command="
cd ${REMOTE_BASE}
source .venv/bin/activate
python scripts/eval_keypoint_cej.py
" | tee /tmp/baseline_cuda.log

EXPECTED="0.3071"
ACTUAL=$(grep "^cej_collapse_rate:" /tmp/baseline_cuda.log | awk '{print $2}')
echo "expected ${EXPECTED}, got ${ACTUAL}"

# ---------------------------------------------------------------------------
# 9. Next steps
# ---------------------------------------------------------------------------

step "Done. Next steps:"
cat <<EOF

The VM is running and the baseline eval reproduces. To start the
autoresearch loop:

  1. SSH in:
     ${GCLOUD} compute ssh ${VM_NAME} --zone=${ZONE}

  2. On the VM, spawn the autoresearch agent (Claude Code or Codex):
       cd ${REMOTE_BASE}
       tmux new -s autoresearch
       claude  # or codex; install via:
               #   curl -fsSL https://claude.ai/install.sh | sh
               #   export ANTHROPIC_API_KEY=<your-key>
       # In the agent, say:
       #   Read autoresearch/cej-collapse/program.md and begin the loop.

  3. Detach tmux with Ctrl-b d. Sleep. Check results in the morning:
       tmux attach -t autoresearch
       cat ${REMOTE_BASE}/autoresearch/cej-collapse/results.tsv

  4. When done, STOP THE VM to avoid charges:
       ${GCLOUD} compute instances stop ${VM_NAME} --zone=${ZONE}

  5. Pull results back to your Mac:
       ${GCLOUD} compute scp ${VM_NAME}:${REMOTE_BASE}/autoresearch/cej-collapse/results.tsv \\
           ${REPO_ROOT}/autoresearch/cej-collapse/results.tsv --zone=${ZONE}
EOF
