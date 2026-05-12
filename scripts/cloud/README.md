# Cloud GPU Setup for Autoresearch

The M4 Max test-drive cycle returned `cej_collapse_rate=1.0000` because 600 sec on MPS only gets ~1 epoch of Keypoint R-CNN training — not enough for the head to differentiate keypoints. Cloud GPU is the path to Karpathy-spec velocity (5-12 experiments/hour).

## TL;DR — what to do tonight

1. Sign up at [RunPod](https://runpod.io) (or use existing account). Add $20 credit.
2. Spin up an **A40 GPU pod** (~$0.40/hour, 48GB VRAM, ~2x M4 Max for fp16). Use the official PyTorch image: `runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04` or similar.
3. SSH into the pod (RunPod gives you the connection string).
4. Paste this one-liner inside the pod to bootstrap:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/lazyreflect/dental-rad-cli/main/scripts/cloud/setup_pod.sh | bash
   ```
5. From your Mac, scp the trained baseline weights up so the cloud machine has the same baseline to compare against:
   ```bash
   scp -P <RUNPOD_SSH_PORT> ~/repos/work/dental-rad-cli/weights/keypoint_cej.pt root@<RUNPOD_HOST>:/workspace/dental-rad-cli/weights/
   ```
   (Only `keypoint_cej.pt` is strictly needed for the CEJ-collapse autoresearch loop.)
6. Inside the pod, verify the baseline reproduces on CUDA:
   ```bash
   cd /workspace/dental-rad-cli
   bash scripts/eval.sh
   ```
   Should print `cej_collapse_rate: 0.3071` (within float tolerance).
7. Spawn the autoresearch agent (see "Spawning the agent" below). Detach. Sleep.

## Cost expectation

| GPU | Hourly | Overnight (8h) | Throughput vs M4 Max |
|---|---|---|---|
| **A40 48GB** | $0.40 | **$3.20** | ~2x faster |
| A100 40GB | $1.20 | $9.60 | ~3x faster |
| H100 80GB | $2.50 | $20 | ~5x faster |

A40 is the recommended sweet spot. For our Keypoint R-CNN workload it gives:
- ~2-3 sec/iter at batch=8 (CUDA + AMP)
- ~162 iters/epoch at batch=4 → ~80 iters/epoch at batch=8 → **~3-4 min/epoch**
- 300 sec budget → **~1.5 epochs per experiment** (still tight but real)
- 600 sec budget → **~3 epochs per experiment** (recommended)

Override the budget for cloud:
```bash
export AR_BUDGET_SECONDS=600    # 10 min per experiment on A40
```

At ~11-12 min/experiment including eval, the loop runs ~5/hour, ~40 overnight. To match Karpathy's spec (12/hour, ~100 overnight) bump to A100 or H100.

## Setup pod script

The `setup_pod.sh` in this directory does:

1. Updates apt, installs git + curl + tmux
2. Clones `dental-rad-cli` to `/workspace/dental-rad-cli`
3. Installs Python dependencies via uv (faster than pip)
4. Downloads DenPAR v3 dataset (~141 MB) into `data/denpar/`
5. Prints next steps

Run it on the pod (NOT on your Mac):
```bash
curl -fsSL https://raw.githubusercontent.com/lazyreflect/dental-rad-cli/main/scripts/cloud/setup_pod.sh | bash
```

## Spawning the autoresearch agent on the cloud pod

Three options, in order of recommendation:

### Option A — Claude Code on the pod (canonical)

```bash
# On the pod, install Claude Code
curl -fsSL https://claude.ai/install.sh | sh    # or whatever the install path is
export ANTHROPIC_API_KEY=<your-key>

# Start the agent
cd /workspace/dental-rad-cli
tmux new -d -s autoresearch "claude --prompt 'Read autoresearch/cej-collapse/program.md and begin the autoresearch loop. Joseph is asleep.'"

# Detach. Sleep. Check results in the morning:
tmux attach -t autoresearch
cat autoresearch/cej-collapse/results.tsv
```

### Option B — Spawn a background agent from your Mac terminal

From this current Claude conversation, you can ask me to spawn a background agent that SSHs into the cloud pod and runs the loop. The agent runs as a sub-conversation with its own context window; it can edit train.py, commit, run via SSH, measure, log to results.tsv, repeat. You'd get a notification when it finishes.

### Option C — Shell-script loop (no LLM)

If you just want to thrash a fixed sweep of pre-defined configs (no LLM-driven experimentation):

```bash
# On the pod, in tmux:
cd /workspace/dental-rad-cli
for config in baseline keypoint_5x clahe_10 hflip_05 cosine; do
    git checkout main
    git checkout -b autoresearch/cej-collapse-$config
    # ... edit train.py for this config ...
    AR_BUDGET_SECONDS=600 python autoresearch/cej-collapse/train.py > run-$config.log 2>&1
    metric=$(grep "^cej_collapse_rate:" run-$config.log | awk '{print $2}')
    echo "$config: $metric" >> sweep_results.txt
done
```

This isn't autoresearch (no LLM judgment, no adaptive next-experiment selection). But it's the simplest possible "run my pre-defined sweep" if you'd rather not deal with agent setup.

## Pulling results back to your Mac

When the run is done:

```bash
# On your Mac
scp -P <RUNPOD_SSH_PORT> root@<RUNPOD_HOST>:/workspace/dental-rad-cli/autoresearch/cej-collapse/results.tsv \
    ~/repos/work/dental-rad-cli/autoresearch/cej-collapse/results.tsv

scp -P <RUNPOD_SSH_PORT> root@<RUNPOD_HOST>:/workspace/dental-rad-cli/weights/keypoint_cej.pt \
    ~/repos/work/dental-rad-cli/weights/keypoint_cej_best_autoresearch.pt
```

Then `git diff` to see what train.py changes the agent kept on the branch.

## Pod hygiene

- Stop the pod when the loop is done (RunPod charges by the second; idle = wasting money)
- For overnight runs, consider RunPod's **Spot/Community Cloud** pricing — about 50% off if you can tolerate occasional preemption. The autoresearch loop tolerates preemption fine because each experiment commits to git; you just lose the in-progress experiment.
- Persistent volumes: if you want data to survive pod restarts, attach a 100GB Network Volume (~$0.10/GB/month). DenPAR + Baasils + weights fit easily.

## Why A40 and not A100/H100

Our specific bottleneck on M4 Max was wall-clock per epoch, not VRAM. A40's 48GB is overkill for Keypoint R-CNN at batch=8 (peak ~12GB). A100's extra 30% TFLOPS vs A40 isn't worth 3x the price for this workload. H100's 5x is overkill unless you're optimizing for "100 experiments per night" exactly.

If we later pivot to foundation-model + LoRA (Sapiens, SAM2, etc.), the math flips: those models have higher VRAM hunger and benefit from A100/H100. Reassess at that point.
