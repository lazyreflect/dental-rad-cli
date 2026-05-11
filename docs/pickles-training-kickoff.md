# Pickles training kickoff prompt

Use this AFTER `BOOTSTRAP STATUS` returns green from pickles-Claude.
Paste the block below verbatim. The orchestrator (workspace-Claude) wrote
this so pickles-Claude has a self-contained brief without needing the
upstream conversation context.

---

```
Bootstrap is green. Before training, pull the latest code (caries
pipeline + analyze.py wiring landed since bootstrap pinned at 3a6eb08).

## Step 0: Sync to latest main

```powershell
cd C:\Users\13038\repos\dental-rad-cli
git pull origin main
git log --oneline -5
```

Expected HEAD: at least 2525f97 ("feat(analyze): wire caries inference
into orchestrator + gitignore eval BWs") or later.

## Step 1: Install caries dependency (v0.5 prep — optional)

```powershell
.\.venv\Scripts\Activate.ps1
uv pip install roboflow
```

The `roboflow` package is needed for the caries adapter, deferred to
v0.5. Installing here is a no-op for v0 training but keeps the
environment ready when the caries dataset gets sorted out.

## Step 2: Roboflow API key (v0.5 prep — optional)

Required only when caries actually trains in v0.5; can be skipped for
v0 bone-loss-only training.

## Step 3: Re-run the test suite

```powershell
pytest tests\ -v
```

Expected: **90 passed** (76 bone-loss + 14 caries). All caries tests
use mocked SDK calls so they pass without real Roboflow data or
trained weights.

## Steps 4 + 5: SKIP for v0

Caries dataset is deferred — see `docs/v0.5-caries-remediation.md` for
the upstream-corruption story and v0.5 remediation paths. Do NOT run
`scripts/download_caries_data.ps1` for v0.

Sanity-check that DenPAR-prepared datasets are still in place from
bootstrap:

```powershell
Get-ChildItem data\prepared | Format-Table Name, LastWriteTime
```

Expected: 4 directories — yolo_tooth_detect, yolo_tooth_seg, yolo_bone_seg,
coco_keypoints. (yolo_caries deferred to v0.5.)

## Training sequence — v0 (bone-loss only, caries DEFERRED)

Caries pipeline is deferred to v0.5 due to upstream dataset corruption
discovered at hour-0 bootstrap (see `docs/v0.5-caries-remediation.md`
for the full story + remediation options).

The repo retains all caries code; the orchestrator gracefully skips
caries when `weights/caries.pt` is absent. v0 ships bone-loss only.

**Order — 6 stages:**
1. tooth_detect      (YOLOv9e, 3-class, ~30-60 min on RTX 4090)
2. segmentation_tooth (YOLOv8x-seg, ~30-60 min)
3. segmentation_bone  (YOLOv8x-seg, ~30-60 min)
4. keypoint_cej       (Keypoint R-CNN ResNet50-FPN, ~30-60 min)
5. keypoint_bone      (Keypoint R-CNN ResNet50-FPN, ~30-60 min)
6. keypoint_apex      (Keypoint R-CNN ResNet50-FPN, ~30-60 min)

Total wall: ~3.5-5 hours on RTX 4090. Patience for early-stop is
20-30 epochs so many will end earlier than the 200-epoch cap.

## Kickoff: serialized full run

Recommended for hour-0. Bash-via-Git is fine since Git for Windows
provides bash.exe at `C:\Program Files\Git\bin\bash.exe`.

```powershell
mkdir -Force logs
bash .\scripts\train_all.sh 2>&1 | Tee-Object -FilePath logs\train_all.log
```

DO NOT run `train_caries.sh` for v0 — caries dataset is deferred (see
`docs/v0.5-caries-remediation.md`). The orchestrator gracefully skips
caries inference when `weights/caries.pt` is absent.

If `bash` is not on PATH for some reason, run each stage manually via
the Python entrypoints — refer to `scripts/train_*.sh` for exact
commands.

## What to report during training

Approximately every 30 minutes (or at each stage boundary):

```
=== TRAINING STATUS @ <hh:mm> ===

CURRENT STAGE: <e.g. keypoint_cej, epoch 12/200>
GPU UTILIZATION: <e.g. 95% / VRAM 18.2 / 24 GB>
ELAPSED: <e.g. 2h 15m total>
COMPLETED: <list of stages with their final val metrics>
FAILURES: <any stages that errored — paste the traceback>

=== END STATUS ===
```

If any stage fails completely, stop the train_all sequence and report.
Don't auto-retry.

## After all 6 land (v0)

When `weights/` has 6 .pt files:

```powershell
Get-ChildItem weights\*.pt | Format-Table Name, Length
```

Expected: tooth_detect.pt (~50 MB), segmentation_{tooth,bone}.pt
(~50 MB each), keypoint_{cej,bone,apex}.pt (~160 MB each).
Total ~580 MB. (caries.pt is NOT expected at v0; see remediation doc.)

Then standby for the orchestrator's next instruction (eval-set
transfer + inference on scrubbed BWs at hour-5 gate).

## Eval-set transfer (deferred until end of training)

Four scrubbed bitewings live on the MacBook at:
```
/Users/josephpitluck/repos/work/dental-rad-cli/examples/eval/bw01.png
/Users/josephpitluck/repos/work/dental-rad-cli/examples/eval/bw02.png
/Users/josephpitluck/repos/work/dental-rad-cli/examples/eval/bw03.png
/Users/josephpitluck/repos/work/dental-rad-cli/examples/eval/bw04.png
```

They're gitignored (PHI-derived even if scrubbed). They need to land on
pickles at `C:\Users\13038\repos\dental-rad-cli\examples\eval\` before
inference. Three transfer options — orchestrator will pick one when
training nears completion:
- Tailscale Drive (cleanest if available)
- Enable Windows OpenSSH server + scp
- Simple `python -m http.server` on MacBook + curl from pickles
  (Tailscale-routed; no SSH needed)

## What you do NOT do during training

- Don't run inference yet — eval set isn't on pickles yet.
- Don't touch source files. If a stage fails, report — orchestrator
  will fix on the MacBook side and push.
- Don't kill long-running processes "just to check status." Let
  early-stopping handle convergence.
- Don't restart bootstrap. Everything's installed.
- Don't echo the ROBOFLOW_API_KEY back into chat.

Standing by after each training stage for the orchestrator's go-ahead
to continue.
```
