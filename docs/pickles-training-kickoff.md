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

## Step 1: Install caries dependency

```powershell
.\.venv\Scripts\Activate.ps1
uv pip install roboflow
```

This adds the `roboflow` package the caries adapter needs (~5-10 MB +
dependencies).

## Step 2: Roboflow API key

Joseph drops the **private** Roboflow API key (the one starting with
`oZGE...`, no `rf_` prefix) into the venv env. From the PowerShell
session in the repo root:

```powershell
# In PowerShell — substitute the actual key, do not commit it.
$env:ROBOFLOW_API_KEY = "oZGE...rest-of-private-key"

# Sanity check:
if (-not $env:ROBOFLOW_API_KEY) { Write-Error "ROBOFLOW_API_KEY not set" } else { Write-Host "ROBOFLOW_API_KEY is set ($($env:ROBOFLOW_API_KEY.Length) chars)" }
```

Joseph will paste this command into your session with the real key. Do
NOT echo the full key into chat after that.

## Step 3: Re-run the full test suite

```powershell
pytest tests\ -v
```

Expected: **90 passed** (76 prior + 14 new caries tests). If anything
fails on Windows but was green on the MacBook, paste the failing test
names + tracebacks.

## Step 4: Download + prepare caries dataset

```powershell
.\scripts\download_caries_data.ps1
```

Downloads the Renielaz caries dataset from Roboflow (~10-50 MB
depending on export format), runs the 3-class collapse adapter, writes
`data/prepared/yolo_caries/` with the standard YOLO layout.

If the script doesn't exist (older commit), run:
```powershell
python -c "from pathlib import Path; from dental_rad_cli.data.caries_adapter import download_renielaz, build_yolo_caries_dataset; r = download_renielaz(Path('data/caries')); build_yolo_caries_dataset(r, Path('data/prepared/yolo_caries'))"
```

## Step 5: Sanity-check ALL prepared datasets

```powershell
Get-ChildItem data\prepared | Format-Table Name, LastWriteTime
```

Expected: 5 directories — yolo_tooth_detect, yolo_tooth_seg, yolo_bone_seg,
coco_keypoints, yolo_caries.

## Training sequence

The repo now has 7 training entrypoints (6 bone-loss models + 1 caries).
Run them serially — RTX 4090 has plenty of VRAM but only one of these at
a time avoids GPU contention.

**Order:**
1. tooth_detect      (YOLOv9e, 3-class, ~30-60 min on RTX 4090)
2. segmentation_tooth (YOLOv8x-seg, ~30-60 min)
3. segmentation_bone  (YOLOv8x-seg, ~30-60 min)
4. keypoint_cej       (Keypoint R-CNN ResNet50-FPN, ~30-60 min)
5. keypoint_bone      (Keypoint R-CNN ResNet50-FPN, ~30-60 min)
6. keypoint_apex      (Keypoint R-CNN ResNet50-FPN, ~30-60 min)
7. caries             (YOLOv8s, ~20-40 min — smaller dataset)

Total wall: ~3.5-7 hours on RTX 4090. Patience for early-stop is
20-30 epochs so many will end earlier than the 200-epoch cap.

## Kickoff: serialized full run

Recommended for hour-0. Bash-via-Git is fine since Git for Windows
provides bash.exe at `C:\Program Files\Git\bin\bash.exe`.

```powershell
mkdir -Force logs
bash .\scripts\train_all.sh 2>&1 | Tee-Object -FilePath logs\train_all.log
bash .\scripts\train_caries.sh 2>&1 | Tee-Object -FilePath logs\training-caries.log
```

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

## After all 7 land

When `weights/` has all 7 .pt files:

```powershell
Get-ChildItem weights\*.pt | Format-Table Name, Length
```

Expected: tooth_detect.pt (~50 MB), segmentation_{tooth,bone}.pt
(~50 MB each), keypoint_{cej,bone,apex}.pt (~160 MB each), caries.pt
(~30 MB). Total ~640-700 MB.

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
