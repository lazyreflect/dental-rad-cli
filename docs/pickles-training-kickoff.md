# Pickles training kickoff prompt

Use this AFTER `BOOTSTRAP STATUS` returns green from pickles-Claude.
Paste the block below verbatim. The orchestrator (workspace-Claude) wrote
this so pickles-Claude has a self-contained brief without needing the
upstream conversation context.

---

```
Bootstrap is green. Time to train the 6 models.

## Quick state check before kickoff

Confirm in one PowerShell session that all of these still hold:

```powershell
cd C:\Users\13038\repos\dental-rad-cli
git pull
.\.venv\Scripts\Activate.ps1
python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"
Test-Path .\data\denpar\Dataset\Training\Images
Test-Path .\data\prepared\yolo_tooth_detect
Test-Path .\data\prepared\coco_keypoints
```

If any fail, STOP and report. Common cases:
- `git pull` shows new commits → that's expected, the orchestrator
  may have pushed updates while bootstrap was running.
- `data\prepared\...` missing → run `.\scripts\prepare_datasets.ps1`.

## Training sequence

The repo has 6 training entrypoints. Order matters because keypoint
training needs tooth-detection weights for the COCO-keypoints adapter
ordering (it does NOT, actually — they're all independent. Run in any
order, but bone-segmentation and tooth-segmentation can race for GPU
memory so do them serially).

**Order:**
1. tooth_detect (YOLOv9e, 3-class, ~30-60 min on RTX 4090)
2. segmentation_tooth (YOLOv8x-seg, ~30-60 min)
3. segmentation_bone (YOLOv8x-seg, ~30-60 min)
4. keypoint_cej (Keypoint R-CNN ResNet50-FPN, ~30-60 min)
5. keypoint_bone (Keypoint R-CNN ResNet50-FPN, ~30-60 min)
6. keypoint_apex (Keypoint R-CNN ResNet50-FPN, ~30-60 min)

Total wall: ~3-6 hours on RTX 4090. Patience for early-stop is set to
20-30 epochs so many will end earlier than the 200-epoch cap.

## Two ways to kick off

**Option A — serialized (recommended for hour-0):**

```powershell
# Convert the bash train_all.sh to PowerShell-runnable sequence:
mkdir -Force logs
bash .\scripts\train_all.sh 2>&1 | Tee-Object -FilePath logs\train_all.log
```

If bash is on PATH (Git for Windows provides it at C:\Program Files\
Git\bin\bash.exe), this works. If not, run each stage manually:

```powershell
.\.venv\Scripts\Activate.ps1
python -m dental_rad_cli.training.tooth_detect --data data\prepared\yolo_tooth_detect\data.yaml --weights weights\tooth_detect.pt 2>&1 | Tee-Object -FilePath logs\training-tooth_detect.log
python -m dental_rad_cli.training.segmentation --target tooth --data data\prepared\yolo_tooth_seg\data.yaml --weights weights\segmentation_tooth.pt 2>&1 | Tee-Object -FilePath logs\training-segmentation_tooth.log
python -m dental_rad_cli.training.segmentation --target bone --data data\prepared\yolo_bone_seg\data.yaml --weights weights\segmentation_bone.pt 2>&1 | Tee-Object -FilePath logs\training-segmentation_bone.log
python -m dental_rad_cli.training.keypoints --landmark cej --dataset data\prepared\coco_keypoints --weights weights\keypoint_cej.pt 2>&1 | Tee-Object -FilePath logs\training-keypoint_cej.log
python -m dental_rad_cli.training.keypoints --landmark bone --dataset data\prepared\coco_keypoints --weights weights\keypoint_bone.pt 2>&1 | Tee-Object -FilePath logs\training-keypoint_bone.log
python -m dental_rad_cli.training.keypoints --landmark apex --dataset data\prepared\coco_keypoints --weights weights\keypoint_apex.pt 2>&1 | Tee-Object -FilePath logs\training-keypoint_apex.log
```

NOTE: the python -m flags above assume the training modules have `if
__name__ == "__main__"` argparse blocks. If they don't (per the
subagent's implementation, `train()` is a function), wrap each call in
a one-line Python invocation that imports and calls `train(...)`. The
bash scripts in `scripts/` already do this — easier to run via bash.

**Option B — start the first one and watch:**

```powershell
.\.venv\Scripts\Activate.ps1
bash .\scripts\train_tooth_detect.sh 2>&1 | Tee-Object -FilePath logs\training-tooth_detect.log
```

After it finishes (or early-stops), report back the metrics and we'll
decide whether to continue with the next stage.

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

## After all 6 land

When `weights/` has all 6 .pt files:

```powershell
Get-ChildItem weights\*.pt | Format-Table Name, Length
```

Expected: tooth_detect.pt (~50 MB), segmentation_{tooth,bone}.pt
(~50 MB each), keypoint_{cej,bone,apex}.pt (~160 MB each). Total
~640 MB.

Then standby for the orchestrator's next instruction (inference on
scrubbed BWs).

## What you do NOT do during training

- Don't run inference yet — Joseph's scrubbed BWs land at hour-5.
- Don't touch source files. If a stage fails, report — orchestrator
  will fix on the MacBook side and push.
- Don't kill long-running processes "just to check status." Let
  early-stopping handle convergence.
- Don't restart bootstrap. Everything's installed.

Standing by after each training stage for the orchestrator's go-ahead
to continue.
```
