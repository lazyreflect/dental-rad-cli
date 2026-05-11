# scripts/prepare_datasets.ps1
# PowerShell equivalent of prepare_datasets.sh for Windows native pickles workflow.
# Materializes YOLO + COCO subsets from raw DenPAR v3 data into data/prepared/.
# Calls the Python adapter functions in dental_rad_cli.data.denpar_adapter.
# Idempotent: the adapter functions skip outputs that already exist.

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

# Activate venv if not already active
$VenvActivate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if ((Test-Path $VenvActivate) -and (-not $env:VIRTUAL_ENV)) {
    Write-Host "[venv] Activating $VenvActivate"
    & $VenvActivate
}

$DenParRoot = Join-Path $RepoRoot "data\denpar\Dataset"
if (-not (Test-Path $DenParRoot)) {
    Write-Error "[fail] DenPAR not found at $DenParRoot. Run scripts\download_denpar.ps1 first."
    exit 1
}

Write-Host "[prepare] Building YOLO + COCO subsets from DenPAR v3..."
python -c @"
from pathlib import Path
from dental_rad_cli.data.denpar_adapter import build_yolo_dataset, build_coco_keypoints

root = Path('data/denpar/Dataset')
out  = Path('data/prepared')

print('[1/4] YOLO tooth_detect ...')
build_yolo_dataset(root, out / 'yolo_tooth_detect', 'tooth_detect')

print('[2/4] YOLO tooth_seg ...')
build_yolo_dataset(root, out / 'yolo_tooth_seg', 'tooth_seg')

print('[3/4] YOLO bone_seg ...')
build_yolo_dataset(root, out / 'yolo_bone_seg', 'bone_seg')

print('[4/4] COCO keypoints (unified file across landmarks) ...')
build_coco_keypoints(root, out / 'coco_keypoints', 'cej')

print('[ok] Prepared datasets at', out.resolve())
"@

if ($LASTEXITCODE -ne 0) {
    Write-Error "[fail] Adapter exited non-zero. Inspect output above."
    exit 1
}

Write-Host "[ok] Prepared datasets ready under data\prepared\"
