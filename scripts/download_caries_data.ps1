# scripts/download_caries_data.ps1
# PowerShell equivalent of download_caries_data.sh for the Windows pickles workflow.
# Downloads the Renielaz Dental Caries X-ray dataset from Roboflow and
# rewrites it as our internal 3-class YOLOv8 layout under data/prepared/yolo_caries/.
#
# Requires the ROBOFLOW_API_KEY environment variable.

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot = Split-Path -Parent $ScriptDir
$RawDir = Join-Path $RepoRoot "data\caries"
$PreparedDir = Join-Path $RepoRoot "data\prepared\yolo_caries"

if (-not $env:ROBOFLOW_API_KEY) {
    Write-Error "caries: ROBOFLOW_API_KEY not set. Get a free key at https://app.roboflow.com/settings/api"
    exit 1
}

New-Item -ItemType Directory -Force -Path $RawDir | Out-Null
New-Item -ItemType Directory -Force -Path $PreparedDir | Out-Null

# Best-effort venv activation.
$VenvActivate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    . $VenvActivate
}
$env:PYTHONPATH = "$RepoRoot\src;$env:PYTHONPATH"

Write-Host "caries: step 1/2 - download Renielaz from Roboflow -> $RawDir"
$Step1 = @"
from pathlib import Path
from dental_rad_cli.data.caries_adapter import download_renielaz
out = download_renielaz(Path(r'$RawDir'))
print(f'caries: roboflow export at {out}')
"@
$Step1 | python -

Write-Host "caries: step 2/2 - re-map ICCMS classes -> 3-class internal layout -> $PreparedDir"
$Step2 = @"
from pathlib import Path
from dental_rad_cli.data.caries_adapter import (
    build_yolo_caries_dataset,
    download_renielaz,
)
raw = Path(r'$RawDir')
roboflow_root = download_renielaz(raw)
yaml = build_yolo_caries_dataset(roboflow_root, Path(r'$PreparedDir'))
print(f'caries: data.yaml at {yaml}')
"@
$Step2 | python -

Write-Host "caries: ready at $PreparedDir"
