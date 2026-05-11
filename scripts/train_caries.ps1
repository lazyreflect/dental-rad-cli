# scripts/train_caries.ps1
# Train the YOLOv8s 3-class caries detector against
# data\prepared\yolo_caries\data.yaml.
# Override $env:DATA_YAML / $env:WEIGHTS_OUT / $env:EPOCHS to customize.

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot = Split-Path -Parent $ScriptDir
$WeightsDir = Join-Path $RepoRoot "weights"

if (-not $env:DATA_YAML) {
    $env:DATA_YAML = Join-Path $RepoRoot "data\prepared\yolo_caries\data.yaml"
}
if (-not $env:WEIGHTS_OUT) {
    $env:WEIGHTS_OUT = Join-Path $WeightsDir "caries.pt"
}
if (-not $env:EPOCHS) {
    $env:EPOCHS = "200"
}

New-Item -ItemType Directory -Force -Path $WeightsDir | Out-Null

$VenvActivate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    . $VenvActivate
}
$env:PYTHONPATH = "$RepoRoot\src;$env:PYTHONPATH"

python -m dental_rad_cli.training.caries `
    --data $env:DATA_YAML `
    --out  $env:WEIGHTS_OUT `
    --epochs $env:EPOCHS
