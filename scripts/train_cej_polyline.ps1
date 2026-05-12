# Launches CEJ polyline training as a detached background process on Windows.
#
# Usage (on pickles, from anywhere):
#   powershell -File C:\Users\13038\repos\dental-rad-cli\scripts\train_cej_polyline.ps1
#
# Or, after `cd repos/dental-rad-cli`:
#   .\scripts\train_cej_polyline.ps1
#
# Stdout → logs/train_cej_polyline.log
# Stderr → logs/train_cej_polyline.err
# PID    → logs/train_cej_polyline.pid
#
# To check progress:
#   Get-Content logs/train_cej_polyline.log -Tail 30
#
# To stop:
#   Get-Process -Id (Get-Content logs/train_cej_polyline.pid) | Stop-Process

param(
    [int]$Epochs = 200
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$LogDir = Join-Path $RepoRoot 'logs'
$LogOut = Join-Path $LogDir 'train_cej_polyline.log'
$LogErr = Join-Path $LogDir 'train_cej_polyline.err'
$PidFile = Join-Path $LogDir 'train_cej_polyline.pid'

if (-not (Test-Path $Python)) {
    throw "Python venv not found at $Python"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Truncate previous logs.
'' | Set-Content -Path $LogOut
'' | Set-Content -Path $LogErr

$argList = @(
    '-u',  # unbuffered stdout — critical for live tailing
    '-m', 'dental_rad_cli.training.segmentation',
    '--target', 'cej',
    '--data', 'data/prepared/yolo_cej_polyline/dataset.yaml',
    '--out', 'weights/segmentation_cej.pt',
    '--epochs', $Epochs.ToString()
)

$proc = Start-Process -FilePath $Python `
    -ArgumentList $argList `
    -WorkingDirectory $RepoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $LogOut `
    -RedirectStandardError $LogErr `
    -PassThru

# Persist PID for monitoring.
$proc.Id | Out-File -FilePath $PidFile -Encoding ascii

Write-Host "Started training: PID $($proc.Id)"
Write-Host "  log:  $LogOut"
Write-Host "  err:  $LogErr"
Write-Host "  pid:  $PidFile"
Write-Host "Run 'Get-Content $LogOut -Tail 30' to see progress."
