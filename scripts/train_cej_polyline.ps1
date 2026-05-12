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

# Spawn the process via WMI Win32_Process.Create. This is the canonical
# Windows pattern for a truly detached, session-independent child
# process — it survives SSH disconnects, doesn't hold a console, and
# doesn't depend on the parent PowerShell staying alive.
#
# Start-Process with -RedirectStandardOutput silently fails when
# invoked through an SSH-cmd-PowerShell layer (no interactive console
# to redirect from). WMI Create avoids this entirely by treating the
# whole command as a shell line with native redirection.

$CommandLine = "cmd /c `"`"$Python`" -u -m dental_rad_cli.training.segmentation " +
    "--target cej " +
    "--data data/prepared/yolo_cej_polyline/dataset.yaml " +
    "--out weights/segmentation_cej.pt " +
    "--epochs $Epochs " +
    "> `"$LogOut`" 2> `"$LogErr`"`""

$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $CommandLine
    CurrentDirectory = $RepoRoot
}

if ($result.ReturnValue -ne 0) {
    throw "Win32_Process.Create failed with return value $($result.ReturnValue)"
}

# Persist PID for monitoring.
$result.ProcessId | Out-File -FilePath $PidFile -Encoding ascii

Write-Host "Started training: PID $($result.ProcessId) (detached cmd wrapper)"
Write-Host "  log:  $LogOut"
Write-Host "  err:  $LogErr"
Write-Host "  pid:  $PidFile"
Write-Host "Run 'Get-Content $LogOut -Tail 30' to see progress."
