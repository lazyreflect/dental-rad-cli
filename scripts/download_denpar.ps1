# scripts/download_denpar.ps1
# PowerShell equivalent of download_denpar.sh for Windows native pickles workflow.
# Downloads DenPAR v3 (Zenodo record 16645076, ~141 MB, CC-BY 4.0) to data/denpar/.
# Idempotent: skips download if the Dataset/ subfolder already exists.

$ErrorActionPreference = "Stop"

# Resolve repo root (this script lives in scripts/)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot = Split-Path -Parent $ScriptDir
$DataDir = Join-Path $RepoRoot "data\denpar"

if (Test-Path (Join-Path $DataDir "Dataset")) {
    Write-Host "[skip] DenPAR Dataset/ already present at $DataDir\Dataset"
    exit 0
}

New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
$ZipPath = Join-Path $DataDir "DenPAR-v3.zip"

if (-not (Test-Path $ZipPath)) {
    Write-Host "[download] DenPAR v3 from Zenodo (record 16645076, ~141 MB)..."
    $Url = "https://zenodo.org/api/records/16645076/files/DenPAR%20Radiographs%20Dataset.zip/content"
    # Use curl.exe (shipped with Windows 10+); falls back to Invoke-WebRequest if missing
    if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
        curl.exe -L -o $ZipPath $Url
    } else {
        Invoke-WebRequest -Uri $Url -OutFile $ZipPath
    }
} else {
    Write-Host "[skip] $ZipPath already exists"
}

$ZipSize = (Get-Item $ZipPath).Length / 1MB
if ($ZipSize -lt 130) {
    Write-Error "[fail] DenPAR-v3.zip size $([math]::Round($ZipSize,1)) MB is below expected ~141 MB. Re-download."
    exit 1
}

Write-Host "[unzip] Extracting to $DataDir\Dataset ..."
Expand-Archive -Path $ZipPath -DestinationPath $DataDir -Force

if (-not (Test-Path (Join-Path $DataDir "Dataset\Training\Images"))) {
    Write-Error "[fail] Expected Dataset\Training\Images\ not found after unzip. Inspect $DataDir."
    exit 1
}

$TrainCount = (Get-ChildItem (Join-Path $DataDir "Dataset\Training\Images") -File).Count
$ValCount   = (Get-ChildItem (Join-Path $DataDir "Dataset\Validation\Images") -File).Count
$TestCount  = (Get-ChildItem (Join-Path $DataDir "Dataset\Testing\Images") -File).Count
Write-Host "[ok] DenPAR v3 ready: train=$TrainCount val=$ValCount test=$TestCount"
