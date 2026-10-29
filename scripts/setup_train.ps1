param([string]$PythonVersion = '3.13')
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Venv = Join-Path $Repo '.venv-train'
if (-not (Test-Path (Join-Path $Venv 'Scripts\python.exe'))) {
    py -$PythonVersion -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create venv; install the requested Python version first.' }
}
$Py = Join-Path $Venv 'Scripts\python.exe'
& $Py -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed.' }
& $Py -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'PyTorch CUDA installation failed.' }
& $Py -m pip install -r (Join-Path $Repo 'requirements-train-extra.txt')
if ($LASTEXITCODE -ne 0) { throw 'Training dependencies installation failed.' }
Write-Host 'Environment ready. Run scripts\run_train.ps1 after copying the required files and dataset.'
