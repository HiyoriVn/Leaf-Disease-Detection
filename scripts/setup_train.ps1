param(
    [string]$PythonVersion = '3.13',
    [string]$Config = 'configs\cassava_training.yaml',
    [switch]$VerifyWeights
)
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ConfigPath = if ([System.IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $Repo $Config }
if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) { throw "Training config not found: $ConfigPath" }
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
if ($VerifyWeights) {
    & $Py (Join-Path $Repo 'scripts\verify_pretrained.py') --config $ConfigPath
    if ($LASTEXITCODE -ne 0) { throw 'Pretrained weight verification failed.' }
}
Write-Host "Environment ready for config: $ConfigPath"
Write-Host 'Run scripts\run_train.ps1 after copying the required files and dataset.'
