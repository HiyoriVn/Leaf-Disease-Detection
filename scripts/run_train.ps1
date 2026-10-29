param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$DataRoot = (Join-Path $PSScriptRoot '..\..\data\extracted'),
    [string]$Python = (Join-Path $PSScriptRoot '..\.venv-train\Scripts\python.exe'),
    [string]$OutputRoot = (Join-Path $PSScriptRoot '..\checkpoints\experiments'),
    [switch]$Resume,
    [switch]$Pilot
)
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$DataRoot = (Resolve-Path $DataRoot).Path
if (-not (Test-Path $Python)) { throw "Training Python not found: $Python. Run setup_train.ps1 first." }
$RunDir = Join-Path $OutputRoot $RunName
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$Log = Join-Path $RunDir 'console.log'
Push-Location $Repo
try {
    & $Python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch environment check failed.' }
    $Args = @('scripts/train_cassava.py', '--run-name', $RunName, '--data-root', $DataRoot, '--output-dir', $RunDir)
    if ($Resume) { $Args += '--resume' }
    if ($Pilot) { $Args += '--pilot' }
    & $Python @Args 2>&1 | Tee-Object -FilePath $Log -Append
    if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE. See $Log" }
}
finally { Pop-Location }
