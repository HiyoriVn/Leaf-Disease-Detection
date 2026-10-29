param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$DataRoot = (Join-Path $PSScriptRoot '..\..\data\extracted'),
    [string]$Python = (Join-Path $PSScriptRoot '..\.venv-train\Scripts\python.exe'),
    [string]$OutputRoot = (Join-Path $PSScriptRoot '..\checkpoints\experiments'),
    [string]$Config = 'configs\cassava_training.yaml',
    [int]$BatchSize = 0,
    [int]$PilotTrainPerClass = 0,
    [int]$PilotValPerClass = 0,
    [ValidateSet('last','all')][string]$FineStage = 'last',
    [switch]$RequireCuda,
    [switch]$Resume,
    [switch]$Pilot
)
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ConfigPath = if ([System.IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $Repo $Config }
if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) { throw "Training config not found: $ConfigPath" }
$DataRoot = (Resolve-Path $DataRoot).Path
if (-not (Test-Path $Python)) { throw "Training Python not found: $Python. Run setup_train.ps1 first." }
$RunDir = Join-Path $OutputRoot $RunName
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$Log = Join-Path $OutputRoot ($RunName + '.console.log')
Push-Location $Repo
try {
    & $Python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch environment check failed.' }
    if ($RequireCuda) {
        & $Python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 2)"
        if ($LASTEXITCODE -ne 0) { throw 'CUDA is not available; refusing to start the GPU run.' }
    }
    $Args = @('scripts/train_cassava.py', '--run-name', $RunName, '--data-root', $DataRoot, '--output-dir', $RunDir, '--config', $ConfigPath)
    $Args += @('--fine-stage', $FineStage)
    if ($BatchSize -gt 0) { $Args += @('--batch-size', "$BatchSize") }
    if ($PilotTrainPerClass -gt 0) { $Args += @('--pilot-train-per-class', "$PilotTrainPerClass") }
    if ($PilotValPerClass -gt 0) { $Args += @('--pilot-val-per-class', "$PilotValPerClass") }
    if ($Resume) { $Args += '--resume' }
    if ($Pilot) { $Args += '--pilot' }
    & $Python @Args 2>&1 | Tee-Object -FilePath $Log -Append
    if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE. See $Log" }
}
finally { Pop-Location }
