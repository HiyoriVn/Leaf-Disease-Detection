param(
    [Parameter(Mandatory = $true)][ValidateSet('weighted','smoothing')][string]$Experiment,
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$DataRoot = 'D:\CassavaWork\data\extracted',
    [string]$Python = (Join-Path $PSScriptRoot '..\.venv-train\Scripts\python.exe'),
    [string]$OutputRoot = (Join-Path $PSScriptRoot '..\checkpoints\experiments'),
    [int]$BatchSize = 16,
    [switch]$Pilot,
    [switch]$RequireCuda
)
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Config = if ($Experiment -eq 'weighted') {
    Join-Path $Repo 'configs\cassava_efficientnet_b3_phase3e_weighted.yaml'
} else {
    Join-Path $Repo 'configs\cassava_efficientnet_b3_phase3e_smoothing.yaml'
}
$Args = @('-RunName', $RunName, '-Config', $Config, '-DataRoot', $DataRoot,
          '-Python', $Python, '-OutputRoot', $OutputRoot,
          '-BatchSize', "$BatchSize", '-FineStage', 'last')
if ($Pilot) { $Args += '-Pilot' }
if ($RequireCuda) { $Args += '-RequireCuda' }
& (Join-Path $PSScriptRoot 'run_train.ps1') @Args
