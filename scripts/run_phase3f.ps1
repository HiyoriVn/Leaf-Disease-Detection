param(
    [Parameter(Mandatory = $true)][ValidateSet('v4','v5','b4')][string]$Experiment,
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$Repo = 'D:\CassavaWork\Leaf-Disease-Detection',
    [string]$DataRoot = 'D:\CassavaWork\data\extracted',
    [string]$Python = (Join-Path $Repo '.venv-train\Scripts\python.exe'),
    [int]$BatchSize = 0,
    [switch]$Pilot,
    [switch]$RequireCuda
)
$ErrorActionPreference = 'Stop'
$Config = switch ($Experiment) {
    v4 { 'configs\cassava_b3_v4_symptom_preserving.yaml' }
    v5 { 'configs\cassava_b3_v5_full_cosine.yaml' }
    b4 { 'configs\cassava_efficientnet_b4.yaml' }
}
$FineStage = if ($Experiment -eq 'v5') { 'all' } else { 'last' }
if ($BatchSize -le 0) { $BatchSize = if ($Experiment -eq 'b4') { 8 } else { 16 } }
$Args = @('-RunName',$RunName,'-Config',$Config,'-DataRoot',$DataRoot,'-Python',$Python,
          '-BatchSize',"$BatchSize",'-FineStage',$FineStage)
if ($RequireCuda) { $Args += '-RequireCuda' }
if ($Pilot) { $Args += '-Pilot' }
& (Join-Path $PSScriptRoot 'run_train.ps1') @Args
