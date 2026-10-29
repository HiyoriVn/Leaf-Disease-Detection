param(
    [string]$DataRoot = 'D:\CassavaWork\data\extracted',
    [string]$Checkpoint = 'D:\CassavaWork\Leaf-Disease-Detection\checkpoints\experiments\cassava-b3-full\best.pth',
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [int]$BatchSize = 16,
    [switch]$RequireCuda,
    [string]$Python = (Join-Path $PSScriptRoot '..\.venv-train\Scripts\python.exe')
)
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$DataRoot = (Resolve-Path -LiteralPath $DataRoot).Path
$Checkpoint = (Resolve-Path -LiteralPath $Checkpoint).Path
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python missing: $Python" }
if ($BatchSize -lt 1) { throw 'BatchSize must be positive.' }
if ([System.IO.Path]::IsPathRooted($OutputDir)) { $ResolvedOutput = $OutputDir }
else { $ResolvedOutput = Join-Path $Repo $OutputDir }
if (Test-Path -LiteralPath $ResolvedOutput) { throw "Output directory exists; choose a new path: $ResolvedOutput" }
$Manifest = Join-Path $Repo 'manifests\cassava.csv'
if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) { throw "Manifest missing: $Manifest" }
$Args = @((Join-Path $Repo 'scripts\evaluate_cassava_validation.py'), '--checkpoint', $Checkpoint,
          '--output', $ResolvedOutput, '--data-root', $DataRoot, '--manifest', $Manifest,
          '--batch-size', "$BatchSize")
if ($RequireCuda) { $Args += '--require-cuda' }
Push-Location $Repo
try {
    & $Python @Args
    if ($LASTEXITCODE -ne 0) { throw "Validation failed with exit code $LASTEXITCODE. See $ResolvedOutput\evaluation.log" }
}
finally { Pop-Location }
