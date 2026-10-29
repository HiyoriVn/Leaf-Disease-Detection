param(
    [Parameter(Mandatory = $true)][string]$Destination,
    [string]$DatasetSource = (Join-Path $PSScriptRoot '..\..\data\extracted')
)
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Destination = [System.IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$Bundle = Join-Path $Destination 'Leaf-Disease-Detection-training.zip'
$Include = @('README.md','LICENSE','requirements-train-extra.txt','configs','manifests','scripts','src','reports\09_experiment_design.md','reports\11_cassava_baseline_evaluation.md','reports\12_cassava_finetuning.md','reports\13_cassava_comparison.md')
$Paths = $Include | ForEach-Object { Join-Path $Repo $_ } | Where-Object { Test-Path $_ }
Compress-Archive -Path $Paths -DestinationPath $Bundle -Force
Write-Host "Created source bundle: $Bundle"
Write-Host 'Copy the ImageNet weights separately into checkpoints\imagenet_mobilenetv3_small_100.safetensors.'
Write-Host "Dataset must be copied to $DatasetSource or pass its extracted directory with -DataRoot."
Write-Host 'For a local disk copy, use: robocopy <source> <destination> /E /MT:8 /R:2 /W:2'
