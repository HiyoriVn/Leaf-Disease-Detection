param(
    [Parameter(Mandatory = $true)][string]$Destination,
    [string]$DatasetSource = (Join-Path $PSScriptRoot '..\..\data\extracted')
)
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Destination = [System.IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$Bundle = Join-Path $Destination 'Leaf-Disease-Detection-training.zip'
$Include = @('README.md','LICENSE','requirements-train-extra.txt','configs','manifests','scripts','src','docs\WINDOWS_GPU_TRAINING.md','reports\09_experiment_design.md','reports\11_cassava_baseline_evaluation.md','reports\12_cassava_finetuning.md','reports\13_cassava_comparison.md','reports\15_efficientnet_b3_preparation.md')
if (Test-Path -LiteralPath $Bundle) { Remove-Item -LiteralPath $Bundle -Force }
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$Archive = [System.IO.Compression.ZipFile]::Open($Bundle, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($Item in $Include) {
        $Path = Join-Path $Repo $Item
        if (-not (Test-Path -LiteralPath $Path)) { throw "Bundle source missing: $Path" }
        $Files = if (Test-Path -LiteralPath $Path -PathType Container) {
            Get-ChildItem -LiteralPath $Path -Recurse -File
        } else { @(Get-Item -LiteralPath $Path) }
        foreach ($File in $Files) {
            if ($File.FullName -match '[\\/]__pycache__[\\/]' -or $File.Extension -eq '.pyc') { continue }
            $EntryName = [System.IO.Path]::GetRelativePath($Repo, $File.FullName).Replace('\', '/')
            $Entry = $Archive.CreateEntry($EntryName, [System.IO.Compression.CompressionLevel]::Optimal)
            $InputStream = [System.IO.File]::OpenRead($File.FullName)
            $OutputStream = $Entry.Open()
            try { $InputStream.CopyTo($OutputStream) }
            finally { $OutputStream.Dispose(); $InputStream.Dispose() }
        }
    }
}
finally { $Archive.Dispose() }
Write-Host "Created source bundle: $Bundle"
Write-Host 'Copy the ImageNet weights separately into checkpoints\imagenet_mobilenetv3_small_100.safetensors or checkpoints\imagenet_efficientnet_b3_ra2_in1k.safetensors (match your config).'
Write-Host "Dataset must be copied to $DatasetSource or pass its extracted directory with -DataRoot."
Write-Host 'For a local disk copy, use: robocopy <source> <destination> /E /MT:8 /R:2 /W:2'
