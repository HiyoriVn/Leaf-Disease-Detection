param([string]$Destination = (Join-Path $PSScriptRoot '..\transfer\phase3e'))
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Destination = [System.IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$Bundle = Join-Path $Destination 'Leaf-Disease-Detection-phase3e.zip'
$Include = @(
    'README.md','LICENSE','requirements-train-extra.txt',
    'configs\cassava_efficientnet_b3_phase3e_weighted.yaml',
    'configs\cassava_efficientnet_b3_phase3e_smoothing.yaml',
    'configs\cassava_efficientnet_b3.yaml','configs\dataset_config.yaml',
    'manifests\cassava.csv','scripts\train_cassava.py','scripts\run_train.ps1',
    'scripts\run_phase3e.ps1','scripts\setup_train.ps1','scripts\verify_pretrained.py',
    'scripts\prepare_phase3e_bundle.ps1','src\__init__.py','src\cassava_inference.py',
    'src\classification_metrics.py','docs\PHASE3E_GPU_RUN.md',
    'reports\16_efficientnet_b3_integration.md','reports\17_cassava_model_comparison.md',
    'reports\18_phase3e_experiment_plan.md'
)
if (Test-Path -LiteralPath $Bundle) { Remove-Item -LiteralPath $Bundle -Force }
Add-Type -AssemblyName System.IO.Compression
$Archive = [System.IO.Compression.ZipFile]::Open($Bundle, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($Item in $Include) {
        $Path = Join-Path $Repo $Item
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Required bundle file missing: $Path" }
        $EntryName = [System.IO.Path]::GetRelativePath($Repo, $Path).Replace('\','/')
        $Entry = $Archive.CreateEntry($EntryName, [System.IO.Compression.CompressionLevel]::Optimal)
        $InputStream = [System.IO.File]::OpenRead($Path); $OutputStream = $Entry.Open()
        try { $InputStream.CopyTo($OutputStream) }
        finally { $OutputStream.Dispose(); $InputStream.Dispose() }
    }
}
finally { $Archive.Dispose() }
Write-Host "Created Phase 3E source/config bundle: $Bundle"
Write-Host 'The bundle excludes dataset images, virtual environments, all checkpoints and training outputs.'
