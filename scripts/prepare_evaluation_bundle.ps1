param([string]$Destination = (Join-Path $PSScriptRoot '..\transfer\evaluation'))
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Destination = [System.IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$Bundle = Join-Path $Destination 'Leaf-Disease-Detection-evaluation.zip'
$Include = @('README.md','LICENSE','requirements-train-extra.txt','configs\cassava_efficientnet_b3_app.json',
 'configs\dataset_config.yaml','manifests\cassava.csv','scripts\train_cassava.py',
 'scripts\evaluate_cassava_validation.py','scripts\run_evaluate_b3.ps1','src\__init__.py',
 'src\cassava_inference.py','src\classification_metrics.py','docs\WINDOWS_GPU_VALIDATION.md')
if (Test-Path -LiteralPath $Bundle) { Remove-Item -LiteralPath $Bundle -Force }
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$Archive = [System.IO.Compression.ZipFile]::Open($Bundle, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($Item in $Include) {
        $Path = Join-Path $Repo $Item
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Required bundle file missing: $Path" }
        $EntryName = [System.IO.Path]::GetRelativePath($Repo, $Path).Replace('\', '/')
        $Entry = $Archive.CreateEntry($EntryName, [System.IO.Compression.CompressionLevel]::Optimal)
        $InputStream = [System.IO.File]::OpenRead($Path)
        $OutputStream = $Entry.Open()
        try { $InputStream.CopyTo($OutputStream) }
        finally { $OutputStream.Dispose(); $InputStream.Dispose() }
    }
}
finally { $Archive.Dispose() }
Write-Host "Evaluation-only bundle created: $Bundle"
Write-Host 'No dataset or checkpoint is included.'
