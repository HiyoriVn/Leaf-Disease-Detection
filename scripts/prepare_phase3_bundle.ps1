param([string]$Destination = (Join-Path $PSScriptRoot '..\transfer\phase3e3f'))
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Destination = [System.IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$Bundle = Join-Path $Destination 'Leaf-Disease-Detection-phase3e3f-source.zip'
$Include = @(
    'README.md','LICENSE','requirements-train-extra.txt','manifests\cassava.csv',
    'configs\dataset_config.yaml','configs\cassava_efficientnet_b3.yaml',
    'configs\cassava_efficientnet_b3_phase3e_weighted.yaml','configs\cassava_efficientnet_b3_phase3e_smoothing.yaml',
    'configs\cassava_b3_v4_symptom_preserving.yaml','configs\cassava_b3_v5_full_cosine.yaml','configs\cassava_efficientnet_b4.yaml',
    'scripts\train_cassava.py','scripts\evaluate_cassava_validation.py','scripts\audit_b3_validation_results.py','scripts\compare_model_predictions.py',
    'scripts\ensemble_prediction_csvs.py','scripts\visual_audit_preprocessing.py','scripts\prepare_timm_weights.py',
    'scripts\run_train.ps1','scripts\run_evaluate_b3_variants.ps1','scripts\run_phase3e_comparison.ps1',
    'scripts\run_phase3e.ps1','scripts\run_phase3f.ps1','scripts\run_phase3f_tta.ps1',
    'scripts\prepare_phase3_bundle.ps1','scripts\setup_train.ps1','scripts\verify_pretrained.py',
    'src\__init__.py','src\cassava_inference.py','src\classification_metrics.py',
    'docs\PHASE3E_GPU_EVALUATION.md','docs\PHASE3F_GPU_PLAN.md',
    'reports\16_efficientnet_b3_integration.md','reports\17_cassava_model_comparison.md',
    'reports\18_phase3e_experiment_plan.md','reports\19_phase3e_results.md','reports\20_phase3f_diagnosis.md',
    'reports\b3_validation_archive_audit.json'
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
Write-Host "Created source-only transfer bundle: $Bundle"
Write-Host 'No dataset images, pretrained files, or trained checkpoints are included.'
