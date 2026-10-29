param(
    [string[]]$Models = @('baseline','v2','v3'),
    [string]$DataRoot = 'D:\CassavaWork\data\extracted',
    [string]$Repo = 'D:\CassavaWork\Leaf-Disease-Detection',
    [string]$Python = (Join-Path $Repo '.venv-train\Scripts\python.exe'),
    [string]$OutputRoot = (Join-Path $Repo 'reports\phase3e_validation'),
    [string]$CheckpointBaseline = (Join-Path $Repo 'checkpoints\experiments\cassava-b3-full\best.pth'),
    [string]$CheckpointV2 = (Join-Path $Repo 'checkpoints\experiments\cassava-b3-v2-weighted\best.pth'),
    [string]$CheckpointV3 = (Join-Path $Repo 'checkpoints\experiments\cassava-b3-v3-smoothing\best.pth'),
    [int]$BatchSize = 16,
    [ValidateSet('none','hflip','dihedral')][string]$TTA = 'none',
    [switch]$RequireCuda
)
$ErrorActionPreference = 'Stop'
$Manifest = Join-Path $Repo 'manifests\cassava.csv'
$Expected = @{
    baseline = '7924296913c5f3a6c9d822212cdb0a8326b6d5c4cd2778078d6c47bebd0d406b'
    v2 = '7398eb98fe0917d330baea90c5c34c6160e2e781aab31dc7f9c73c9c653e2901'
    v3 = '0e76a24e1d979ee29f508b83fe6059f89f7a13a167effcaddc3eacfea0cf669c'
}
$Checkpoints = @{
    baseline = $CheckpointBaseline
    v2 = $CheckpointV2
    v3 = $CheckpointV3
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python environment missing: $Python" }
if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) { throw "Manifest missing: $Manifest" }
foreach ($Model in $Models) {
    if (-not $Expected.ContainsKey($Model)) { throw "Unknown model id: $Model" }
    $Checkpoint = (Resolve-Path -LiteralPath $Checkpoints[$Model]).Path
    $Actual = (Get-FileHash -LiteralPath $Checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Actual -ne $Expected[$Model]) { throw "$Model checkpoint SHA mismatch: $Actual" }
    $Out = Join-Path $OutputRoot $Model
    if (Test-Path -LiteralPath $Out) { throw "Output exists; choose a fresh OutputRoot: $Out" }
    $Args = @('scripts\evaluate_cassava_validation.py','--model',$Model,'--checkpoint',$Checkpoint,
              '--output',$Out,'--data-root',$DataRoot,'--manifest',$Manifest,'--batch-size',"$BatchSize",
              '--tta',$TTA)
    if ($RequireCuda) { $Args += '--require-cuda' }
    Push-Location $Repo
    try {
        & $Python @Args
        if ($LASTEXITCODE -ne 0) { throw "$Model evaluation failed with exit code $LASTEXITCODE" }
    } finally { Pop-Location }
}
Write-Host "Evaluation outputs are under $OutputRoot"
