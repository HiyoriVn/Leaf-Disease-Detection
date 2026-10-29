param(
    [string[]]$Models = @('baseline','v3'),
    [ValidateSet('hflip','dihedral')][string]$TTA = 'hflip',
    [string]$Repo = 'D:\CassavaWork\Leaf-Disease-Detection',
    [string]$DataRoot = 'D:\CassavaWork\data\extracted',
    [string]$OutputRoot = (Join-Path $Repo "reports\phase3f_$TTA"),
    [switch]$RequireCuda
)
$ErrorActionPreference = 'Stop'
$Python = Join-Path $Repo '.venv-train\Scripts\python.exe'
$Checkpoints = @{
    baseline = Join-Path $Repo 'checkpoints\experiments\cassava-b3-full\best.pth'
    v2 = Join-Path $Repo 'checkpoints\experiments\cassava-b3-v2-weighted\best.pth'
    v3 = Join-Path $Repo 'checkpoints\experiments\cassava-b3-v3-smoothing\best.pth'
}
$Hashes = @{
    baseline = '7924296913c5f3a6c9d822212cdb0a8326b6d5c4cd2778078d6c47bebd0d406b'
    v2 = '7398eb98fe0917d330baea90c5c34c6160e2e781aab31dc7f9c73c9c653e2901'
    v3 = '0e76a24e1d979ee29f508b83fe6059f89f7a13a167effcaddc3eacfea0cf669c'
}
foreach ($Model in $Models) {
    if (-not $Checkpoints.ContainsKey($Model)) { throw "Unsupported model: $Model" }
    $Checkpoint = (Resolve-Path -LiteralPath $Checkpoints[$Model]).Path
    $Hash = (Get-FileHash -LiteralPath $Checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Hash -ne $Hashes[$Model]) { throw "$Model checkpoint SHA mismatch: $Hash" }
    $Out = Join-Path $OutputRoot $Model
    $Args = @('scripts\evaluate_cassava_validation.py','--model',$Model,'--checkpoint',$Checkpoint,
        '--output',$Out,'--data-root',$DataRoot,'--manifest','manifests\cassava.csv',
        '--batch-size','16','--tta',$TTA)
    if ($RequireCuda) { $Args += '--require-cuda' }
    Push-Location $Repo
    try {
        & $Python @Args
        if ($LASTEXITCODE -ne 0) { throw "$Model TTA evaluation failed" }
    } finally { Pop-Location }
}
