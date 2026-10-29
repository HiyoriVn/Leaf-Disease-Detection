param(
    [string]$Repo = 'D:\CassavaWork\Leaf-Disease-Detection',
    [string]$EvaluationRoot = (Join-Path $Repo 'reports\phase3e_validation'),
    [string]$Output = (Join-Path $Repo 'reports\phase3e_comparison')
)
$ErrorActionPreference = 'Stop'
$Python = Join-Path $Repo '.venv-train\Scripts\python.exe'
$Args = @('scripts\compare_model_predictions.py',
          '--baseline', (Join-Path $EvaluationRoot 'baseline\predictions.csv'),
          '--v2', (Join-Path $EvaluationRoot 'v2\predictions.csv'),
          '--v3', (Join-Path $EvaluationRoot 'v3\predictions.csv'),
          '--output', $Output)
Push-Location $Repo
try {
    & $Python @Args
    if ($LASTEXITCODE -ne 0) { throw "Comparison failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
