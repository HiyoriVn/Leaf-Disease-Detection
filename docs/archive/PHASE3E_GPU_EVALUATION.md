# Phase 3E GPU evaluation — baseline, V2 and V3

This workflow evaluates the fixed **3,210 validation** rows only. It verifies the checkpoint hash for each model and the fixed manifest hash. It never scores `internal_test`. Do not copy laptop checkpoints over PC files; the PC's trained checkpoints and data stay in place.

## Copy source-only ZIP to the PC

Create the ZIP on the laptop:

```powershell
cd C:\Users\Admin\Downloads\Leaf-Disease-Detection
.\scripts\prepare_phase3_bundle.ps1 -Destination .\transfer\phase3e3f
Get-FileHash .\transfer\phase3e3f\Leaf-Disease-Detection-phase3e3f-source.zip -Algorithm SHA256
```

Copy `Leaf-Disease-Detection-phase3e3f-source.zip` to `D:\CassavaWork` on the PC (for example, copy it there from USB). On the PC, merge updated source files into the existing repository:

```powershell
$repo = 'D:\CassavaWork\Leaf-Disease-Detection'
$zip = 'D:\CassavaWork\Leaf-Disease-Detection-phase3e3f-source.zip'
$stage = 'D:\CassavaWork\phase3e3f-source-stage'
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Expand-Archive -LiteralPath $zip -DestinationPath $stage
  $manifestBefore = (Get-FileHash -LiteralPath "$repo\manifests\cassava.csv" -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($manifestBefore -ne '570b0d52ea950dec13de0eed792614b67c039b08b2cc28c8e588bb87b2f61082') { throw "Existing manifest SHA mismatch: $manifestBefore" }
  Get-ChildItem -LiteralPath $stage -File -Recurse | Where-Object { $_.FullName -notmatch '\\manifests\\' } | ForEach-Object {
      $relative = [System.IO.Path]::GetRelativePath($stage, $_.FullName)
      $target = Join-Path $repo $relative
      New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
      Copy-Item -LiteralPath $_.FullName -Destination $target -Force
  }
  $manifestAfter = (Get-FileHash -LiteralPath "$repo\manifests\cassava.csv" -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($manifestAfter -ne $manifestBefore) { throw "Manifest changed during source update: $manifestAfter" }
```

The ZIP contains source/config/manifest/docs only. It excludes data images, pretrained weight files, and trained checkpoints. This selective copy adds the new Phase 3F configs and preserves all Phase 3E configs, weights, checkpoints, dataset and environment. Keep the PC's frozen manifest; verify its SHA-256 below instead of replacing or regenerating it. The existing `run_train.ps1` and Phase 3E configs from the earlier transfer remain in use.

## Verify and run the exact validation

```powershell
cd D:\CassavaWork\Leaf-Disease-Detection
$expectedManifest = '570b0d52ea950dec13de0eed792614b67c039b08b2cc28c8e588bb87b2f61082'
$actualManifest = (Get-FileHash .\manifests\cassava.csv -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualManifest -ne $expectedManifest) { throw "Manifest SHA mismatch: $actualManifest" }
  .\scripts\run_phase3f_tta.ps1 -Models v3 -TTA hflip -Repo $PWD.Path -DataRoot 'D:\CassavaWork\data\extracted' -OutputRoot 'D:\CassavaWork\Leaf-Disease-Detection\reports\phase3f_v3_hflip' -RequireCuda
  .\scripts\run_phase3f_tta.ps1 -Models v3 -TTA dihedral -Repo $PWD.Path -DataRoot 'D:\CassavaWork\data\extracted' -OutputRoot 'D:\CassavaWork\Leaf-Disease-Detection\reports\phase3f_v3_dihedral' -RequireCuda
```

The runner expects baseline, V2 and V3 checkpoints at:

- `checkpoints\experiments\cassava-b3-full\best.pth`
- `checkpoints\experiments\cassava-b3-v2-weighted\best.pth`
- `checkpoints\experiments\cassava-b3-v3-smoothing\best.pth`

Expected SHA-256 respectively: baseline `7924296913c5f3a6c9d822212cdb0a8326b6d5c4cd2778078d6c47bebd0d406b`, V2 `7398eb98fe0917d330baea90c5c34c6160e2e781aab31dc7f9c73c9c653e2901`, V3 `0e76a24e1d979ee29f508b83fe6059f89f7a13a167effcaddc3eacfea0cf669c`. The evaluator independently checks checkpoint and manifest hashes. If the checkpoint paths differ, pass `-Models` only after editing the runner mapping to the correct existing locations; keep the expected hashes unchanged.

Each model folder gets `metrics.json`, `classification_report.csv`, `confusion_matrix.csv`, `confusion_matrix.png`, `predictions.csv`, `misclassified.csv`, `evaluation.log`, and `run_info.json`. A failed or incomplete output directory is never overwritten; choose a new `OutputRoot` before retrying.

## Compare and return the results

```powershell
.\scripts\run_phase3e_comparison.ps1 -Repo $PWD.Path -EvaluationRoot 'D:\CassavaWork\Leaf-Disease-Detection\reports\phase3e_validation_20261030' -Output 'D:\CassavaWork\Leaf-Disease-Detection\reports\phase3e_comparison_20261030'
Compress-Archive -Path 'D:\CassavaWork\Leaf-Disease-Detection\reports\phase3e_validation_20261030\*','D:\CassavaWork\Leaf-Disease-Detection\reports\phase3e_comparison_20261030\*' -DestinationPath 'D:\CassavaWork\phase3e_gpu_results_20261030.zip'
```

Return that results ZIP to the laptop for per-class gains/regressions and targeted error analysis. It includes predictions so it contains filenames and probabilities; transfer it only through the agreed private project channel. No Streamlit default changes are part of this workflow.
