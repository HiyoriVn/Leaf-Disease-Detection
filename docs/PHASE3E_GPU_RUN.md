# Phase 3E B3 experiments on the RTX 3060 PC

The returned B3 validation archive is verified and documented in reports 16–18. This bundle prepares two controlled experiments; it does not contain checkpoints or dataset images and no experiment has been started.

## Create and transfer the source bundle

On the laptop:

```powershell
cd C:\Users\Admin\Downloads\Leaf-Disease-Detection
.\scripts\prepare_phase3e_bundle.ps1 -Destination .\transfer\phase3e
Get-FileHash .\transfer\phase3e\Leaf-Disease-Detection-phase3e.zip -Algorithm SHA256
```

Copy the ZIP to the PC, for example `E:\CassavaTransfer`. On the PC, preserve existing work and extract into an empty source staging directory, then merge only source files into the existing repo. Do not copy any `checkpoints` directory from the bundle (none is included):

```powershell
$repo = 'D:\CassavaWork\Leaf-Disease-Detection'
$stage = 'D:\CassavaWork\phase3e-source'
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Expand-Archive -LiteralPath 'E:\CassavaTransfer\Leaf-Disease-Detection-phase3e.zip' -DestinationPath $stage
Copy-Item "$stage\configs\cassava_efficientnet_b3_phase3e_*.yaml" "$repo\configs\" -Force
Copy-Item "$stage\scripts\train_cassava.py" "$repo\scripts\" -Force
Copy-Item "$stage\scripts\run_train.ps1" "$repo\scripts\" -Force
Copy-Item "$stage\scripts\run_phase3e.ps1" "$repo\scripts\" -Force
```

The existing PC B3 environment, ImageNet weights, frozen manifest, data and trained baseline checkpoint are reused. Confirm the baseline checkpoint exists and its SHA is `7924296913c5f3a6c9d822212cdb0a8326b6d5c4cd2778078d6c47bebd0d406b`; it is for reference only and is not loaded by Phase 3E. Verify config weights and manifest hashes before running. Python environment setup is only needed if the PC environment was removed; it may require network access.

## Run a technical pilot before each full experiment

In PowerShell on the RTX 3060 PC:

```powershell
cd D:\CassavaWork\Leaf-Disease-Detection
.\scripts\run_phase3e.ps1 -Experiment weighted -RunName cassava-b3-v2-weighted-pilot -DataRoot D:\CassavaWork\data\extracted -BatchSize 16 -Pilot -RequireCuda
.\scripts\run_phase3e.ps1 -Experiment smoothing -RunName cassava-b3-v3-smoothing-pilot -DataRoot D:\CassavaWork\data\extracted -BatchSize 16 -Pilot -RequireCuda
```

Pilot uses only a small balanced subset and is not evidence of model performance. Check `run.json`, `epochs.jsonl` and `.console.log`; verify CUDA, manifest hash, input size, and no exceptions. Then run full training with fresh run names:

```powershell
.\scripts\run_phase3e.ps1 -Experiment weighted -RunName cassava-b3-v2-weighted -DataRoot D:\CassavaWork\data\extracted -BatchSize 16 -RequireCuda
.\scripts\run_phase3e.ps1 -Experiment smoothing -RunName cassava-b3-v3-smoothing -DataRoot D:\CassavaWork\data\extracted -BatchSize 16 -RequireCuda
```

If a full run has CUDA OOM, leave its checkpoint directory intact and restart from scratch with a distinct run name and `-BatchSize 8`. For interruption recovery, use `scripts\run_train.ps1 -RunName <same-run> -Config <same-config> -DataRoot D:\CassavaWork\data\extracted -BatchSize 16 -FineStage last -Resume -RequireCuda`; all original parameters must match. Do not use `-Resume` for a different config.

Each run writes `best.pth`, `last.pth`, `run.json`, `epochs.jsonl` and console output under `checkpoints\experiments\<RunName>`. Keep the original baseline folder unchanged. Return both complete run directories (or ZIP them separately with logs) for analysis. Do not run internal-test evaluation until the Phase 3E candidates are analyzed and one candidate is frozen.
