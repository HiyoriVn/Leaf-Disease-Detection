# Phase 3F — Controlled diagnosis and RTX 3060 plan

The supplied Phase 3E final GPU archive was verified: baseline, V2 and V3 have the same ordered 3,210 IDs/labels, all five probability columns and matching checkpoint/manifest hashes. Verified class results and paired errors are in [report 19](../reports/19_phase3e_results.md) and [report 17](../reports/17_cassava_model_comparison.md). Existing V2/V3 probability means (50/50 and 25/75) are computed; neither beats V3 macro-F1. The controlled hypotheses and pipeline audit are in [report 20](../reports/20_phase3f_diagnosis.md). No PC run is launched automatically.

## Transfer source-only updates to `D:\CassavaWork`

On the laptop:

```powershell
cd C:\Users\Admin\Downloads\Leaf-Disease-Detection
.\scripts\prepare_phase3_bundle.ps1 -Destination .\transfer\phase3e3f
Get-FileHash .\transfer\phase3e3f\Leaf-Disease-Detection-phase3e3f-source.zip -Algorithm SHA256
```

Copy the ZIP into `D:\CassavaWork` on the PC, then follow [the Phase 3E transfer/update section](PHASE3E_GPU_EVALUATION.md#copy-source-only-zip-to-the-pc). The source ZIP has no dataset images or checkpoints. Do not overwrite `checkpoints\experiments\cassava-b3-full`, V2/V3, or any ImageNet weights.

## Ordered execution

1. Run the prespecified no-training suite below; keep V3 no-TTA as the comparator. Existing V2/V3 50/50 and 25/75 results are recorded; do not retune blend weights on this validation split.
2. Treat TTA runs as diagnostics on reused validation data and record latency. Avoid promotion based solely on choosing the best of repeated validation variants.
3. Review TTA outcomes and error examples, then run the B3-V4 pilot. If path/config/VRAM are sound, run V4 full; keep V3 unchanged as comparator.
4. Run B3-V5 as a separate controlled experiment, not combined with V4. Compare each against V3 on identical validation rows.
5. EfficientNet-B4 remains optional only if B3 gains stall; pilot at batch 8, retry batch 4 with a new run name after OOM.
6. Freeze model and preprocessing before internal-test evaluation. Never use internal-test output for model or TTA selection.

## Pilot commands (after code transfer; not run yet)

```powershell
cd D:\CassavaWork\Leaf-Disease-Detection
.\scripts\run_phase3f.ps1 -Experiment v4 -RunName cassava-b3-v4-pilot -Repo $PWD.Path -DataRoot 'D:\CassavaWork\data\extracted' -Pilot -RequireCuda
.\scripts\run_phase3f.ps1 -Experiment v5 -RunName cassava-b3-v5-pilot -Repo $PWD.Path -DataRoot 'D:\CassavaWork\data\extracted' -Pilot -RequireCuda
```

Only after reviewing pilot logs and confirming VRAM/manifest/pretrained hashes, launch distinct full run names:

```powershell
.\scripts\run_phase3f.ps1 -Experiment v4 -RunName cassava-b3-v4-symptom -Repo $PWD.Path -DataRoot 'D:\CassavaWork\data\extracted' -RequireCuda
.\scripts\run_phase3f.ps1 -Experiment v5 -RunName cassava-b3-v5-full-cosine -Repo $PWD.Path -DataRoot 'D:\CassavaWork\data\extracted' -RequireCuda
```

EfficientNet-B4 is optional and downloads ImageNet weights from the configured timm model on the PC (network required unless cached):

```powershell
& .\.venv-train\Scripts\python.exe scripts\prepare_timm_weights.py --architecture efficientnet_b4.ra2_in1k --output checkpoints\imagenet_efficientnet_b4_ra2_in1k.safetensors
.\scripts\run_phase3f.ps1 -Experiment b4 -RunName cassava-b4-pilot -Repo $PWD.Path -DataRoot 'D:\CassavaWork\data\extracted' -BatchSize 8 -Pilot -RequireCuda
```

After a successful pilot, a separate full run may use batch 8. Record the printed pretrained weights SHA and actual run `run.json` SHA. If CUDA OOM, preserve the failed run and retry with a new run name at batch 4.

After any V4/V5/B4 full run is selected by validation macro-F1, evaluate its checkpoint with an explicit measured SHA (replace the path with the chosen `best.pth`):

```powershell
$checkpoint = 'D:\CassavaWork\Leaf-Disease-Detection\checkpoints\experiments\cassava-b3-v4-symptom\best.pth'
$checkpointSha = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
& .\.venv-train\Scripts\python.exe scripts\evaluate_cassava_validation.py --model custom --expected-checkpoint-sha256 $checkpointSha --checkpoint $checkpoint --output 'reports\phase3f_v4_validation' --data-root 'D:\CassavaWork\data\extracted' --manifest 'manifests\cassava.csv' --batch-size 16 --require-cuda
```

The evaluator still enforces the frozen manifest SHA and all 3,210 validation rows. Compare new prediction files to V3 like this:

```powershell
& .\.venv-train\Scripts\python.exe scripts\compare_model_predictions.py --baseline 'reports\phase3e_validation_20261030\v3\predictions.csv' --candidate 'v4=reports\phase3f_v4_validation\predictions.csv' --output 'reports\phase3f_v4_vs_v3'
```

## TTA and probability averaging

The no-training configurations are fixed in advance: (1) V3 no-TTA, already observed; (2) V3 horizontal flip TTA (identity + horizontal flip); (3) V3 8-view dihedral TTA (four rotations, with and without horizontal reflection); (4) V2/V3 50/50 probability mean; and (5) V2/V3 25/75 probability mean. The last two are already post-processed. Do not search weights or TTA subsets. Dihedral transforms are a plausible orientation-invariance diagnostic for leaf images, but the effect is empirical.

Run TTA on the PC, each into a fresh output directory:

```powershell
cd D:\CassavaWork\Leaf-Disease-Detection
.\scripts\run_phase3f_tta.ps1 -Models v3 -TTA hflip -Repo $PWD.Path -DataRoot 'D:\CassavaWork\data\extracted' -OutputRoot 'D:\CassavaWork\Leaf-Disease-Detection\reports\phase3f_v3_hflip' -RequireCuda
.\scripts\run_phase3f_tta.ps1 -Models v3 -TTA dihedral -Repo $PWD.Path -DataRoot 'D:\CassavaWork\data\extracted' -OutputRoot 'D:\CassavaWork\Leaf-Disease-Detection\reports\phase3f_v3_dihedral' -RequireCuda
```

The evaluator checks the model-specific checkpoint SHA and frozen manifest hash, then writes all 3,210 rows, metrics, per-class CSV, confusion matrix, predictions, misclassified rows and latency/run info. Each output path must be fresh. Eight-view TTA takes eight forwards per batch; prior no-TTA B3 validation took 24.75 seconds. A linear 8× reference is ~198 seconds, an estimate only; use actual latency in run info. CSV probability means are valid only with identical class order and row sequence; the script checks those conditions.

## Measured compute references

The available local GTX 1650 Max-Q MobileNet run took 1,358.46 seconds for eight epochs; this is not a B3/RTX 3060 estimate. The PC B3 training epoch logs were not supplied, so do not extrapolate exact full-training time for V4/V5/B4. Each PC pilot and first full epoch should record elapsed time and peak VRAM; use that measured timing to plan the remaining epochs. See [report 20](../reports/20_phase3f_diagnosis.md).
