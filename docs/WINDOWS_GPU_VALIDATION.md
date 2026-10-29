# EfficientNet-B3 validation on the RTX 3060 PC

This procedure updates source only. The evaluation bundle contains no checkpoint or dataset. Do **not** run training or evaluate `internal_test`.

## 1. Create and copy the laptop bundle

On the laptop:

```powershell
cd C:\Users\Admin\Downloads\Leaf-Disease-Detection
.\scripts\prepare_evaluation_bundle.ps1 -Destination .\transfer\evaluation
Get-FileHash .\transfer\evaluation\Leaf-Disease-Detection-evaluation.zip -Algorithm SHA256
Get-FileHash .\manifests\cassava.csv -Algorithm SHA256
```

Copy `C:\Users\Admin\Downloads\Leaf-Disease-Detection\transfer\evaluation\Leaf-Disease-Detection-evaluation.zip` to the USB drive. The current fixed manifest hash is `570b0d52ea950dec13de0eed792614b67c039b08b2cc28c8e588bb87b2f61082`.

## 2. Update source on the PC without touching the trained checkpoint

Assuming the USB appears as `E:`:

```powershell
$repo = 'D:\CassavaWork\Leaf-Disease-Detection'
$zip = 'E:\CassavaTransfer\Leaf-Disease-Detection-evaluation.zip'
$stage = 'D:\CassavaWork\evaluation-source-update'
Expand-Archive -LiteralPath $zip -DestinationPath $stage -Force
Copy-Item "$stage\scripts\evaluate_cassava_validation.py" "$repo\scripts\" -Force
Copy-Item "$stage\scripts\run_evaluate_b3.ps1" "$repo\scripts\" -Force
Copy-Item "$stage\scripts\train_cassava.py" "$repo\scripts\" -Force
Copy-Item "$stage\src\cassava_inference.py" "$repo\src\" -Force
Copy-Item "$stage\src\classification_metrics.py" "$repo\src\" -Force
Copy-Item "$stage\configs\cassava_efficientnet_b3_app.json" "$repo\configs\" -Force
```

The ZIP has no files under `checkpoints` and no image files. The trained checkpoint at `D:\CassavaWork\Leaf-Disease-Detection\checkpoints\experiments\cassava-b3-full\best.pth` is not overwritten. The runner uses the existing PC manifest; it must match the fixed hash above.

## 3. Verify CUDA, checkpoint and manifest

```powershell
cd D:\CassavaWork\Leaf-Disease-Detection
& .\.venv-train\Scripts\python.exe -c "import torch,timm; print('torch',torch.__version__,'timm',timm.__version__,'cuda',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
(Get-FileHash .\checkpoints\experiments\cassava-b3-full\best.pth -Algorithm SHA256).Hash
(Get-FileHash .\manifests\cassava.csv -Algorithm SHA256).Hash
```

Expected checkpoint hash: `7924296913c5f3a6c9d822212cdb0a8326b6d5c4cd2778078d6c47bebd0d406b`. Expected manifest hash: `570b0d52ea950dec13de0eed792614b67c039b08b2cc28c8e588bb87b2f61082`. Stop if either differs. The script also enforces both hashes and the checkpoint class/preprocessing metadata.

## 4. Run exactly the validation split

```powershell
$out = 'D:\CassavaWork\Leaf-Disease-Detection\reports\efficientnet_b3_validation_20261030'
.\scripts\run_evaluate_b3.ps1 -DataRoot 'D:\CassavaWork\data\extracted' -Checkpoint 'D:\CassavaWork\Leaf-Disease-Detection\checkpoints\experiments\cassava-b3-full\best.pth' -OutputDir $out -BatchSize 16 -RequireCuda
```

This evaluates only the 3,210 manifest rows marked `val`, uses 300×300 preprocessing, CUDA and FP16 autocast, and prints progress every ten batches. In another PowerShell window, monitor it with:

```powershell
Get-Content 'D:\CassavaWork\Leaf-Disease-Detection\reports\efficientnet_b3_validation_20261030\evaluation.log' -Wait
```

If CUDA reports out of memory, use a new output directory and `-BatchSize 8`; never overwrite or reuse an incomplete output directory. If CUDA is unavailable, `-RequireCuda` causes a nonzero failure rather than silently running the long evaluation on CPU. The script exits nonzero on any error and records it in `evaluation.log`.

## 5. Confirm and return the result

```powershell
$out = 'D:\CassavaWork\Leaf-Disease-Detection\reports\efficientnet_b3_validation_20261030'
Get-ChildItem $out | Select-Object Name,Length
(Import-Csv "$out\predictions.csv").Count
(Get-Content "$out\metrics.json" -Raw | ConvertFrom-Json).n
Compress-Archive -Path "$out\*" -DestinationPath 'D:\CassavaWork\efficientnet_b3_validation_20261030.zip' -CompressionLevel Optimal
```

Both counts must be 3210. Confirm the folder has `metrics.json`, `classification_report.csv`, `confusion_matrix.csv`, `confusion_matrix.png`, `predictions.csv`, `misclassified.csv`, `evaluation.log`, and `run_info.json`. Copy `D:\CassavaWork\efficientnet_b3_validation_20261030.zip` back to the USB, then to `C:\Users\Admin\Downloads\Leaf-Disease-Detection\transfer\evaluation-results\` on the laptop. Codex can analyze the returned archive next; no PC access or Codex installation is needed.

## Common issues

- **CUDA false:** confirm the NVIDIA driver and that `.venv-train` is the PC's CUDA environment.
- **SHA mismatch:** stop; do not substitute weights or manifest. Check that the right checkpoint/archive was copied.
- **Missing train image:** ensure the PC data root contains `cassava-leaf-disease-classification\train_images`.
- **CUDA out of memory:** restart with batch 8 and a new output folder.
- **Output folder exists:** choose a new folder name; the runner intentionally refuses to overwrite results.
