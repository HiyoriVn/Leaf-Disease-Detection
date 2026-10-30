# EfficientNet-B3 Cassava: preparation and technical pilot (29/10/2026)

## Configuration

New config `configs/cassava_efficientnet_b3.yaml` selects timm `efficientnet_b3.ra2_in1k`, ImageNet-1k pretrained weights, 300×300 input (343 resize, center crop, cubic interpolation), ImageNet mean/std, CUDA AMP, initial CUDA batch 16, head training then final two backbone blocks. `-FineStage all` is available as a separate experiment. MobileNetV3-Small config and working app remain unchanged. The [source model card](https://huggingface.co/timm/efficientnet_b3.ra2_in1k) lists Apache-2.0 and original training/testing sizes 288/320; 300 is our chosen experimental size.

Local safetensors file is 49,335,454 bytes, SHA-256 `279d2a53898aa89dab43fd6bd7df9f706aea4cb9cf916223988bbcb8e5850469`. `strict=True` loaded every key into a 1,000-class timm model; after resetting the classifier to five Cassava classes, a `[1,3,300,300]` tensor produced `[1,5]`. Verified with PyTorch 2.8.0+cu128 and timm 1.0.20. `scripts/verify_pretrained.py` repeats these checks on the destination PC.

## Small pilot actually run

One **CPU** pilot completed locally: two training and one validation image per class (10/5 total), batch 2, head stage only, one epoch, about 26 seconds. It produced `checkpoints/experiments/efficientnet_b3_pilot_local_20261029/best.pth` and `last.pth`; best checkpoint SHA-256 `be66172f657e5dc545bfce15ef3dfb466f4936b8eed059ac8364bdc6f0dd1186`. Pilot validation macro-F1 was 0.08 on **five images**. This number is purely a technical smoke result and says nothing reliable about model quality. No full training or internal-test evaluation ran. A resume invocation using the same config completed without another training epoch. Four focused training tests passed, including the unchanged MobileNetV3 preprocessing contract and B3 input/weights check.

The frozen Cassava manifest SHA-256 is `570b0d52ea950dec13de0eed792614b67c039b08b2cc28c8e588bb87b2f61082`; its 14,978/3,210/3,209 train/val/internal_test rows are checked before training. Only train and val images enter DataLoaders. Best model selection uses validation macro-F1. Internal test must remain untouched until final selection. The evaluator now supports B3 with 300×300 preprocessing but has **not** been run for B3.

## Transfer and next run

Recreate the source bundle after these changes, then transfer the bundle, the B3 safetensors file and the Cassava `train_images` directory. Exact source and destination PowerShell commands, pilot, batch adjustment, full run and resume are in [WINDOWS_GPU_TRAINING.md](../docs/WINDOWS_GPU_TRAINING.md). Start with the PC pilot; if batch 16 runs out of VRAM during the full run, use a **new** run name with batch 8. Compare full-run validation macro-F1 with the Phase 3B MobileNetV3 result before choosing a model. Do not infer B3 performance from the local pilot.
