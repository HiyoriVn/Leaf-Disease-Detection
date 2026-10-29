# 16 — EfficientNet-B3 integration and full validation (29/10/2026)

## Integration

The existing B3 app adapter was verified separately from this result: checkpoint `efficientnet_b3.ra2_in1k`, five local Cassava classes in manifest order, resize 343 → center crop 300, `INTER_CUBIC`, ImageNet mean/std. The working published MobileNet app default and all checkpoint files remain unchanged. See the app integration notes and `configs/model_registry.json` for model selection. The returned results identify PyTorch 2.8.0+cu128, timm 1.0.20, CUDA on an NVIDIA RTX 3060, batch 16 and AMP enabled.

## Returned full validation and integrity checks

The supplied `efficientnet_b3_validation_results.zip` contains the required metrics, class report, confusion matrix CSV/PNG, 3,210 predictions, 542 misclassifications, log and run metadata. Laptop audit recomputed every row from `predictions.csv`, checked probability vectors and argmax, recomputed class and aggregate metrics, compared both confusion matrix forms, and confirmed `misclassified.csv` equals the filtered predictions.

| Integrity item | Observed | Result |
|---|---|---|
| Validation predictions | 3,210; 2,668 correct, 542 wrong | Verified |
| Checkpoint SHA-256 | `7924296913c5f3a6c9d822212cdb0a8326b6d5c4cd2778078d6c47bebd0d406b` | Matches local full B3 checkpoint and archive |
| Manifest SHA-256 | `570b0d52ea950dec13de0eed792614b67c039b08b2cc28c8e588bb87b2f61082` | Matches `manifests/cassava.csv` |
| Returned ZIP SHA-256 | `668f9a080e3a8b0e56cf28e26a4b87c8785485988adddb49d206be752a3ceb31` | Captured from supplied archive |
| Split counts | 14,978 train / 3,210 val / 3,209 internal_test | Verified; only val predictions scored |
| Evaluator run | CUDA / RTX 3060 / FP16 autocast / 16 batch / 24.75 s | Consistent with `run_info.json` and completion log |
| Best checkpoint metadata | `best_epoch=6`, `val_macro_f1=0.7051832563873977` | Exact match to validation metric |

The archive contains an **evaluation** log, not the B3 training `epochs.jsonl` or `run.json`. Therefore the validation metric matches the best-epoch metadata embedded in the checkpoint, but the epoch-6 training record and complete learning curve cannot be independently reproduced from the returned files. The `reports/training_cassava/full_backbone_smooth.json` file is a MobileNetV3-Small run, not the B3 epoch history; it is not used as evidence for B3.

## Observed full-validation metrics

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| CBB | 163 | 0.5077 | 0.6074 | 0.5531 |
| CBSD | 328 | 0.7186 | 0.6463 | 0.6806 |
| CGM | 358 | 0.7687 | 0.6592 | 0.7098 |
| CMD | 1,974 | 0.9195 | 0.9493 | 0.9342 |
| Healthy | 387 | 0.6587 | 0.6382 | 0.6483 |

Accuracy is **0.8312**, macro-F1 **0.7052**, and balanced accuracy **0.7001**. These figures recompute exactly (within floating point precision) from the 3,210 predictions and agree with `metrics.json`, the report CSV and the log. The checkpoint contains the same best validation macro-F1.

The 542 errors include 39 CBB→Healthy and 42 Healthy→CBB, and 70 CGM→CMD. These are label confusions on this fixed source dataset; they do not prove biological equivalence or model calibration. The row-normalized confusion fractions and more pairs are in [report 17](17_cassava_model_comparison.md).

## Scope and limitations

This is a full evaluation of the fixed validation split, but not an independent field evaluation. The images belong to the same Kaggle Cassava collection used by the project, and this validation set was used during model selection. Do not call these external-generalization metrics. `internal_test` remained excluded. The report PNG is a visualization of the checked confusion counts; counts are verified against predictions.

Audit can be repeated with `scripts/audit_b3_validation_results.py`. B3-V2 and B3-V3 checkpoints have since been supplied and verified; their aggregate values and limitations are recorded in [report 19](19_phase3e_results.md). Row-level GPU evaluation and comparison on PC are prepared but still pending.
