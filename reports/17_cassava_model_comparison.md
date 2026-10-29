# 17 — Cassava model comparison (updated 29/10/2026)

## Same fixed validation split

| Model | Accuracy | Macro-F1 | Balanced accuracy | Evaluation status |
|---|---:|---:|---:|---|
| Published MobileNetV3-Small disease checkpoint | 0.8458 | 0.7688 | 0.7959 | Observed on all 3,210 validation rows; potentially contaminated because the checkpoint was trained on the same Kaggle dataset |
| Team MobileNetV3-Small, ImageNet initialized | 0.8262 | 0.7044 | 0.7036 | Observed on validation; checkpoint selected using this validation split |
| Team EfficientNet-B3, ImageNet initialized | 0.8312 | 0.7052 | 0.7001 | Returned RTX 3060 run, 3,210 rows, archive audit verified |

Within the two team-trained models, B3 exceeds MobileNetV3 by only **0.00498 absolute accuracy** (about 0.50 percentage points) and **0.00078 macro-F1**, while balanced accuracy is lower by **0.00345**. That tiny macro-F1 difference is not persuasive evidence that B3 is better; no repeated seeds or uncertainty intervals are available. B3 has not demonstrated a practical gain over the existing MobileNet. Both were selected/evaluated on the same in-source validation protocol, so results are useful for engineering comparison, not external generalization.

The published disease-specific checkpoint scores higher on this split, but same-source training overlap makes that comparison potentially contaminated. Do not interpret its advantage as evidence of better out-of-source performance. CropNet remains taxonomy-incompatible because it uses “Green Mite” and “Unknown” rather than the project's five-class contract.

## EfficientNet-B3 class report

| Class | Support | Precision | Recall | F1 | Errors from this true class |
|---|---:|---:|---:|---:|---:|
| CBB | 163 | 0.5077 | 0.6074 | 0.5531 | 64 |
| CBSD | 328 | 0.7186 | 0.6463 | 0.6806 | 116 |
| CGM | 358 | 0.7687 | 0.6592 | 0.7098 | 122 |
| CMD | 1,974 | 0.9195 | 0.9493 | 0.9342 | 100 |
| Healthy | 387 | 0.6587 | 0.6382 | 0.6483 | 140 |

## Most frequent directed confusions

Rows are ground truth; columns are prediction. Percentages use the support for the true class.

| True → predicted | Count | Share of true class |
|---|---:|---:|
| CGM → CMD | 70 | 19.6% |
| Healthy → CMD | 45 | 11.6% |
| Healthy → CBB | 42 | 10.9% |
| CBB → Healthy | 39 | 23.9% |
| CBSD → CMD | 39 | 11.9% |
| CBSD → CBB | 36 | 11.0% |
| CMD → CGM | 34 | 1.7% |
| CBSD → Healthy | 33 | 10.1% |
| CGM → Healthy | 30 | 8.4% |

The CBB/Healthy pair is roughly balanced in count (39 vs 42), but CBB→Healthy affects a larger fraction of its smaller class. CGM→CMD is asymmetric and comprises 19.6% of CGM validation examples. These are priorities for a targeted validation error review; do not relabel examples or merge categories from model output alone.

## Evaluation integrity

The returned archive has 3,210 predictions, 2,668 correct, and 542 incorrect. Confusion-matrix CSV and JSON, classification CSV, metrics JSON, log totals and row-level predictions agree. The checkpoint SHA-256 is `7924296913c5f3a6c9d822212cdb0a8326b6d5c4cd2778078d6c47bebd0d406b`; manifest SHA-256 is `570b0d52ea950dec13de0eed792614b67c039b08b2cc28c8e588bb87b2f61082`. It is not an external test. No `internal_test` images were scored or used in this analysis.

The supplied archive does not contain B3 training epoch logs. `best_epoch=6` and `val_macro_f1=0.7051832563873977` are consistent with checkpoint metadata, not independently reproduced from an epoch log. The MobileNet epoch JSON must not be mistaken for B3 training evidence.

## Decision

Keep both current checkpoints and the current app default unchanged. Do not promote B3 based on a 0.0008 macro-F1 difference. Phase 3E should use controlled training-only changes, select by validation macro-F1, record per-class recall, and keep internal test sealed until model/config selection is complete. See [Phase 3E plan](18_phase3e_experiment_plan.md).
