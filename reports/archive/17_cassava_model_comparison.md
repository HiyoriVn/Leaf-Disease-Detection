# 17 — Cassava model comparison (updated 29/10/2026)

## Same fixed validation split

| Model | Accuracy | Macro-F1 | Balanced accuracy | Evaluation status |
|---|---:|---:|---:|---|
| Published MobileNetV3-Small disease checkpoint | 0.8458 | 0.7688 | 0.7959 | Observed on all 3,210 validation rows; potentially contaminated because the checkpoint was trained on the same Kaggle dataset |
| Team MobileNetV3-Small, ImageNet initialized | 0.8262 | 0.7044 | 0.7036 | Observed on validation; checkpoint selected using this validation split |
| Team EfficientNet-B3, ImageNet initialized | 0.8312 | 0.7052 | 0.7001 | Returned RTX 3060 run, 3,210 rows, archive audit verified |
| B3-V2, tempered weighted CE | 0.8137 | 0.6948 | 0.7192 | Verified from the supplied 3,210-row GPU predictions and metrics |
| B3-V3, label smoothing 0.05 | 0.8355 | 0.7125 | 0.7048 | Verified from the supplied 3,210-row GPU predictions and metrics |

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

## V2/V3 row-level comparison

V3 fixes 48 images that baseline gets wrong and makes 34 baseline-correct images wrong (net +14 correct). V3 F1 is higher in CBB, CBSD, CMD and Healthy, and lower in CGM. V2 raises CBB recall from 0.6074 to 0.6994 and CGM recall from 0.6592 to 0.7207, but CMD recall falls from 0.9493 to 0.9048. Its CGM→CMD errors drop from 70 to 41, while CMD→CGM rises from 34 to 79. CBB→Healthy / Healthy→CBB counts are 39/42 baseline, 36/54 V2, and 46/36 V3.

Wrong predictions with confidence ≥0.90: 91 baseline, 88 V2 and 56 V3. At ≥0.95: 55, 46 and 24. Twenty-three rows are wrong with ≥0.95 confidence in all three. These are softmax scores, not calibrated certainty; review images and source labels before proposing data corrections.

## Probability ensemble post-processing

All inputs were verified to have the same ordered 3,210 image IDs and true labels, all five class probability columns, valid probability sums, and matching class mapping. The script forms a weighted arithmetic mean of class probabilities before argmax.

| Ensemble | Accuracy | Macro-F1 | Balanced accuracy | Correct |
|---|---:|---:|---:|---:|
| V2 50% + V3 50% | 0.82772586 | 0.70692666 | 0.71699379 | 2657/3210 |
| V2 25% + V3 75% | 0.83115265 | 0.70722528 | 0.70853840 | 2668/3210 |

Neither improves over standalone V3 in accuracy or macro-F1. The equal-weight blend increases balanced accuracy but decreases accuracy and macro-F1. These are post-hoc results on the reused validation set, not independent evidence.

## Evaluation integrity

The returned archive has 3,210 predictions, 2,668 correct, and 542 incorrect. Confusion-matrix CSV and JSON, classification CSV, metrics JSON, log totals and row-level predictions agree. The checkpoint SHA-256 is `7924296913c5f3a6c9d822212cdb0a8326b6d5c4cd2778078d6c47bebd0d406b`; manifest SHA-256 is `570b0d52ea950dec13de0eed792614b67c039b08b2cc28c8e588bb87b2f61082`. It is not an external test. No `internal_test` images were scored or used in this analysis.

The archive contains GPU evaluation logs/run info and all prediction artifacts. It reproduces the supplied validation metrics and epoch-6 selection values to stored precision, but not training epoch history. The MobileNet epoch JSON must not be mistaken for B3 training evidence. Full integrity details are in [report 19](19_phase3e_results.md).

## Decision

Keep all B3 checkpoints and the current app default unchanged until row-level GPU evaluations and class-wise comparisons are reviewed. Keep internal test sealed until model/config selection is complete. See [Phase 3E plan](18_phase3e_experiment_plan.md) and [Phase 3E results](19_phase3e_results.md).
