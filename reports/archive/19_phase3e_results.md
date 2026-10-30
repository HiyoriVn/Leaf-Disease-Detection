# 19 — Phase 3E final GPU results and Phase 3F handoff (29/10/2026)

## Evidence received and checkpoint verification

The supplied `phase3e_gpu_results_final.zip` SHA-256 is `5e89f36189d3d75b5bc288cb7140a23955eac174ac39cceca16d2fd78c733124`. The archive contains baseline/V2/V3 results, paired transitions, class metrics, logs and run info. All three prediction files have 3,210 unique image IDs in the same order, matching true indices and class names. Every row has five probability columns (`prob_0` … `prob_4`); probabilities sum to 1 within 2e-5 and their argmax agrees with the stored prediction. Recomputed confusion matrices match `metrics.json`. The frozen manifest SHA is `570b0d52ea950dec13de0eed792614b67c039b08b2cc28c8e588bb87b2f61082`. GPU run records identify RTX 3060/CUDA, batch 16, AMP. Checkpoint SHA values match the expected model-specific hashes below. No internal-test records were used.

| Model | Best epoch | Accuracy | Macro-F1 | Balanced accuracy | Correct | Checkpoint SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| B3 baseline | 6 | 0.83115265 | 0.70518326 | 0.70010113 | 2668/3210 | `7924296913c5f3a6c9d822212cdb0a8326b6d5c4cd2778078d6c47bebd0d406b` |
| B3-V2 tempered weighted CE | 6 | 0.81370717 | 0.69478861 | 0.71920369 | 2612/3210 | `7398eb98fe0917d330baea90c5c34c6160e2e781aab31dc7f9c73c9c653e2901` |
| B3-V3 label smoothing 0.05 | 6 | 0.83551402 | 0.71245032 | 0.70484034 | 2682/3210 | `0e76a24e1d979ee29f508b83fe6059f89f7a13a167effcaddc3eacfea0cf669c` |

The manifest remains `570b0d52ea950dec13de0eed792614b67c039b08b2cc28c8e588bb87b2f61082`, with 14,978 train, 3,210 validation and 3,209 internal-test rows. No internal-test results were supplied or used.

The row-level metrics reproduce the supplied epoch-6 validation metrics exactly to stored precision; the summary JSON, per-class CSV and confusion matrix were independently re-read/recomputed from predictions. This verifies the evaluation artifact against the log/metadata values, not the training process itself.

## Per-class results and errors

| Model | Class | Support | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| baseline | CBB | 163 | 0.5077 | 0.6074 | 0.5531 |
| baseline | CBSD | 328 | 0.7186 | 0.6463 | 0.6806 |
| baseline | CGM | 358 | 0.7687 | 0.6592 | 0.7098 |
| baseline | CMD | 1974 | 0.9195 | 0.9493 | 0.9342 |
| baseline | Healthy | 387 | 0.6587 | 0.6382 | 0.6483 |
| V2 | CBB | 163 | 0.4711 | 0.6994 | 0.5630 |
| V2 | CBSD | 328 | 0.7104 | 0.6433 | 0.6752 |
| V2 | CGM | 358 | 0.6701 | 0.7207 | 0.6945 |
| V2 | CMD | 1974 | 0.9485 | 0.9048 | 0.9261 |
| V2 | Healthy | 387 | 0.6030 | 0.6279 | 0.6152 |
| V3 | CBB | 163 | 0.5385 | 0.6012 | 0.5681 |
| V3 | CBSD | 328 | 0.7564 | 0.6341 | 0.6899 |
| V3 | CGM | 358 | 0.7785 | 0.6480 | 0.7073 |
| V3 | CMD | 1974 | 0.9233 | 0.9509 | 0.9369 |
| V3 | Healthy | 387 | 0.6327 | 0.6899 | 0.6601 |

V3 F1 is higher than baseline for CBB, CBSD, CMD and Healthy, but lower for CGM. V2 recall rises for CBB (+9.20 percentage points) and CGM (+6.15 pp), while CMD recall falls by 4.46 pp. Paired errors show V3 fixes 48 baseline errors and introduces 34 new errors (net +14 correct). This is the exact row-level paired count, not a rounded-metric inference.

| Error direction | Baseline | V2 | V3 |
|---|---:|---:|---:|
| CBB → Healthy | 39 | 36 | 46 |
| Healthy → CBB | 42 | 54 | 36 |
| CGM → CMD | 70 | 41 | 68 |
| CMD → CGM | 34 | 79 | 31 |

V2 substantially reduces CGM→CMD but shifts many more CMD examples into CGM, matching the lower CMD recall. CBB/Healthy confusions remain material for every model.

Wrong predictions with model confidence ≥0.90: baseline 91/542, V2 88/598, V3 56/528. At ≥0.95: 55, 46 and 24 respectively. Twenty-three validation images are wrong at ≥0.95 for all three models. Examples include CBSD→CMD (`1244824731.jpg`, `1683677421.jpg`), CMD→CBSD (`1403621003.jpg`), CMD→CGM (`1999385559.jpg`, `2140444069.jpg`), CGM→CMD (`2602722844.jpg`) and Healthy→CBB (`1485753217.jpg`, `1562043567.jpg`, `3609301419.jpg`). These are softmax scores, not calibrated probabilities or proof of label error; inspect image/label provenance before drawing conclusions.

## Prespecified probability ensemble diagnostics

The CSVs passed ID/order, ground-truth mapping, all-five-probability, sum-to-one and argmax checks. The script averages class probability vectors, never predicted labels. Results below are post-hoc diagnostics on the already reused validation split; they do not provide independent generalization evidence.

| Fixed combination | Accuracy | Macro-F1 | Balanced accuracy | Correct |
|---|---:|---:|---:|---:|
| V2 50% + V3 50% | 0.82772586 | 0.70692666 | 0.71699379 | 2657/3210 |
| V2 25% + V3 75% | 0.83115265 | 0.70722528 | 0.70853840 | 2668/3210 |

Neither combination exceeds V3 alone on accuracy or macro-F1; the 50/50 blend has higher balanced accuracy than V3 but lower accuracy and macro-F1. Keep V3 as the standalone comparator. Do not promote an ensemble from these post-hoc results. Fresh hflip and 8-view dihedral predictions are still required from the PC; use the same model SHA, manifest, row order and preprocessing contract.

Relative to baseline, V2 changes accuracy by −0.01744548, macro-F1 by −0.01039465 and balanced accuracy by +0.01910256. V2 therefore trades overall accuracy/macro-F1 for higher balanced accuracy. It is not a net improvement under the stated macro-F1 objective without per-class evidence.

V3 changes accuracy by +0.00436137, macro-F1 by +0.00726706 and balanced accuracy by +0.00473921. V3 is the best supplied aggregate result on this fixed validation split. This is a modest improvement, not evidence that 88–90% accuracy is guaranteed.

Reaching 88% on 3,210 rows requires at least 2,825 correct (143 more than V3); 90% requires at least 2,889 (207 more). These are thresholds, not forecasts.

## Evidence limitations and next action

The archive supplies evaluation logs and run info but not epoch-by-epoch training curves; therefore it verifies reported epoch-6 model selection/validation outcomes, not overfit dynamics over training.

`scripts/run_evaluate_b3_variants.ps1` is prepared to evaluate baseline, V2 and V3 on the same fixed 3,210 validation rows with model-specific expected checkpoint SHA-256 and the frozen manifest hash. `scripts/compare_model_predictions.py` verifies identical image/label order, produces per-class scores and paired correct↔wrong transitions. Full GPU evaluation has not been run from the laptop; PC run is pending.

Keep the Streamlit default and all three checkpoints unchanged. Validation images come from the same Kaggle source and have influenced checkpoint/experiment selection; this is not an independent field test. See [Phase 3F GPU plan](../docs/PHASE3F_GPU_PLAN.md) for the locked TTA and controlled training sequence.
