# 18 — Phase 3E controlled optimization plan (29/10/2026)

## Evidence motivating the experiments

The B3 baseline scored 0.8312 accuracy / 0.7052 macro-F1 / 0.7001 balanced accuracy on the fixed validation split. Its minority-class F1 values are CBB 0.5531 and Healthy 0.6483; CMD is 0.9342 with 1,974/3,210 validation images. The larger directed confusions are CGM→CMD (70/358), CBB→Healthy (39/163), and Healthy→CBB (42/387). This pattern motivates modest minority-sensitive loss experiments, while the current accuracy/macro-F1 gap cautions against letting the majority class dominate.

Training curve evidence is incomplete: the returned archive contains no B3 training logs. Thus overfitting cannot yet be quantified from train-vs-validation curves. The PC bundle instructs each run to retain `run.json`, `epochs.jsonl`, console log, and best/last checkpoints so Phase 3E can measure the gap and stopping behavior.

## Frozen controls

- Same manifest SHA-256 `570b0d52ea950dec13de0eed792614b67c039b08b2cc28c8e588bb87b2f61082`, same train/validation rows and label order.
- Same ImageNet B3 weights SHA-256 `279d2a53898aa89dab43fd6bd7df9f706aea4cb9cf916223988bbcb8e5850469`, 300 input, resize 343, cubic interpolation, ImageNet normalization, seed, batch 16, AMP, optimizer schedule and early stopping.
- No disease-specific B3 checkpoint initialization. Never inspect/use internal-test predictions for hyperparameter/model selection.
- Keep `cassava-b3-fullest.pth` and all pre-existing checkpoints unchanged. Every run gets its own `RunName` directory.

## Prepared experiments; not yet run

| Order | Run | Single change from baseline | Purpose |
|---|---|---|---|
| 1 | `cassava-b3-v2-weighted` | Weighted CE with tempered inverse-frequency class weights, power 0.5; no label smoothing | Test whether modest reweighting lifts CBB/Healthy/CBSD/CGM recall without severe CMD precision loss |
| 2 | `cassava-b3-v3-smoothing` | Standard CE plus label smoothing 0.05; no reweighting | Test regularization/calibration effects separately from class weighting |

Both configurations retain the baseline head/final-two-block schedule, validation macro-F1 checkpoint selection and patience 3. Weighted loss uses `(N/(K*n_class))**0.5`; this is deliberately less aggressive than the inverse-frequency exponent 1.0. The default value 1.0 preserves prior trainer behavior. The trainer now records the weight power in `run.json`.

Start each run with the 32-train/16-validation-per-class pilot on the PC only. The pilot is a pipeline/hardware check, not a model comparison. After successful pilot, run the corresponding full experiment. Do not change batch size, seed, schedule, or other parameters mid-run; use a new run name and document any changed variable. If GPU memory is insufficient, restart that experiment in a new run name at batch 8 and report it as a deviation.

## Selection and stopping criteria

Use validation macro-F1 as the sole checkpoint selection criterion, with per-class recall and balanced accuracy as diagnostics. A candidate should not be considered a clear improvement based only on a tiny metric difference. Compare repeated runs only if time permits, and report seed variability. Review the confusion pairs and representative image paths without changing labels. If weighted loss improves macro-F1 but causes unacceptable CMD precision/recall loss, retain the baseline.

Once Phase 3E runs finish, compare baseline and candidates on the same fixed validation rows. Choose one model/config, freeze it, and only then run the internal-test evaluator exactly once for final reporting. This remains same-source evaluation, not a field/generalization test. No Phase 3E train, pilot or validation has been run as part of preparing this plan.
