# Verification status — 28 October 2026

The repository now contains the Phase 1–3B data reports, frozen manifests, Cassava app, and reproducible training scripts. Source images remain outside the repository at `C:\Users\Admin\Downloads\data\extracted`; model weights and virtual environments are local and ignored by Git.

| Crop | Checkpoint status | Dataset/scope | Locally observed metrics |
|---|---|---|---|
| Cassava | Published MobileNetV3-Small verified; new ImageNet-initialized MobileNetV3-Small trained and tested | 14,978 train / 3,210 val / 3,209 internal test | New model internal test: accuracy 0.8180, macro-F1 0.7011, balanced accuracy 0.7039; same-source Kaggle, no external validation |
| Rice | No compatible 10-class checkpoint integrated | Frozen manifest; near-duplicate review pending | None |
| Coffee | No compatible 6-class checkpoint integrated | Frozen plant-group manifest; label disputes pending | None |

The Cassava Streamlit app defaults to the published checkpoint and offers the new model as an experimental option. The published checkpoint's 3,210-image validation diagnostic (accuracy 0.8458, macro-F1 0.7688) may be contaminated because it trained on the same Kaggle source. Do not compare it with the new model's internal test as if both were independent field evaluations.

See `reports/archive/11_cassava_baseline_evaluation.md`, `reports/archive/12_cassava_finetuning.md`, and `reports/archive/13_cassava_comparison.md`. The 85% internal-test accuracy target was not reached. A separately sourced, expert-labeled field test remains necessary to assess real-world performance.
