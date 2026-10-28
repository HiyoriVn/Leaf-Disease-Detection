# Capstone defense preparation

## Suggested narrative

1. **Problem:** plant disease identification from leaf photographs can support early screening, but image variation and class imbalance complicate automated classification.
2. **Scope:** separate CNN classifiers for rice, coffee, and cassava; user selects the crop and submits a leaf image.
3. **System:** Streamlit upload and crop selection → crop-specific checkpoint → shared RGB/resize/normalization pipeline → indexed class mapping → top-3 output.
4. **Evaluation:** report a held-out dataset per crop with sample counts, accuracy, macro-F1, and confusion matrix. Add class-level analysis and error examples from actual outputs.
5. **Limitations:** checkpoint provenance and dataset domain determine validity; softmax confidence is not calibrated; field diagnosis requires expert review.
6. **Next steps:** verify artifacts, evaluate held-out splits, test representative uploads, document licenses, and consider calibration and external validation.

## Results slide template

Populate only after running `python -m src.evaluate --data DATASET --output results`:

| Crop | Dataset/split | N | Accuracy | Macro-F1 | Evidence |
|---|---|---:|---:|---:|---|
| Rice | Pending | — | — | — | `results/evaluation.json` |
| Coffee | Pending | — | — | — | `results/evaluation.json` |
| Cassava | Pending | — | — | — | `results/evaluation.json` |

Published metrics belong on a separate slide with a citation and the source paper's dataset/split. They are not a comparison unless label sets, evaluation protocol, and dataset domain are comparable.

## Demo checklist

- Run the app in the same environment used for the rehearsal.
- Verify all three checkpoint files and class names against training output.
- Demonstrate one valid image for each crop and an invalid or unsupported image.
- Explain preprocessing and label order; show measured results and confusion matrices only after evaluation.
- Keep a recorded demo and screenshots available in case the room's network or machine fails.
