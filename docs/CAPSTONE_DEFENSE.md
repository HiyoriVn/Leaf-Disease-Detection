# Capstone defense preparation

## Suggested narrative

1. **Problem:** plant disease identification from leaf photographs can support early screening, but image variation and class imbalance complicate automated classification.
2. **Scope:** verified demo classifiers for coffee and cassava; Rice is future work until a compatible checkpoint and adapter are validated.
3. **System:** Streamlit crop/model selection → crop-specific checkpoint and preprocessing → checkpoint-specific class mapping → top-3 output.
4. **Evaluation:** report a held-out dataset per crop with sample counts, accuracy, macro-F1, and confusion matrix. Add class-level analysis and error examples from actual outputs.
5. **Limitations:** checkpoint provenance and dataset domain determine validity; softmax confidence is not calibrated; field diagnosis requires expert review.
6. **Next steps:** verify artifacts, evaluate held-out splits, test representative uploads, document licenses, and consider calibration and external validation.

## Results slide template

Populate only after running `python -m src.evaluate --data DATASET --output results`:

| Crop | Dataset/split | N | Accuracy | Macro-F1 | Evidence |
|---|---|---:|---:|---:|---|
| Rice | Pending | — | — | — | `results/evaluation.json` |
| Coffee | Not evaluated | — | — | — | no verified evaluation artifact |
| Cassava | Source-specific reports | 3,210 validation for B3 | See reports 16/19 | See reports 16/19 | same-source validation, not field evaluation |

Published metrics belong on a separate slide with a citation and the source paper's dataset/split. They are not a comparison unless label sets, evaluation protocol, and dataset domain are comparable.

## Demo checklist

- Run the app in the same environment used for the rehearsal.
- Verify all available checkpoint hashes and class names against the registry.
- Demonstrate valid Cassava and Coffee uploads and an invalid or unsupported image. Do not present a live UI demo as verified until it has been rehearsed in a normal local browser.
- Explain preprocessing and label order; show measured results and confusion matrices only after evaluation.
- Keep a recorded demo and screenshots available in case the room's network or machine fails.
