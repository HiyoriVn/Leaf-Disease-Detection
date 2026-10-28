# Model and dataset contract

## Checkpoint contract

The baseline intentionally accepts TorchScript files only. Put one checkpoint per crop in `models/` and configure `models/manifest.json`. Each entry must provide a checkpoint filename, positive image size, three-channel mean/std, and class names in the exact output-index order used during training. Input handling applies EXIF orientation, RGB conversion, square bilinear resize, `[0,1]` scaling, channel normalization, and a batch dimension. The output must be logits shaped `[1, classes]`; softmax is applied for display.

This contract cannot establish that the checkpoint is genuinely pretrained, trained for the stated crop, or paired with the correct labels. Record provenance, architecture, source, training split, license, and checksum before defense. Do not rename an unknown checkpoint or guess its label order. A plain `state_dict` requires its original architecture definition and is not compatible with this loader as-is.

## Evaluation layout

Provide a held-out labeled folder such as:

```
DATASET/
  rice/<class-name>/*.{jpg,png}
  coffee/<class-name>/*.{jpg,png}
  cassava/<class-name>/*.{jpg,png}
```

Folder names must match model labels exactly. Evaluation reports per-crop sample count, accuracy, macro-F1, and a confusion matrix whose rows/columns follow manifest order. Use a held-out split independent of training and checkpoint selection. Preserve dataset version, source, license, split procedure, class counts, and a data checksum alongside generated `results/evaluation.json`.

## Interpretation

Confidence is the model's softmax score, not a calibrated probability or diagnosis certainty. Class imbalance is why macro-F1 is reported alongside accuracy. Results on a single curated dataset do not demonstrate field performance or cross-region generalization.
