# Plant Disease Classification Capstone

Streamlit inference and evaluation scaffold for rice, coffee, and cassava leaf disease classifiers. This checkout was found empty: no checkpoints, class mappings, training code, or labeled datasets were present. The application therefore refuses to invent models or metrics and reports missing artifacts clearly.

## Quick start

1. Create an environment with Python 3.10–3.12 and install `pip install -r requirements.txt`.
2. Place each compatible TorchScript checkpoint at `models/{rice,coffee,cassava}.pt` and edit `models/manifest.json` to match its class labels, image size, and normalization. See `docs/MODEL_AND_DATA_CONTRACT.md`.
3. Run `streamlit run app.py`.
4. To evaluate a labeled folder laid out as `DATASET/crop/class_name/image.jpg`, run `python -m src.evaluate --data DATASET --output results`.

No metric is reported until evaluation actually runs on labeled data. Published metrics must be cited separately and must not be represented as local experiments.

## Project map

- `app.py`: crop selection, image upload, prediction, top three scores, input/model errors.
- `src/inference.py`: strict checkpoint loading and shared preprocessing/inference.
- `src/evaluate.py`: accuracy, macro-F1, confusion matrix, per-crop results.
- `models/manifest.json`: class-index and preprocessing contract.
- `docs/`: model and dataset setup, technical notes, and defense outline.
- `tests/`: artifact-free checks for manifest and preprocessing/error behavior.

## Important

The supplied folder contained no model checkpoints or datasets. This repository does not include pretrained weights. Add the original project artifacts before claiming model verification or scientific performance. See `docs/STATUS.md`.
