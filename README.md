# Plant Disease Classification Capstone

Streamlit inference and evaluation scaffold for rice, coffee, and cassava leaf disease classifiers. The source images are stored outside this repository at `C:\Users\Admin\Downloads\data\extracted`. Phase 1 audited the archives and images; Phase 2 created manifests and ran a technical smoke test on one published cassava checkpoint. The app still requires **TorchScript** checkpoints and populated `models/manifest.json`; the published checkpoint downloaded for Phase 2 is a PyTorch state dictionary and is not wired into the app.

## Phase 2 artifacts

- [Model feasibility](reports/07_model_feasibility.md), [preprocessing protocol](reports/08_preprocessing_protocol.md), and [experiment design](reports/09_experiment_design.md).
- `configs/model_registry.json` records candidate architectures, label contracts, licenses, and overlap risks.
- `configs/dataset_config.yaml` fixes data root, seed, split fractions, and pHash thresholds. Edit `data_root` on another machine.
- `manifests/{rice,cassava,coffee}.csv` list image paths and SHA-256 without copying images. Coffee dispute and exact-duplicate queues, and rice pHash candidates, are also in `manifests/`.
- `reports/smoke_cassava/` contains five predictions and a JSON log from a real checkpoint inference. These are **not** independent evaluation metrics.

To reproduce the manifests after editing the config:

```powershell
python scripts/prepare_manifests.py --config configs/dataset_config.yaml
python scripts/find_near_duplicates.py --config configs/dataset_config.yaml --dataset rice --apply-groups
```

For the optional cassava smoke test, create a separate Python 3.13 virtual environment and install `torch==2.8.0`, `torchvision==0.23.0`, `timm==1.0.20`, and `opencv-python-headless==4.12.0.88`. Download `mobilenetv3_best.pth` from the [author's Space](https://huggingface.co/spaces/ctibenkana/cassava-disease-classifier/tree/main) into `checkpoints/cassava_mobilenetv3_best.pth`, then run:

```powershell
.\.venv-phase2\Scripts\python.exe scripts\smoke_test_cassava.py --checkpoint checkpoints\cassava_mobilenetv3_best.pth --manifest manifests\cassava.csv --data-root 'C:\Users\Admin\Downloads\data\extracted' --output-dir reports\smoke_cassava
```

The author used older library versions; the local smoke log records the versions actually run. `checkpoints/` and `.venv-phase2/` are ignored by Git.

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

This repository does not commit source images or model weights. The Phase 2 cassava checkpoint was tested only for loading and inference, and the three manifests contain internally derived splits. No Phase 3 fine-tuning or scientific evaluation has run. See `reports/07_model_feasibility.md` and `docs/STATUS.md` for scope.
