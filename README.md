# Cassava and Coffee Leaf Disease Detection

Streamlit demo for image classification of cassava and coffee leaves. The app supports image upload and camera capture, and reports model scores as predictions rather than confirmed agricultural diagnoses.

## Run the app

Requirements: Windows, Python 3.13. From the repository folder:

```powershell
py -3.13 -m venv .venv-phase2
.\.venv-phase2\Scripts\python.exe -m pip install -r requirements-app.txt
.\.venv-phase2\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Open <http://127.0.0.1:8501>.

The app only offers checkpoints that are present and match the SHA-256 recorded in the model configuration. Checkpoints are excluded from Git. See [Cassava app guide](docs/CASSAVA_APP_GUIDE.md) for obtaining and installing Cassava weights.

Coffee requires the separately supplied ResNet18 checkpoint. Set its location before starting Streamlit:

```powershell
$env:COFFEE_MODEL_PATH = 'C:\path\to\coffee_model.pth'
.\.venv-phase2\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

The Coffee checkpoint hash and four class labels are checked against [the model registry](configs/plant_model_registry.json). Its redistribution permission is unknown, so it is not included in this repository or a release.

## Model status

| Model | Local status | Evaluation status |
|---|---|---|
| Cassava MobileNetV3 published | Checkpoint verified; default | Author-reported metrics, same-source data |
| Cassava MobileNetV3 fine-tuned | Checkpoint verified | Internal test: 81.80% accuracy, 0.7011 macro-F1; same-source Kaggle |
| Cassava EfficientNet-B3 baseline | Checkpoint verified; optional | Validation: 83.12% accuracy, 0.7052 macro-F1; not external field data |
| Cassava EfficientNet-B3 V3 | Not present in checkout | Validation artifact: 83.55% accuracy, 0.71245 macro-F1; inference unavailable without checkpoint |
| Coffee ResNet18 | External file required | No verified accuracy metric |
| Rice | Not integrated | No compatible verified checkpoint |

Model and data limitations are detailed in [Phase 4 integration notes](reports/21_phase4_integration.md). No checkpoint is currently cleared for public redistribution; dataset and derived-weight terms need confirmation before attaching weights to a GitHub Release.

## Development

Install app dependencies with `python -m pip install -r requirements-app.txt`. Run the integration suite with:

```powershell
.\.venv-phase2\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp\phase4
```

Training dependencies and the Windows GPU workflow are documented in [Windows GPU training](docs/WINDOWS_GPU_TRAINING.md). Cassava app details are in [the app guide](docs/CASSAVA_APP_GUIDE.md); the older TorchScript contract is described in [the model and data contract](docs/MODEL_AND_DATA_CONTRACT.md).

## Research archive

Historical phase plans, dataset audits, training results, and validation reports are retained under [`reports/archive/`](reports/archive/). Superseded operational notes are under [`docs/archive/`](docs/archive/). Start with the current app and training guides above.
