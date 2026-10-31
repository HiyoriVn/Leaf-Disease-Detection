# Cassava and Coffee Leaf Disease Detection

A Streamlit demo for Cassava and Coffee leaf image classification. Predictions are decision support and are not confirmed agricultural diagnoses.

## Clone and install

Requirements: Windows and Python 3.13. From a clean clone:

```powershell
git clone https://github.com/HiyoriVn/Leaf-Disease-Detection.git
cd Leaf-Disease-Detection
py -3.13 -m venv .venv-phase2
.\.venv-phase2\Scripts\python.exe -m pip install -r requirements-app.txt
```

## Checkpoint availability

The current [Phase 4 source release](https://github.com/HiyoriVn/Leaf-Disease-Detection/releases/tag/phase4-integration) contains source archives only. **It has no downloadable model weights.** The local candidate checkpoints below were hash and inference checked, but they are not uploaded because redistribution rights are not documented.

| Demo model | Expected local path | SHA-256 | Status |
|---|---|---|---|
| Team fine-tuned Cassava MobileNetV3-Small | `checkpoints/experiments/full_backbone_smooth_20261028/best.pth` | `7cb54b7de90ec775eb346a0a5ade31dfcdac7fff2e03db5264c1aa8c6fc98cd0` | Locally verified; public release blocked pending Kaggle data and derived-weight permission review |
| Coffee ResNet18 | `models/coffee_model.pth`, or path in `COFFEE_MODEL_PATH` | `12f958c9c08b0cc0b13311e86a23f1089c3e3476a8984a87db79cfd78f884900` | Locally verified; public release blocked because checkpoint source, training data, and redistribution terms are undocumented |

`checksums.sha256` records the candidate file hashes for later verification. It does not indicate that these files are downloadable release assets. Published Cassava weights and EfficientNet-B3 baseline are also present locally, but are not selected as release assets; their data and derived-weight terms have not been confirmed. The missing EfficientNet-B3 V3 checkpoint is out of demo scope; its research configuration and findings remain archived.

## Install approved checkpoints

After obtaining an approved checkpoint from a future model release or another authorized source, copy the Cassava file to the exact path in the table above. Put Coffee at `models/coffee_model.pth` or set its external path:

```powershell
$env:COFFEE_MODEL_PATH = 'C:\path\to\coffee_model.pth'
```

Verify candidate hashes before running:

```powershell
Get-FileHash .\checkpoints\experiments\full_backbone_smooth_20261028\best.pth -Algorithm SHA256
Get-FileHash .\models\coffee_model.pth -Algorithm SHA256
```

The app validates checkpoint SHA-256, architecture, preprocessing, and class order using `configs/plant_model_registry.json` and the model-specific configs. Cassava uses `timm mobilenetv3_small_100`, RGB, OpenCV `INTER_AREA` resize 256, center crop 224, and ImageNet mean/std. Its five output labels are CBB, CBSD, CGM (Cassava Green Mottle), CMD, and Healthy. Coffee uses torchvision ResNet18, RGB resize 224×224, ImageNet mean/std, and its checkpoint's four-class order: `Cerscospora`, `Healthy`, `Leaf rust`, `Phoma`.

## Run and test

```powershell
.\.venv-phase2\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Open <http://127.0.0.1:8501>. To run the integration suite:

```powershell
.\.venv-phase2\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp\phase4
```

See [Cassava app guide](docs/CASSAVA_APP_GUIDE.md), [Windows GPU training](docs/WINDOWS_GPU_TRAINING.md), [Phase 4 integration notes](reports/21_phase4_integration.md), and [model release audit](reports/22_model_release_audit.md). Historical experiments remain under [`reports/archive/`](reports/archive/).

## Deploy on Streamlit Community Cloud

1. Open [share.streamlit.io](https://share.streamlit.io) and create an app from this GitHub repository.
2. Select branch `codex/streamlit-deploy` and entrypoint `app.py`.
3. Streamlit installs the app dependencies from `requirements.txt` and provides a public `streamlit.app` link to share with the class.

This branch includes the Coffee ResNet18 checkpoint in `models/coffee_model.pth` and the Cassava checkpoints tracked on the current `main` branch. The app checks the configured SHA-256 and class mapping before offering a model. Rice inference is a demo only: preprocessing remains unverified and the quality gate is uncalibrated. Do not treat Rice predictions as validated diagnoses. The Phase 4 GitHub Release remains source-only; Coffee and Cassava checkpoint files are in the repository branch.
