# Phase 4 — Repository integration and release readiness

Updated: 2026-10-30

## Repository and teammate changes

- Local `main` is clean at start of this work and points to `c6e6668` (`Add coffee leaf disease detection`), the same commit as `origin/main`.
- The latest commit is by `HiyoriVn` and contains `.gitignore`, README, `app.py`, `src/coffee_inference.py`, and Coffee train/evaluate scripts.
- PR #1 still appears open on GitHub at head `02b0417`, based on `0989992`. Its Coffee changes are already present on `main` in commit `c6e6668`; do not merge or cherry-pick that stale branch again. The PR should be reconciled/closed by its owner after checking its current branch state.
- No pre-existing uncommitted Cassava work was present at audit time. Dataset ZIPs are under the sibling `Downloads/data` directory, not inside this Git worktree. `.gitignore` excludes local `dataset/`, `checkpoints/`, training environments and temporary outputs; `models/coffee_model.pth` is specifically ignored.
- GitHub currently has no Releases for this repository. The repository's `LICENSE` covers the source project, but it does not establish redistribution rights for every checkpoint or source dataset.

## Integration delivered

- Added `configs/plant_model_registry.json` as the demo model catalog and artifact manifest. Entries record model IDs, crops, architectures, class order, checkpoint path/name, SHA-256, size when locally known, preprocessing, pretrained source, metrics provenance, evaluation limits and license status.
- Added `src/model_registry.py`. The app now lists only enabled models whose checkpoint exists and whose SHA-256 matches the registry/config.
- Left legacy `models/manifest.json`/`src/inference.py` untouched. That TorchScript contract has empty label arrays and no matching `.pt` artifacts; it is not the Streamlit model registry and does not prove Rice/Coffee inference availability.
- Added `configs/cassava_efficientnet_b3_v3_app.json` for the verified V3 contract. The V3 checkpoint is not present locally, so it is not offered in the UI. The present B3 checkpoint is the baseline, with SHA-256 `7924296913c5f3a6c9d822212cdb0a8326b6d5c4cd2778078d6c47bebd0d406b`; it remains an optional model and the published MobileNet remains first/default.
- Coffee uses its own ResNet18 adapter, checkpoint class mapping, 224×224 RGB/ImageNet preprocessing, hash check and `COFFEE_MODEL_PATH` override. The supplied checkpoint's class order is `Cerscospora`, `Healthy`, `Leaf rust`, `Phoma`.
- The app offers upload or camera input for each available crop. It retains separate Cassava and Coffee preprocessing and inference code. Rice remains unavailable because there is no verified compatible checkpoint/adapter.
- Aligned the broad Pillow dependency upper bound with the app environment and updated README setup/artifact notes.

## Checkpoint audit

| Model | Local status | SHA-256 | Evidence / limitation |
|---|---|---|---|
| Cassava published MobileNetV3-Small | Present, 6,224,097 bytes | `4043a480d7a48d636f4feefbe31df4c52eeadd8edb8404d31319eddd807ac4bb` | Strict-load and inference smoke passed; no independent external metric claim. |
| Cassava team MobileNetV3-Small | Present, 6,201,897 bytes | `7cb54b7de90ec775eb346a0a5ade31dfcdac7fff2e03db5264c1aa8c6fc98cd0` | Strict-load and inference smoke passed; internal-test metrics remain same-source Kaggle results. |
| Cassava EfficientNet-B3 baseline | Present, 43,308,023 bytes | `7924296913c5f3a6c9d822212cdb0a8326b6d5c4cd2778078d6c47bebd0d406b` | Strict-load and inference smoke passed; 3,210-image same-source validation report is recorded in report 16. |
| Cassava EfficientNet-B3 V3 | Missing | Expected `0e76a24e1d979ee29f508b83fe6059f89f7a13a167effcaddc3eacfea0cf669c` | Validation predictions/report support accuracy 83.5514%, macro-F1 0.71245032, balanced accuracy 0.70484034 on 3,210 same-source validation images. Checkpoint absent; no app inference test. |
| Coffee ResNet18 | Present only at the user's external `Downloads/coffee_model.pth`, 44,792,523 bytes | `12f958c9c08b0cc0b13311e86a23f1089c3e3476a8984a87db79cfd78f884900` | Safe load, class mapping and inference smoke passed. No verified accuracy metric; current checkpoint predates split metadata. |
| Rice | Missing | — | No model offered. |

Smoke checks used one actual Cassava training image and one RoCoLe coffee image read directly from the supplied ZIP archives in memory, without extracting or changing dataset files. All three available Cassava models and Coffee returned valid top-three predictions. This is a pipeline smoke test, not an accuracy evaluation. Coffee weights and their training-data provenance/license are not documented, so the Coffee checkpoint is not ready for public redistribution.

## Artifact and GitHub plan

- Keep datasets, pretrained initialization weights, checkpoints, and training outputs out of normal commits. Current `.gitignore` does this; the audit found no model/data files tracked or staged.
- Prefer GitHub Release assets over Git LFS for the small, versioned demo checkpoints. Release assets are outside Git history and each is below GitHub's documented 2 GiB per-file limit. The release asset should be downloaded into its registry path and checked against SHA-256 before use.
- Candidate assets after provenance/license approval: published Cassava MobileNet, team MobileNet, B3 baseline, and Coffee ResNet18. The V3 asset cannot be prepared until its file is supplied. Do not release pilot/`last.pth`, datasets, pretrained initialization files, or Coffee weights before rights are confirmed.
- No commit was created, no push was attempted, no release/tag was created, and no PR state was changed. Those remote actions require the user's final approval.

The [official Cassava competition data page](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/data) says use is subject to competition rules; the redistribution terms for the project's fine-tuned weights still need a specific review. The [RoCoLe article](https://www.sciencedirect.com/science/article/pii/S2352340919307693) describes an open-access dataset under a Creative Commons license, but the supplied Coffee checkpoint has no provenance tying it to a particular RoCoLe version/license or documenting its own terms. Do not treat the project MIT source license or dataset availability as blanket permission to publish all weights.

## Verification

- `py_compile`: app, Coffee adapter, registry, Coffee training/evaluation scripts passed.
- Full tests: `30 passed`.
- Checkpoint inference smoke: all three available Cassava checkpoints and the supplied Coffee checkpoint loaded and returned three top predictions.
- Missing Coffee checkpoint path is rejected with `CoffeeInferenceError`; modified hashes are filtered by registry and the loader checks the configured expected hash/class mapping.
- Streamlit server started on `127.0.0.1:8501` and its `/_stcore/health` endpoint returned HTTP 200. Streamlit AppTest switched Cassava → Coffee, uploaded one actual image from each supplied ZIP, rendered results with zero app exceptions, and verified the missing-Coffee-checkpoint warning. This is app-level UI harness coverage, not a manual browser/real-camera rehearsal.
- No GPU training, full validation, Phase 3F/3G, TTA, data extraction or manifest modification was performed.

## Launch

```powershell
cd C:\Users\Admin\Downloads\Leaf-Disease-Detection
\.venv-phase2\Scripts\Activate.ps1
$env:COFFEE_MODEL_PATH = 'C:\Users\Admin\Downloads\coffee_model.pth'
python -m streamlit run app.py --server.address 127.0.0.1
```

For another computer, install `requirements-app.txt`, provide the approved checkpoints, and set `COFFEE_MODEL_PATH` to its Coffee checkpoint. The published Cassava model can be obtained from the source linked in `docs/CASSAVA_APP_GUIDE.md`; other checkpoints need an approved release or private transfer until a release is available.
