# Rice LiSA integration — branch hehe

## Unified UI

Run from repository root:

```powershell
python -m pip install -r requirements-app.txt
python -m pip install -r requirements-rice.txt
python -m streamlit run app.py
```

Only ONE Streamlit page is shown. The crop selector has Rice, Cassava, Coffee.
The old Streamlit multipage file pages/03_Rice.py was removed and the Rice
panel is now src/rice_page.py. Existing Cassava/Coffee inference modules
remain untouched. Their imports are lazy, so Rice alone does not import timm.

## P0: preprocessing verification — REQUIRED

Current config: configs/rice_app.json has pixel_scale=0_255 and
preprocessing_verified=false. This has NOT been established as the real
training pipeline. Do not flip the flag by assumption.

Download/extract the ORIGINAL Paddy Doctor train_images to a folder with
10 class subfolders. For example:
```powershell
python -m pip install tensorflow==2.20.0
python scripts/verify_rice_inference.py --dataset-root "D:\paddy\train_images" --samples 30
```

The script checks RGB, TFLite shapes and six candidate resize/scale contracts
by comparing TFLite with Keras and SAVED notebook test predictions.
Examine reports/rice_contract_local.json, inspect the notebook training
loader, then set configs/rice_app.json with the matching transform.
If no candidate passes, do not claim successful verification. The scripted
search is diagnostic, not a substitute for reading the notebook.

## P1: RGB image validity / OOD rejection

There is NO trained OOD classifier artifact in the repo at present.
A preliminary rule can reject certain obvious red logos on white backgrounds.
That rule does NOT validate that other images contain rice leaves.

Without models/rice_gate.json, the default application refuses to issue a
disease classification. Research-only checkbox can bypass this unverified
state, but cannot bypass an explicitly rejected image.

To train a gate, prepare separate folders of rice-leaf photos and varied OOD
photos. Do not reuse near-duplicate images across train/validation/test.
```powershell
python -m pip install scikit-learn
python scripts/train_rice_gate.py --rice-dir "D:\data\rice_train" --other-dir "D:\data\ood_train"
```

This generates models/rice_gate.json with a transparent logistic regression
baseline over color/texture features and a held-out validation threshold.
It must be externally evaluated; this does NOT constitute a production
grade plant detector. Better future baseline: trained plant/rice detector
plus embedding-based OOD scoring.

## P2: OOD and calibration evaluation

Test on completely unseen rice and non-rice photos:
```powershell
python scripts/evaluate_rice_ood.py --rice-dir "D:\data\rice_test" --other-dir "D:\data\ood_test"
```

Record rice rejection rate, OOD false acceptance rate, and counts. Check
examples of logos, objects, people, other leaves, rice in field backgrounds.
The gate may fail badly on real domain shift; do not quote validation AUROC
as guaranteed real-world OOD performance.

Softmax calibration uses validation prediction CSV for temperature fitting,
test prediction CSV only for measurement:
```powershell
python scripts/calibrate_rice_confidence.py --val-csv "D:\data\val_predictions.csv" --test-csv "Rice-leaf\riceLeafDisease_trainingResult\test_predictions.csv"
```

The repository already has test_predictions.csv but does not have
val_predictions.csv. Produce the missing validation predictions using the
verified inference pipeline and the frozen val_split.csv. Never fit the
temperature on the test split.

## Limitations

The current training notebook is ~15 MB and the connector couldn't inspect
its full contents. The assistant has not executed Keras/TFLite parity against
original images or trained/evaluated the OOD gate. Thus P0/P1/P2 have working
verification/evaluation tooling, not independently validated results.
Do not call high Softmax confidence a diagnosis.
