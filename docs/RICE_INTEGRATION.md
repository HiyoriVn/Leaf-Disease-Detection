# Rice LiSA — add-only Streamlit integration

Existing `app.py`, cassava and coffee source modules, registry, and dependencies remain unchanged. `pages/03_Rice.py` is auto-discovered by Streamlit's built-in multipage mechanism when the existing `app.py` is launched.

## Run (from repository root)

```powershell
pip install -r requirements-app.txt
pip install -r requirements-rice.txt
python -m streamlit run app.py
```

Open the new **Rice** page in Streamlit sidebar. The old page retains its existing Cassava/Coffee selector. The LiSA model is loaded from `Rice-leaf/riceLeafDisease_trainingResult/lisa_paddy_fp32.tflite` and class indices are taken **without sorting** from `class_names.json`.

## Important: preprocessing contract not verified yet

`configs/rice_app.json` sets `pixel_scale: "0_255"`, direct bilinear resize to 224x224, and `preprocessing_verified: false` as an **explicit provisional configuration**, not as an inferred fact from the notebook. Do not use these predictions for performance reporting until all of the following pass:

1. Inspect the notebook's validation/test loader to confirm RGB, EXIF handling, resize/interpolation, and pixel range. Check whether rescaling is embedded in the Keras graph.
2. Confirm TFLite model input tensor shape/dtype `[1, 224, 224, 3] / float32` and ten output scores.
3. Compare a set of labeled held-out images across Keras and TFLite with exactly the same preprocessing. Check top-1 agreement and max absolute probability delta; record thresholds and results.
4. Only after successful verification update the **new file** `configs/rice_app.json` to set `preprocessing_verified: true` and the correct `pixel_scale`.

Existing tests can be run using `python -m pytest -q tests/test_rice_inference.py`; those tests check the adapter's shapes and class mapping but **do not validate Keras/TFLite equivalence**. No existing application files are modified.
