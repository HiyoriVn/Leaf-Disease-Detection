# LeafSense – UI, debug, and disease advice

## Unified Streamlit interface

The sidebar contains a single crop selector. The Rice screen shows input/model
status, a 10-class horizontal confidence chart, exact score table, and a collapsible
technical JSON report. Crop advice is stored in the independent disease_advise/
folder and linked by **exact** checkpoint label name.

The theme is explicitly light in .streamlit/config.toml to avoid white-on-white
text in browser dark mode. The Rice heading uses a rice ear (🌾), not a leaf.

## Windows run commands

    git switch hehe
    git pull --ff-only origin hehe
    python -m pip install -r requirements-app.txt
    python -m pip install -r requirements-rice.txt
    python -m streamlit run app.py

Restart Streamlit after changing .streamlit/config.toml (Ctrl+C and rerun).
If a previous session persists, reload the browser with Ctrl+F5.

## What "Tệp model = Đã tìm thấy" really means

Only file existence. "Dung lượng = Khớp" checks bytes against rice_app.json.
Click Analyze to load TFLite, validate input/output tensor metadata, preprocess,
invoke and return ten scores. The debug expander reports SHA-256, tensor shapes,
pixel range, latency, OOD gate state, and prediction scores. If a stage fails,
the report records the error instead of silently producing a fake result.

Independent smoke command:

    python scripts/diagnose_rice_model.py --image "D:\pictures\rice.jpg"

This writes reports/rice_smoke_local.json. Without --image it invokes on a synthetic
green image **only to confirm that the runtime runs**, not as a valid diagnosis.

## OOD handling

The model does not contain an independently validated OOD detector. An existing
handcrafted rule rejects some obvious synthetic logo images. If no calibrated
gate exists, the UI now **still allows experimental inference**, explicitly
marks "Unverified" and does not claim the picture contains rice. Rejected
images remain blocked. For a production gate, follow docs/RICE_INTEGRATION.md,
train on real rice + representative non-rice examples, then validate externally.

## Disease advice

- disease_advise/rice.md (10 checkpoint labels)
- disease_advise/cassava.md (5 checkpoint labels)
- disease_advise/coffee.md (4 checkpoint labels)

Advice does not depend on probability thresholds and is educational, not a
chemical prescription. Every section contains crop practice, nutrient decision
and appropriate phytosanitary cautions plus source URLs. The 3 files do not
contain locally licensed pesticide products or dosage, because country, region,
diagnosis, phenology and products have not been verified for this user.

## Still open research work

P0: notebook preprocessing/Keras–TFLite parity has not been independently
confirmed because original training/validation images were not available.
P1: no trained, externally validated OOD detector currently ships.
P2: no completed external OOD audit or validation-fitted temperature artifact.

The test suite for advice checks label completeness but is not a substitute
for field validation or a live Windows app run.
