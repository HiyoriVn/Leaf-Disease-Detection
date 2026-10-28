"""Streamlit interface for rice, coffee, and cassava leaf classifiers."""
from __future__ import annotations

import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.inference import ModelError, load_manifest, load_model, predict

st.set_page_config(page_title="Leaf Disease Classifier", page_icon="🌿", layout="centered")
st.title("🌿 Plant Disease Classification")
st.caption("CNN assistance for rice, coffee, and cassava leaf images. Predictions are educational and should be reviewed by an agricultural expert.")

try:
    configs = load_manifest()
except ModelError as exc:
    st.error(str(exc))
    st.stop()

crop = st.selectbox("Select crop", ["rice", "coffee", "cassava"])
upload = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png", "webp"])
if upload:
    try:
        image = Image.open(upload)
        st.image(image, caption=f"Uploaded {crop} leaf", use_container_width=True)
        if st.button("Classify leaf", type="primary"):
            with st.spinner("Loading model and classifying…"):
                model = load_model(configs[crop])
                results = predict(image, configs[crop], model)
            st.subheader(results[0]["label"])
            st.metric("Confidence", f'{results[0]["confidence"]:.1%}')
            st.markdown("**Top predictions**")
            for result in results:
                st.write(f'{result["label"]} — {result["confidence"]:.1%}')
                st.progress(float(result["confidence"]))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"Unable to read this image: {exc}")
    except ModelError as exc:
        st.error(str(exc))
    except Exception as exc:  # Keep unexpected inference faults visible and actionable.
        st.error(f"Classification failed: {exc}")
