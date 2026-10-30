"""Additive Streamlit page. The existing root app.py stays unchanged."""
from __future__ import annotations

import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.rice_inference import (
    RiceInferenceError, load_rice_config, load_rice_interpreter, predict_rice,
)

st.title("Nhận diện bệnh lá lúa — LiSA")
st.caption("Ảnh RGB với kích thước bất kỳ; hệ thống tự chuyển về 224 × 224.")
st.info("Dự đoán AI chỉ mang tính tham khảo, không phải chẩn đoán nông nghiệp.")

try:
    cfg = load_rice_config()
except RiceInferenceError as exc:
    st.error(str(exc))
    st.stop()

if not cfg.preprocessing_verified:
    st.warning(
        "Tiền xử lý LiSA chưa được đối chiếu đầy đủ với notebook train. "
        "Đang dùng resize bilinear 224×224, RGB float32 trong khoảng 0–255. "
        "Các kết quả chỉ phục vụ thử nghiệm, chưa được xác nhận tương đương Keras/TFLite."
    )

mode = st.radio("Nguồn ảnh", ("Tải ảnh lên", "Chụp bằng camera"), horizontal=True)
source = (st.file_uploader("Chọn ảnh lá lúa", type=["jpg", "jpeg", "png", "webp"])
          if mode == "Tải ảnh lên" else st.camera_input("Chụp ảnh lá lúa"))
image = None
if source is not None:
    try:
        data = source.getvalue()
        if len(data) > 20_000_000:
            raise RiceInferenceError("Ảnh vượt quá giới hạn 20 MB")
        import io
        with Image.open(io.BytesIO(data)) as opened:
            if opened.width * opened.height > cfg.max_pixels:
                raise RiceInferenceError("Ảnh vượt quá giới hạn 24 triệu pixel")
            opened.load()
            image = opened.copy()
        st.image(image, caption=f"Ảnh gốc: {image.width} × {image.height}", use_container_width=True)
    except (RiceInferenceError, UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        st.error(f"Không đọc được ảnh: {exc}")

if st.button("Dự đoán bệnh lá lúa", type="primary", use_container_width=True):
    if image is None:
        st.warning("Hãy tải lên hoặc chụp một ảnh trước khi dự đoán")
    else:
        try:
            with st.spinner("Đang chạy LiSA TFLite..."):
                interpreter = load_rice_interpreter(cfg)
                result = predict_rice(image, interpreter, cfg)
            st.subheader(result["predicted_label"])
            st.metric("Điểm dự đoán", f"{result['confidence']:.1%}")
            st.write("**Top-3 lớp dự đoán**")
            for item in result["top3"]:
                st.write(f"{item['label']} — {item['probability']:.1%}")
                st.progress(item["probability"])
        except RiceInferenceError as exc:
            st.error(str(exc))

with st.expander("10 lớp bệnh lá lúa"):
    for label in cfg.labels:
        st.write(label)
