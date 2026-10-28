"""Cassava leaf disease prediction with the verified MobileNetV3-Small checkpoint."""
from __future__ import annotations

from pathlib import Path
import streamlit as st

from src.cassava_inference import CassavaInferenceError, decode_image, load_cassava_model, load_config, predict_image


st.set_page_config(page_title="Phân loại bệnh lá sắn", page_icon="🌿", layout="centered")


ROOT = Path(__file__).resolve().parent
MODEL_OPTIONS = {
    "Checkpoint công bố (mặc định)": ROOT / "configs" / "cassava_app.json",
    "Mô hình nhóm fine-tune (thử nghiệm)": ROOT / "configs" / "cassava_finetuned_app.json",
}


@st.cache_resource(show_spinner=False)
def get_model(config_path: str):
    """One shared, read-only CPU model per checkpoint across Streamlit reruns."""
    return load_cassava_model(load_config(Path(config_path)))


st.title("🌿 Phân loại bệnh lá sắn")
st.write("Tải ảnh lá sắn hoặc chụp ảnh trực tiếp để xem dự đoán từ mô hình CNN MobileNetV3-Small.")
st.info("Đây là dự đoán của AI, không phải chẩn đoán nông nghiệp đã xác nhận. Hãy tham khảo chuyên gia trước khi xử lý cây trồng.")

model_choice = st.selectbox("Chọn mô hình Cassava", list(MODEL_OPTIONS))
model_config = load_config(MODEL_OPTIONS[model_choice])
if model_choice == "Mô hình nhóm fine-tune (thử nghiệm)":
    st.caption("Mô hình mới: accuracy 81,8%, macro-F1 0,701 trên internal test cùng nguồn Kaggle; chưa kiểm định ngoài thực địa.")
else:
    st.caption("Checkpoint công bố: đánh giá trên dữ liệu cùng nguồn huấn luyện có nguy cơ trùng lặp; giữ làm lựa chọn mặc định.")

with st.expander("Mẹo chụp ảnh rõ hơn"):
    st.write("Chụp một lá sắn rõ nét, đủ sáng, ít che khuất và để lá chiếm phần lớn khung hình. Ứng dụng chỉ dành cho lá sắn.")

method = st.radio("Chọn nguồn ảnh", ["Tải ảnh lên", "Chụp bằng camera"], horizontal=True)
if method == "Tải ảnh lên":
    source = st.file_uploader("Chọn ảnh lá sắn", type=["jpg", "jpeg", "png", "webp"], key="cassava_upload")
else:
    source = st.camera_input("Chụp ảnh lá sắn", key="cassava_camera")

image = None
if source is not None:
    try:
        image = decode_image(source.getvalue(), model_config)
        st.image(image, caption="Ảnh sẽ được phân tích", width="stretch")
    except CassavaInferenceError as exc:
        st.error(str(exc))

if st.button("Dự đoán bệnh", type="primary", width="stretch"):
    if image is None:
        st.warning("Vui lòng tải lên hoặc chụp một ảnh lá sắn hợp lệ trước khi dự đoán.")
    else:
        try:
            with st.spinner("Đang phân tích ảnh…"):
                result = predict_image(image, get_model(str(MODEL_OPTIONS[model_choice])), model_config)
            st.subheader(result["predicted_label"])
            st.metric("Độ tin cậy của mô hình", f'{result["confidence"]:.1%}')
            st.caption("Điểm softmax thể hiện mức ưu tiên giữa 5 lớp; điểm cao không bảo đảm dự đoán đúng.")
            st.markdown("**Ba khả năng cao nhất**")
            for item in result["top3"]:
                st.write(f'{item["label"]} — {item["probability"]:.1%}')
                st.progress(item["probability"])
        except CassavaInferenceError as exc:
            st.error(str(exc))

with st.expander("Năm lớp mô hình hỗ trợ"):
    for label in model_config.class_names:
        st.write(f"• {label}")
