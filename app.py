"""Cassava leaf disease prediction with the verified MobileNetV3-Small checkpoint."""
from __future__ import annotations

from pathlib import Path
import streamlit as st

from src.cassava_inference import CassavaInferenceError, decode_image, load_cassava_model, load_config, predict_image
from src.cropnet_bridge import CLASS_NAMES as CROPNET_CLASSES, available as cropnet_available, predict_cropnet


st.set_page_config(page_title="Phân loại bệnh lá sắn", page_icon="🌿", layout="centered")


ROOT = Path(__file__).resolve().parent
MODEL_OPTIONS = {
    "Checkpoint công bố (mặc định)": ROOT / "configs" / "cassava_app.json",
    "Mô hình nhóm fine-tune (thử nghiệm)": ROOT / "configs" / "cassava_finetuned_app.json",
    "EfficientNet-B3 (Fine-tuned by our team)": ROOT / "configs" / "cassava_efficientnet_b3_app.json",
}
CROPNET_OPTION = "Google CropNet (thử nghiệm, 6 lớp)"
if cropnet_available():
    MODEL_OPTIONS[CROPNET_OPTION] = None


@st.cache_resource(show_spinner=False)
def get_model(config_path: str):
    """One shared, read-only CPU model per checkpoint across Streamlit reruns."""
    return load_cassava_model(load_config(Path(config_path)))


st.title("🌿 Phân loại bệnh lá sắn")
st.write("Tải ảnh lá sắn hoặc chụp ảnh trực tiếp để xem dự đoán từ mô hình CNN đã chọn.")
st.info("Đây là dự đoán của AI, không phải chẩn đoán nông nghiệp đã xác nhận. Hãy tham khảo chuyên gia trước khi xử lý cây trồng.")

model_choice = st.selectbox("Chọn mô hình Cassava", list(MODEL_OPTIONS))
is_cropnet = model_choice == CROPNET_OPTION
model_config = load_config(MODEL_OPTIONS[model_choice] or MODEL_OPTIONS["Checkpoint công bố (mặc định)"])
if is_cropnet:
    st.caption("CropNet dùng nhãn Green Mite và thêm Unknown; nhãn local là Green Mottle. Đây là chế độ khám phá, chưa có metric đánh giá hợp lệ trên dữ liệu local. Lần đầu suy luận có thể chậm do TensorFlow khởi động.")
elif model_choice == "EfficientNet-B3 (Fine-tuned by our team)":
    st.caption("EfficientNet-B3 · ImageNet pretrained, fine-tuned trên Cassava · input 300×300 · theo run PC: validation accuracy 83,12%, macro-F1 0,7052 (chờ tái lập). Đây là validation cùng nguồn, không phải internal test hay ảnh thực địa độc lập.")
elif model_choice == "Mô hình nhóm fine-tune (thử nghiệm)":
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
                result = predict_cropnet(source.getvalue()) if is_cropnet else predict_image(image, get_model(str(MODEL_OPTIONS[model_choice])), model_config)
            st.subheader(result["predicted_label"])
            st.metric("Độ tin cậy của mô hình", f'{result["confidence"]:.1%}')
            st.caption("Xác suất thể hiện mức ưu tiên giữa 6 lớp CropNet." if is_cropnet else "Điểm softmax thể hiện mức ưu tiên giữa 5 lớp; điểm cao không bảo đảm dự đoán đúng.")
            st.markdown("**Ba khả năng cao nhất**")
            for item in result["top3"]:
                st.write(f'{item["label"]} — {item["probability"]:.1%}')
                st.progress(item["probability"])
        except (CassavaInferenceError, RuntimeError) as exc:
            st.error(str(exc))

with st.expander("Các lớp mô hình hỗ trợ"):
    for label in (CROPNET_CLASSES if is_cropnet else model_config.class_names):
        st.write(f"• {label}")
