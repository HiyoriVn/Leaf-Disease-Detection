"""Hệ thống phân loại bệnh cây trồng (Lá Sắn & Lá Cà Phê) với Streamlit."""
from __future__ import annotations

from pathlib import Path
import os
import streamlit as st
from PIL import Image

# Import phần xử lý lá sắn sẵn có của bạn bạn
from src.cassava_inference import CassavaInferenceError, decode_image, load_cassava_model, load_config, predict_image
from src.cropnet_bridge import CLASS_NAMES as CROPNET_CLASSES, available as cropnet_available, predict_cropnet

# Import phần xử lý lá cà phê do bạn phụ trách
from src.coffee_inference import CoffeeInferenceError, load_coffee_model, predict_coffee
from src.model_registry import available_models

st.set_page_config(page_title="Phân loại bệnh lá cây AI", page_icon="🌿", layout="centered")

ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "configs" / "plant_model_registry.json"


@st.cache_data(ttl=600, show_spinner=False)
def get_available_models(crop: str, coffee_model_path: str):
    return available_models(REGISTRY_PATH, ROOT, crop, coffee_model_path or None)

# --- MENU CHỌN LOẠI CÂY TRỒNG ---
st.title("🌿 Hệ thống Phân loại Bệnh Cây Trồng bằng AI")
st.write("Đồ án Thị giác máy tính - Hỗ trợ chẩn đoán bệnh trên Lá Sắn và Lá Cà Phê Robusta.")
st.info("Đây là dự đoán của AI, không phải chẩn đoán nông nghiệp đã xác nhận. Hãy tham khảo chuyên gia trước khi xử lý cây trồng.")

plant_type = st.sidebar.selectbox(
    "📂 Chọn loại cây cần kiểm tra:",
    ("Lá Sắn (Cassava)", "Lá Cà Phê (Coffee)")
)

# ==========================================
# 1. NẾU NGƯỜI DÙNG CHỌN: LÁ SẮN (CASSAVA)
# ==========================================
if plant_type == "Lá Sắn (Cassava)":
    available_cassava_models = get_available_models("cassava", "")
    MODEL_OPTIONS = {entry["label"]: entry for entry in available_cassava_models}
    CROPNET_OPTION = "Google CropNet (thử nghiệm, 6 lớp)"
    if cropnet_available():
        MODEL_OPTIONS[CROPNET_OPTION] = None

    @st.cache_resource(show_spinner=False)
    def get_cassava_model(config_path: str):
        return load_cassava_model(load_config(Path(config_path)))

    st.subheader("🥔 Chẩn đoán bệnh trên lá Sắn")
    model_choice = st.selectbox("Chọn mô hình Cassava", list(MODEL_OPTIONS)) if MODEL_OPTIONS else None
    if not MODEL_OPTIONS:
        st.warning("Chưa có checkpoint Cassava hợp lệ. Hãy tải checkpoint theo hướng dẫn trong README.")
    is_cropnet = model_choice == CROPNET_OPTION
    selected_model = MODEL_OPTIONS.get(model_choice) if model_choice else None
    config_path = selected_model["config_path"] if selected_model else ROOT / "configs" / "cassava_app.json"
    model_config = load_config(Path(config_path))

    if is_cropnet:
        st.caption("CropNet dùng nhãn Green Mite và thêm Unknown; nhãn local là Green Mottle. Đây là chế độ khám phá, chưa có metric đánh giá hợp lệ trên dữ liệu local. Lần đầu suy luận có thể chậm do TensorFlow khởi động.")
    elif selected_model:
        st.caption(selected_model["evaluation_note"])

    with st.expander("Mẹo chụp ảnh rõ hơn"):
        st.write("Chụp một lá sắn rõ nét, đủ sáng, ít che khuất và để lá chiếm phần lớn khung hình.")

    method = st.radio("Chọn nguồn ảnh", ["Tải ảnh lên", "Chụp bằng camera"], horizontal=True, key="cassava_method")
    source = st.file_uploader("Chọn ảnh lá sắn", type=["jpg", "jpeg", "png", "webp"], key="cassava_upload") if method == "Tải ảnh lên" else st.camera_input("Chụp ảnh lá sắn", key="cassava_camera")

    image = None
    if source is not None:
        try:
            image = decode_image(source.getvalue(), model_config)
            st.image(image, caption="Ảnh lá sắn sẽ được phân tích", width="stretch")
        except CassavaInferenceError as exc:
            st.error(str(exc))

    if st.button("Dự đoán bệnh Sắn", type="primary", width="stretch"):
        if image is None:
            st.warning("Vui lòng tải lên hoặc chụp một ảnh lá sắn hợp lệ.")
        else:
            try:
                with st.spinner("Đang phân tích ảnh sắn…"):
                    if selected_model is None and not is_cropnet:
                        st.warning("Chưa có checkpoint Cassava hợp lệ để dự đoán.")
                        st.stop()
                    result = predict_cropnet(source.getvalue()) if is_cropnet else predict_image(image, get_cassava_model(str(config_path)), model_config)
                st.subheader(result["predicted_label"])
                st.metric("Độ tin cậy của mô hình", f'{result["confidence"]:.1%}')
                st.markdown("**Ba khả năng cao nhất**")
                for item in result["top3"]:
                    st.write(f'{item["label"]} — {item["probability"]:.1%}')
                    st.progress(item["probability"])
            except (CassavaInferenceError, RuntimeError) as exc:
                st.error(str(exc))

    with st.expander("Các lớp mô hình hỗ trợ"):
        for label in (CROPNET_CLASSES if is_cropnet else model_config.class_names):
            st.write(f"• {label}")

# ==========================================
# 2. NẾU NGƯỜI DÙNG CHỌN: LÁ CÀ PHÊ (COFFEE)
# ==========================================
else:
    st.subheader("☕ Chẩn đoán bệnh trên lá Cà phê ")
    st.write("Tải lên ảnh chụp lá cà phê để AI nhận diện các bệnh (Cerscospora, Leaf rust, Phoma, Healthy).")
    st.caption("💡 **Mẹo:** Để đạt độ chính xác cao nhất, hãy chụp/chọn ảnh cận cảnh 1 chiếc lá cà phê, rõ vết bệnh và hạn chế dính bàn tay hoặc phông nền quá phức tạp.")
    coffee_override = os.environ.get("COFFEE_MODEL_PATH", "")
    available_coffee_models = get_available_models("coffee", coffee_override)
    COFFEE_OPTIONS = {entry["label"]: entry for entry in available_coffee_models}
    coffee_choice = st.selectbox("Chọn mô hình Coffee", list(COFFEE_OPTIONS)) if COFFEE_OPTIONS else None
    selected_coffee_model = COFFEE_OPTIONS.get(coffee_choice) if coffee_choice else None
    if selected_coffee_model is None:
        st.warning("Chưa có checkpoint Coffee hợp lệ. Đặt file theo hướng dẫn trong README.")
    else:
        st.caption(selected_coffee_model["evaluation_note"])

    @st.cache_resource(show_spinner=False)
    def get_coffee_model(model_path: str, expected_sha256: str, expected_classes: tuple[str, ...]):
        return load_coffee_model(Path(model_path), expected_sha256, list(expected_classes))

    coffee_method = st.radio("Chọn nguồn ảnh cà phê", ["Tải ảnh lên", "Chụp bằng camera"], horizontal=True, key="coffee_method")
    coffee_source = st.file_uploader("Chọn ảnh lá cà phê", type=["jpg", "jpeg", "png", "webp"], key="coffee_file_uploader_unique") if coffee_method == "Tải ảnh lên" else st.camera_input("Chụp ảnh lá cà phê", key="coffee_camera")

    coffee_image = None
    if coffee_source is not None:
        try:
            coffee_image = Image.open(coffee_source).convert("RGB")
            st.image(coffee_image, caption="Ảnh lá cà phê sẽ được phân tích", width="stretch")
        except Exception as e:
            st.error(f"Không thể đọc file ảnh: {e}")

    if st.button("Dự đoán bệnh Cà phê", type="primary", key="coffee_predict_btn"):
        if coffee_image is None:
            st.warning("Vui lòng tải lên một ảnh lá cà phê hợp lệ trước khi dự đoán.")
        else:
            try:
                with st.spinner("Đang phân tích ảnh cà phê…"):
                    if selected_coffee_model is None:
                        st.warning("Chưa có checkpoint Coffee hợp lệ để dự đoán.")
                        st.stop()
                    coffee_model, coffee_classes = get_coffee_model(
                        str(selected_coffee_model["checkpoint_path"]),
                        selected_coffee_model["sha256"],
                        tuple(selected_coffee_model["class_names"]),
                    )
                    label, score, top3_list = predict_coffee(coffee_image, coffee_model, coffee_classes)

                display_labels = selected_coffee_model.get("display_labels", {})
                display_label = display_labels.get(label, label)
                st.subheader(f"Kết quả: {display_label}")
                st.metric("Độ tin cậy của mô hình", f"{score:.2f}%")
                st.markdown("**Ba khả năng cao nhất:**")
                for item in top3_list:
                    item_label_vi = display_labels.get(item["label"], item["label"])
                    st.write(f"{item_label_vi} — {item['probability']:.1%}")
                    st.progress(item["probability"])

                if label == "Healthy":
                    st.info("Lá cà phê hoàn toàn khỏe mạnh!")
                else:
                    st.warning("Phát hiện triệu chứng bệnh trên lá cà phê. Nên kiểm tra để phòng trừ kịp thời.")
            except CoffeeInferenceError as exc:
                st.error(str(exc))
