"""Hệ thống phân loại bệnh lá cây (Cassava / Coffee / Rice)."""
from __future__ import annotations

from pathlib import Path
import os
import streamlit as st
from PIL import Image

# Import phần xử lý lá sắn sẵn có của bạn bạn
from src.model_registry import available_models

st.set_page_config(page_title="LeafSense | Plant Health AI", page_icon="🌱", layout="wide")

ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "configs" / "plant_model_registry.json"


@st.cache_data(ttl=600, show_spinner=False)
def get_available_models(crop: str, coffee_model_path: str):
    return available_models(REGISTRY_PATH, ROOT, crop, coffee_model_path or None)

# Một giao diện, ngữ cảnh riêng cho từng loại cây.
CROP_MODES = {
    "Lá Lúa (Rice)": {
        "icon": "🌾",
        "title": "Nhận diện bệnh trên lúa",
        "sidebar": "🌾 Lúa (Rice)",
        "description": "LiSA TFLite · 10 nhóm bệnh, triệu chứng và trạng thái",
    },
    "Lá Sắn (Cassava)": {
        "icon": "🌿",
        "title": "Nhận diện bệnh trên lá sắn",
        "sidebar": "🌿 Sắn (Cassava)",
        "description": "MobileNetV3 / CropNet · Các nhóm bệnh lá sắn",
    },
    "Lá Cà Phê (Coffee)": {
        "icon": "☕",
        "title": "Nhận diện bệnh trên lá cà phê",
        "sidebar": "☕ Cà phê (Coffee)",
        "description": "ResNet18 · 4 nhóm bệnh và trạng thái lá cà phê",
    },
}

st.sidebar.title("LeafSense")
plant_type = st.sidebar.selectbox(
    "Chọn loại cây cần kiểm tra",
    options=tuple(CROP_MODES),
    format_func=lambda key: CROP_MODES[key]["sidebar"],
    key="crop_selector",
)
crop_ui = CROP_MODES[plant_type]

if st.session_state.get("_last_leafsense_crop") != plant_type:
    st.session_state.pop("rice_last_report", None)
    st.session_state["_last_leafsense_crop"] = plant_type

st.title("LeafSense")
st.subheader(f'{crop_ui["icon"]} {crop_ui["title"]}')
st.caption(crop_ui["description"])
st.sidebar.caption("Chế độ đang chọn: " + crop_ui["title"])

# ==========================================
# 1. NẾU NGƯỜI DÙNG CHỌN: LÁ SẮN (CASSAVA)
# ==========================================
if plant_type == "Lá Lúa (Rice)":
    from src.rice_page import render_rice_page
    render_rice_page()

elif plant_type == "Lá Sắn (Cassava)":
    from src.cassava_inference import CassavaInferenceError, decode_image, load_cassava_model, load_config, predict_image
    from src.cropnet_bridge import CLASS_NAMES as CROPNET_CLASSES, available as cropnet_available, predict_cropnet
    available_cassava_models = get_available_models("cassava", "")
    MODEL_OPTIONS = {entry["label"]: entry for entry in available_cassava_models}
    CROPNET_OPTION = "Google CropNet (thử nghiệm, 6 lớp)"
    if cropnet_available():
        MODEL_OPTIONS[CROPNET_OPTION] = None

    @st.cache_resource(show_spinner=False)
    def get_cassava_model(config_path: str):
        return load_cassava_model(load_config(Path(config_path)))

    st.markdown("**Phân tích ảnh lá sắn · 5 lớp (model nội bộ)**")
    model_choice = st.selectbox("Chọn mô hình Cassava", list(MODEL_OPTIONS)) if MODEL_OPTIONS else None
    if not MODEL_OPTIONS:
        st.warning("Chưa sẵn sàng: chưa tìm thấy checkpoint hợp lệ của mô hình lá sắn.")
        st.info("Đặt checkpoint theo đường dẫn và SHA-256 trong README.md, sau đó khởi động lại ứng dụng. Không thể phân tích ảnh sắn khi chưa có model.")
        with st.expander("5 lớp của model sắn"):
            for label in load_config(ROOT / "configs" / "cassava_app.json").class_names:
                st.write(label)
        st.stop()
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
                from src.disease_advice import show_advice
                if not is_cropnet:
                    show_advice("cassava", result["predicted_label"])
            except (CassavaInferenceError, RuntimeError) as exc:
                st.error(str(exc))

    with st.expander("Các lớp mô hình hỗ trợ"):
        for label in (CROPNET_CLASSES if is_cropnet else model_config.class_names):
            st.write(f"• {label}")

# ==========================================
# 2. NẾU NGƯỜI DÙNG CHỌN: LÁ CÀ PHÊ (COFFEE)
# ==========================================
else:
    from src.coffee_inference import CoffeeInferenceError, load_coffee_model, predict_coffee
    st.markdown("**Phân tích ảnh lá cà phê · 4 lớp**")
    st.write("Mô hình cà phê hỗ trợ các nhãn Cerscospora, Leaf rust, Phoma và Healthy.")
    st.caption("Mẹo ảnh cà phê: chụp rõ cả hai mặt lá, đủ sáng và hạn chế nền phức tạp.")
    coffee_override = os.environ.get("COFFEE_MODEL_PATH", "")
    available_coffee_models = get_available_models("coffee", coffee_override)
    COFFEE_OPTIONS = {entry["label"]: entry for entry in available_coffee_models}
    coffee_choice = st.selectbox("Chọn mô hình Coffee", list(COFFEE_OPTIONS)) if COFFEE_OPTIONS else None
    selected_coffee_model = COFFEE_OPTIONS.get(coffee_choice) if coffee_choice else None
    if selected_coffee_model is None:
        st.warning("Chưa sẵn sàng: chưa tìm thấy checkpoint hợp lệ của mô hình lá cà phê.")
        st.info("Đặt checkpoint theo README.md hoặc cấu hình COFFEE_MODEL_PATH. Khi chưa có model, không thể phân tích ảnh cà phê.")
        st.stop()
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
                from src.disease_advice import show_advice
                show_advice("coffee", label)

                if label == "Healthy":
                    st.info("Mô hình dự đoán lớp Healthy trong 4 nhóm được hỗ trợ.")
                else:
                    st.warning("Phát hiện triệu chứng bệnh trên lá cà phê. Nên kiểm tra để phòng trừ kịp thời.")
            except CoffeeInferenceError as exc:
                st.error(str(exc))
