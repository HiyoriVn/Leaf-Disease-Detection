"""Hệ thống phân loại bệnh cây trồng (Lá Sắn & Lá Cà Phê) với Streamlit."""
from __future__ import annotations

from pathlib import Path
import streamlit as st
from PIL import Image

# Import phần xử lý lá sắn sẵn có của bạn bạn
from src.cassava_inference import CassavaInferenceError, decode_image, load_cassava_model, load_config, predict_image
from src.cropnet_bridge import CLASS_NAMES as CROPNET_CLASSES, available as cropnet_available, predict_cropnet

# Import phần xử lý lá cà phê do bạn phụ trách
from src.coffee_inference import predict_coffee
import torch
import torch.nn as nn
from torchvision import models

st.set_page_config(page_title="Phân loại bệnh lá cây AI", page_icon="🌿", layout="centered")

ROOT = Path(__file__).resolve().parent

# --- MENU CHỌN LOẠI CÂY TRỒNG ---
st.title("🌿 Hệ thống Phân loại Bệnh Cây Trồng bằng AI")
st.write("Đồ án Thị giác máy tính - Hỗ trợ chẩn đoán bệnh trên Lá Sắn và Lá Cà Phê Robusta.")

plant_type = st.sidebar.selectbox(
    "📂 Chọn loại cây cần kiểm tra:",
    ("Lá Sắn (Cassava)", "Lá Cà Phê (Coffee)")
)

# ==========================================
# 1. NẾU NGƯỜI DÙNG CHỌN: LÁ SẮN (CASSAVA)
# ==========================================
if plant_type == "Lá Sắn (Cassava)":
    MODEL_OPTIONS = {
        "Checkpoint công bố (mặc định)": ROOT / "configs" / "cassava_app.json",
        "Mô hình nhóm fine-tune (thử nghiệm)": ROOT / "configs" / "cassava_finetuned_app.json",
        "EfficientNet-B3 (Fine-tuned by our team)": ROOT / "configs" / "cassava_efficientnet_b3_app.json",
    }
    CROPNET_OPTION = "Google CropNet (thử nghiệm, 6 lớp)"
    if cropnet_available():
        MODEL_OPTIONS[CROPNET_OPTION] = None

    @st.cache_resource(show_spinner=False)
    def get_cassava_model(config_path: str):
        return load_cassava_model(load_config(Path(config_path)))

    st.subheader("🥔 Chẩn đoán bệnh trên lá Sắn")
    model_choice = st.selectbox("Chọn mô hình Cassava", list(MODEL_OPTIONS))
    is_cropnet = model_choice == CROPNET_OPTION
    model_config = load_config(MODEL_OPTIONS[model_choice] or MODEL_OPTIONS["Checkpoint công bố (mặc định)"])
    
    if is_cropnet:
        st.caption("CropNet dùng nhãn Green Mite và thêm Unknown...")
    else:
        st.caption(f"Đang sử dụng mô hình: {model_choice}")

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
                    result = predict_cropnet(source.getvalue()) if is_cropnet else predict_image(image, get_cassava_model(str(MODEL_OPTIONS[model_choice])), model_config)
                st.subheader(result["predicted_label"])
                st.metric("Độ tin cậy của mô hình", f'{result["confidence"]:.1%}')
                st.markdown("**Ba khả năng cao nhất**")
                for item in result["top3"]:
                    st.write(f'{item["label"]} — {item["probability"]:.1%}')
                    st.progress(item["probability"])
            except (CassavaInferenceError, RuntimeError) as exc:
                st.error(str(exc))

# ==========================================
# 2. NẾU NGƯỜI DÙNG CHỌN: LÁ CÀ PHÊ (COFFEE)
# ==========================================
else:
    st.subheader("☕ Chẩn đoán bệnh trên lá Cà phê ")
    st.write("Tải lên ảnh chụp lá cà phê để AI nhận diện các bệnh (Cerscospora, Leaf rust, Phoma, Healthy).")
    st.caption("💡 **Mẹo:** Để đạt độ chính xác cao nhất, hãy chụp/chọn ảnh cận cảnh 1 chiếc lá cà phê, rõ vết bệnh và hạn chế dính bàn tay hoặc phông nền quá phức tạp.")
    # 1. Định nghĩa từ điển chuyển đổi tên bệnh sang Tiếng Việt
    COFFEE_VIETNAMESE_NAMES = {
        "Cerscospora": "Bệnh đốm lá (Cercospora)",
        "Healthy": "Lá khỏe mạnh (Không bệnh)",
        "Leaf rust": "Bệnh rỉ sắt (Leaf rust)",
        "Phoma": "Bệnh Phoma"
    }
    
    coffee_source = st.file_uploader("Chọn ảnh lá cà phê", type=["jpg", "jpeg", "png", "webp"], key="coffee_file_uploader_unique")
    
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
            with st.spinner("Đang phân tích ảnh cà phê…"):
                # Nhận thêm top3_list từ hàm predict_coffee
                label, score, top3_list = predict_coffee(coffee_image)
                
            # Chuyển đổi nhãn kết quả chính sang tiếng Việt
            display_label = COFFEE_VIETNAMESE_NAMES.get(label, label)
                
            st.subheader(f"Kết quả: {display_label}")
            st.metric("Độ tin cậy của mô hình", f"{score:.2f}%")
            
            # --- HIỂN THỊ TOP 3 KHẢ NĂNG ---
            st.markdown("**Ba khả năng cao nhất:**")
            for item in top3_list:
                item_label_vi = COFFEE_VIETNAMESE_NAMES.get(item["label"], item["label"])
                prob_pct = item["probability"] * 100
                st.write(f"{item_label_vi} — {prob_pct:.1f}%")
                st.progress(item["probability"])
            
            # Đánh giá trạng thái lá
            if "Healthy" in label:
                st.info("Lá cà phê hoàn toàn khỏe mạnh!")
            else:
                st.warning("Phát hiện triệu chứng bệnh trên lá cà phê. Nên kiểm tra để phòng trừ kịp thời.")