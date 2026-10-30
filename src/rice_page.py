"""Streamlit Rice panel, embedded inside the single app.py crop selector."""
from __future__ import annotations

from io import BytesIO
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from src.rice_inference import (
    RiceInferenceError, load_rice_config, load_rice_interpreter, predict_rice,
)
from src.rice_quality_gate import check_rice_image

DISPLAY = {
    "bacterial_leaf_blight": "Bạc lá vi khuẩn",
    "bacterial_leaf_streak": "Sọc trong vi khuẩn",
    "bacterial_panicle_blight": "Cháy bông vi khuẩn",
    "blast": "Đạo ôn",
    "brown_spot": "Đốm nâu",
    "dead_heart": "Chết đọt",
    "downy_mildew": "Sương mai",
    "hispa": "Sâu gai",
    "normal": "Khỏe mạnh",
    "tungro": "Vàng lùn Tungro",
}


def render_rice_page() -> None:
    st.markdown(
        """<style>
        .rice-hero {padding: 20px 24px; background: linear-gradient(115deg,#113d31,#218160);
            border-radius: 18px; color: #fff; margin-bottom: 22px;}
        .rice-hero h2 {color:#fff; font-size:1.65rem; margin:0 0 8px 0;}
        .rice-hero p {color:#d8efe6; margin:0; font-size:0.94rem;}
        </style>""", unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="rice-hero"><h2>Nhận diện bệnh lá lúa · LiSA</h2>'
        '<p>Ảnh RGB bất kỳ · 10 nhóm triệu chứng · Mô hình TFLite</p></div>',
        unsafe_allow_html=True,
    )

    try:
        cfg = load_rice_config()
    except RiceInferenceError as exc:
        st.error(str(exc))
        return

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.subheader("01 / Ảnh đầu vào")
        method = st.segmented_control(
            "Nguồn ảnh", options=["Tải ảnh", "Camera"], default="Tải ảnh",
            key="rice_source",
        )
        source = (
            st.file_uploader("Chọn ảnh JPG, PNG hoặc WebP", type=["jpg", "jpeg", "png", "webp"],
                             key="rice_image")
            if method == "Tải ảnh"
            else st.camera_input("Chụp ảnh lá lúa", key="rice_camera")
        )
        image = None
        if source is not None:
            try:
                data = source.getvalue()
                if len(data) > 20_000_000:
                    raise RiceInferenceError("File ảnh vượt quá 20 MB.")
                with Image.open(BytesIO(data)) as original:
                    if original.width * original.height > cfg.max_pixels:
                        raise RiceInferenceError("Ảnh vượt quá giới hạn 24 triệu pixel.")
                    original.load()
                    image = ImageOps.exif_transpose(original).convert("RGB")
                st.image(image, caption=f"Ảnh gốc · {image.width} × {image.height} px",
                         use_container_width=True)
            except (UnidentifiedImageError, OSError, ValueError,
                    Image.DecompressionBombError, RiceInferenceError) as exc:
                st.error(f"Không đọc được ảnh: {exc}")
        with st.expander("Danh sách 10 nhãn"):
            st.write(", ".join(f"{DISPLAY.get(label, label)} ({label})" for label in cfg.labels))

    with right:
        st.subheader("02 / Kết quả phân tích")
        st.caption("Ảnh không phải lá lúa có thể bị từ chối trước khi chạy LiSA.")
        experimental = st.checkbox(
            "Chế độ thử nghiệm (chỉ dành cho kiểm thử)",
            value=False, key="rice_experimental",
            help="Chỉ bỏ qua bước chờ hiệu chỉnh bộ lọc. Không bỏ qua các ảnh bị chặn rõ ràng.",
        )
        if st.button("Phân tích ảnh", key="rice_analyze", type="primary",
                     use_container_width=True):
            if image is None:
                st.warning("Cần chọn hoặc chụp một ảnh.")
                return
            try:
                decision = check_rice_image(image)
                if decision.status == "reject":
                    st.error("Ảnh bị từ chối: " + decision.reason)
                    st.caption("Không hiển thị dự đoán bệnh cho ảnh này.")
                    return
                if decision.status == "unverified" and not experimental:
                    st.warning(
                        "Bộ kiểm tra ảnh lá lúa chưa được huấn luyện/hiệu chỉnh. "
                        "Chưa thể xác nhận đây là lá lúa, nên hệ thống không xuất kết quả bệnh."
                    )
                    st.caption("Có thể bật chế độ thử nghiệm để kiểm tra kỹ thuật mô hình.")
                    return
                with st.spinner("Đang chạy LiSA..."):
                    interpreter = load_rice_interpreter(cfg)
                    result = predict_rice(image, interpreter, cfg)
                name = DISPLAY.get(result["predicted_label"], result["predicted_label"])
                if decision.status == "unverified" or not cfg.preprocessing_verified:
                    st.caption("KẾT QUẢ THỬ NGHIỆM · CHƯA XÁC MINH ĐẦY ĐỦ")
                else:
                    st.caption("KẾT QUẢ SUY LUẬN")
                st.markdown(f"### {name}")
                st.caption(result["predicted_label"])
                st.metric("Điểm Softmax · không phải độ chắc chắn chẩn đoán",
                          f"{result['confidence']:.1%}")
                st.markdown("**Top 3 kết quả**")
                for item in result["top3"]:
                    st.write(f"{DISPLAY.get(item['label'], item['label'])} — {item['probability']:.1%}")
                    st.progress(float(item["probability"]))
                if decision.status == "unverified":
                    st.info("Ảnh chưa được xác nhận thuộc phạm vi lá lúa; không dùng kết quả để chẩn đoán.")
            except RiceInferenceError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Không thể xử lý ảnh: {exc}")

    st.divider()
    st.caption("AI hỗ trợ tham khảo, không thay thế kiểm tra thực tế. "
               "Điểm Softmax cao không chứng minh ảnh đầu vào là lá lúa.")
