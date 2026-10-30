"""Unified Rice diagnostics dashboard. One crop selector in app.py."""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st

from src.rice_inference import (
    RiceInferenceError, load_rice_config, load_rice_interpreter,
    predict_rice, preprocess_rice,
)
from src.rice_quality_gate import check_rice_image
from src.disease_advice import show_advice

LABELS = {
    "bacterial_leaf_blight": "Bạc lá vi khuẩn",
    "bacterial_leaf_streak": "Sọc trong vi khuẩn",
    "bacterial_panicle_blight": "Cháy bông vi khuẩn",
    "blast": "Đạo ôn",
    "brown_spot": "Đốm nâu",
    "dead_heart": "Chết đọt / sâu đục thân",
    "downy_mildew": "Sương mai",
    "hispa": "Sâu gai hại lúa",
    "normal": "Khỏe mạnh",
    "tungro": "Vàng lùn Tungro",
}


def _model_state(config):
    path = config.model_path
    found = path.is_file()
    size = path.stat().st_size if found else None
    return {
        "file_found": found,
        "path": str(path),
        "expected_bytes": config.expected_model_bytes,
        "actual_bytes": size,
        "size_matches": bool(found and size == config.expected_model_bytes),
        "preprocessing_verified": config.preprocessing_verified,
        "pixel_scale": config.pixel_scale,
    }


def _process(image, config, gate):
    """Observable, timed path: tensor -> TFLite load -> invoke -> probabilities."""
    log = {"steps": [], "quality_gate": {
        "status": gate.status, "reason": gate.reason, "score": gate.score
    }}
    start = time.perf_counter()
    def step(name, ok, details):
        log["steps"].append({"step": name, "status": "OK" if ok else "ERROR", "details": details})
    try:
        t0 = time.perf_counter()
        state = _model_state(config)
        log["model"] = state
        if not state["size_matches"]:
            raise RiceInferenceError("Model TFLite không tồn tại hoặc dung lượng khác artifact gốc.")
        log["model"]["sha256"] = sha256(config.model_path.read_bytes()).hexdigest()
        step("Kiểm tra model", True, f"{state['actual_bytes']:,} bytes")
        log["timings_ms"] = {}
        t1 = time.perf_counter()
        tensor = preprocess_rice(image, config)
        log["input"] = {
            "image_size": [image.width, image.height],
            "tensor_shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "min": float(tensor.min()),
            "max": float(tensor.max()),
            "mean": float(tensor.mean()),
        }
        log["timings_ms"]["preprocess"] = round((time.perf_counter()-t1)*1000, 2)
        step("Tiền xử lý", True, str(log["input"]["tensor_shape"]))
        t1 = time.perf_counter()
        interpreter = load_rice_interpreter(config)
        log["timings_ms"]["load_tflite"] = round((time.perf_counter()-t1)*1000, 2)
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        log["tflite"] = {
            "input_shape": [int(v) for v in inp["shape"]],
            "input_dtype": str(inp["dtype"]),
            "output_shape": [int(v) for v in out["shape"]],
            "output_dtype": str(out["dtype"]),
        }
        step("Tải TFLite", True, "Model và input/output tensor hợp lệ")
        t1 = time.perf_counter()
        result = predict_rice(image, interpreter, config)
        log["timings_ms"]["inference"] = round((time.perf_counter()-t1)*1000, 2)
        log["timings_ms"]["total"] = round((time.perf_counter()-start)*1000, 2)
        log["prediction"] = {
            "top1": result["predicted_label"],
            "confidence": result["confidence"],
            "probability_sum": float(sum(result["probabilities"])),
            "all_probabilities": {label: float(prob) for label, prob in zip(config.labels, result["probabilities"])},
            "temperature_calibrated": result.get("temperature_calibrated", False),
        }
        step("Suy luận", True, f"{result['predicted_label']} · {result['confidence']:.2%}")
        return result, log
    except Exception as exc:
        log["timings_ms"] = log.get("timings_ms", {})
        log["timings_ms"]["total"] = round((time.perf_counter()-start)*1000, 2)
        step("Lỗi xử lý", False, str(exc))
        return None, log


def render_rice_page():
    st.subheader("🌾 Phân tích bệnh và triệu chứng trên lúa")
    st.caption("LiSA-MobileNetV2 · 10 nhãn · Ảnh RGB, kích thước bất kỳ")

    try:
        config = load_rice_config()
    except RiceInferenceError as exc:
        st.error(f"Lỗi cấu hình Rice: {exc}")
        return

    state = _model_state(config)
    s1, s2, s3 = st.columns(3)
    s1.metric("Tệp model", "Đã tìm thấy" if state["file_found"] else "Không tìm thấy")
    s2.metric("Kiểm tra dung lượng", "Khớp" if state["size_matches"] else "Không khớp")
    s3.metric("Tiền xử lý", "Đã đối chiếu" if config.preprocessing_verified else "Chưa xác minh")

    if not state["size_matches"]:
        st.error("Không thể chạy LiSA. Kiểm tra đường dẫn và kích thước file trong phần kỹ thuật.")
    with st.expander("Kiểm tra model và cấu hình", expanded=False):
        st.json(state)
        st.caption("Có file và đúng kích thước chưa chứng minh model dự đoán đúng. Hãy chạy P0.")

    left, right = st.columns([1, 1.2], gap="large")
    image = None
    image_key = None
    with left:
        st.markdown("#### 1. Ảnh đầu vào")
        method = st.radio("Nguồn ảnh", ["Tải ảnh", "Camera"], horizontal=True, key="rice_source")
        source = (st.file_uploader("Chọn JPEG / PNG / WebP (tối đa 20 MB)",
                                  type=["jpg", "jpeg", "png", "webp"], key="rice_image")
                  if method == "Tải ảnh" else st.camera_input("Chụp ảnh lúa", key="rice_camera"))
        if source is not None:
            try:
                data = source.getvalue()
                if len(data) > 20_000_000:
                    raise RiceInferenceError("Kích thước file vượt 20 MB.")
                image_key = sha256(data).hexdigest()
                with Image.open(BytesIO(data)) as img:
                    if img.width * img.height > config.max_pixels:
                        raise RiceInferenceError("Độ phân giải ảnh vượt 24 triệu pixel.")
                    img.load()
                    image = ImageOps.exif_transpose(img).convert("RGB")
                st.image(image, caption=f"Ảnh {image.width} × {image.height} px", width="stretch")
            except (RiceInferenceError, UnidentifiedImageError, OSError, ValueError,
                    Image.DecompressionBombError) as exc:
                st.error(f"Không đọc được ảnh: {exc}")
        if st.button("Phân tích ảnh", key="rice_analyze", type="primary",
                     disabled=not state["size_matches"], width="stretch"):
            if image is None:
                st.warning("Hãy tải hoặc chụp một ảnh hợp lệ.")
            else:
                st.session_state.pop("rice_last_report", None)
                with st.status("Đang kiểm tra đầu vào và chạy LiSA...", expanded=True) as status:
                    t0 = time.perf_counter()
                    gate = check_rice_image(image)
                    gate_elapsed_ms = round((time.perf_counter()-t0)*1000, 2)
                    st.write("Kiểm tra đầu vào:", gate.status, "—", gate.reason)
                    if gate.status == "reject":
                        st.session_state["rice_last_report"] = {
                            "image_key": image_key,
                            "result": None,
                            "log": {"quality_gate": {"status": gate.status, "reason": gate.reason},
                                    "model": state, "status": "rejected"},
                        }
                        status.update(label="Ảnh bị từ chối trước suy luận", state="error")
                    else:
                        if gate.status == "unverified":
                            st.write("Bộ lọc OOD chưa hiệu chỉnh: chạy LiSA để kiểm thử, không xác nhận đây là lá lúa.")
                        st.write("Bắt đầu kiểm tra file, tensor đầu vào và TFLite...")
                        result, log = _process(image, config, gate)
                        log["timings_ms"]["quality_gate"] = gate_elapsed_ms
                        st.session_state["rice_last_report"] = {
                            "image_key": image_key, "result": result, "log": log,
                        }
                        if result is None:
                            status.update(label="Suy luận thất bại — xem log phía dưới", state="error")
                        else:
                            status.update(label="TFLite chạy thành công — đã có biểu đồ", state="complete")

    with right:
        st.markdown("#### 2. Kết quả phân tích")
        report = st.session_state.get("rice_last_report")
        if report and image_key and report["image_key"] == image_key:
            result, log = report["result"], report["log"]
            gate_status = log.get("quality_gate", {}).get("status", "unknown")
            if result is None:
                if gate_status == "reject":
                    st.error("Ảnh không được chấp nhận. Không đưa ra chẩn đoán bệnh.")
                else:
                    st.error("Model chưa chạy thành công. Mở mục kỹ thuật để xem bước lỗi.")
            else:
                if gate_status != "accept":
                    st.warning("Đang chạy thử nghiệm: chưa xác minh ảnh là lá lúa. Không dùng để chẩn đoán.")
                if not config.preprocessing_verified:
                    st.caption("Preprocessing chưa được xác minh Keras/TFLite (P0).")
                label = result["predicted_label"]
                st.metric("Dự đoán cao nhất", LABELS.get(label, label))
                st.metric("Điểm Softmax", f"{result['confidence']*100:.2f}%")
                if label == "normal":
                    st.info("Model dự đoán nhóm khỏe mạnh. Kiểm tra thực địa vẫn cần thiết.")
                st.markdown("**Biểu đồ xác suất của 10 lớp**")
                ordered = sorted(zip(config.labels, result["probabilities"]),
                                 key=lambda x: x[1], reverse=True)
                rows = [{"Bệnh / trạng thái": LABELS.get(k, k),
                         "Xác suất (%)": round(float(p)*100, 4),
                         "Mã lớp": k}
                        for k, p in ordered]
                plot = pd.DataFrame(rows)
                st.bar_chart(plot, x="Bệnh / trạng thái", y="Xác suất (%)",
                             horizontal=True, sort=False, height=415,
                             color="#1f8562")
                with st.expander("Chi tiết 10 lớp và xác suất", expanded=True):
                    st.dataframe(plot, hide_index=True, width="stretch")
                show_advice("rice", label)
            with st.expander("Debug kỹ thuật — model, tensor và thời gian", expanded=result is None):
                st.json(log)
                st.download_button("Tải báo cáo JSON", json.dumps(log, ensure_ascii=False, indent=2),
                                   file_name="rice_inference_debug.json", mime="application/json",
                                   key="rice_debug_download")
        else:
            st.info("Chọn ảnh và bấm **Phân tích ảnh** để xem biểu đồ và dữ liệu chi tiết.")
    st.caption("Lưu ý: điểm Softmax chỉ là điểm phân loại trong 10 lớp, không phải xác suất bệnh đã được xác thực ngoài thực địa.")
