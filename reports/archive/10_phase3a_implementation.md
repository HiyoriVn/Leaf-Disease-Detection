# Phase 3A — Cassava app implementation (2026-10-28)

## Phạm vi và kiến trúc

Ứng dụng Streamlit tại `app.py` dùng adapter riêng `src/cassava_inference.py` cho checkpoint PyTorch state_dict. Mã TorchScript cũ không bị sửa. `@st.cache_resource` nạp và dùng lại một model CPU sau lần suy luận đầu tiên trong process. Adapter kiểm tra SHA-256, gọi `timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=5)`, `torch.load(..., map_location="cpu", weights_only=True)`, `load_state_dict(strict=True)`, `eval()` và `torch.inference_mode()`.

Tiền xử lý: decode bằng Pillow, RGB, OpenCV `INTER_AREA` resize vuông 256, center crop 224, chia 255 thành float32, chuẩn hoá ImageNet, NCHW `[1,3,224,224]`. Chỉ softmax logits `[1,5]`. Thứ tự lớp được kiểm tra cố định trong code và cấu hình: CBB, CBSD, CGM, CMD, Healthy. Ảnh nguồn và manifest không thay đổi.

Checkpoint `checkpoints/cassava_mobilenetv3_best.pth` có SHA-256 `4043a480d7a48d636f4feefbe31df4c52eeadd8edb8404d31319eddd807ac4bb` và là bản đã kiểm chứng Phase 2. Không có checkpoint khác được tải xuống.

## Kiểm thử thực tế

Môi trường `.venv-phase2` trên Windows, Python 3.13, PyTorch 2.8.0 CPU, timm 1.0.20, Streamlit 1.66.1. Lệnh `python -m pytest -q --basetemp .pytest-tmp\phase3a-final tests\test_cassava_app.py tests\test_contract.py` chạy **12 passed in 6.84s**. Các bài test bao phủ decode, ảnh sai, RGB/tensor shape, xác suất/tên lớp, output sai, state_dict không tương thích, và dự đoán của checkpoint thật trên cùng năm ảnh với CSV smoke test Phase 2. Nhãn và confidence khớp trong sai số tuyệt đối 1e-5.

Ứng dụng đã khởi động tại `http://127.0.0.1:8501/`. Trên giao diện thực, ảnh `1000015157.jpg` (135.6 KB) từ Cassava train_images đã tải lên và hiện preview. Sau khi bấm **Dự đoán bệnh**, UI hiển thị CBB 30.9%; top 3: CBB 30.9%, Healthy 25.8%, CGM 25.4%. Đây là kiểm thử luồng hoạt động, **không phải accuracy**. Chế độ camera đã render widget và nút chụp; trình duyệt kiểm thử chưa cấp quyền camera nên chụp ảnh thật chưa được xác nhận.

Ứng dụng báo rõ khi chưa chọn ảnh, ảnh sai định dạng/nội dung, ảnh quá lớn, checkpoint thiếu/sai hash, trọng số không tương thích hoặc đầu ra không hợp lệ. Trên UI, bấm dự đoán không có ảnh đã hiện cảnh báo yêu cầu tải/chụp ảnh. Giới hạn upload Streamlit được đặt 20 MB trong `.streamlit/config.toml` và UI đã hiển thị `20MB per file`; decoder giới hạn thêm 24 triệu pixel.

## Hạn chế và Phase 3B

Checkpoint của bên thứ ba được huấn luyện từ cùng bộ dữ liệu Cassava; split nội bộ mới từ nguồn này không tạo test độc lập. Confidence softmax chưa được hiệu chuẩn. Chưa xác minh camera trên phần cứng và trình duyệt của người dùng. UI chỉ hỗ trợ lá sắn; chưa tích hợp Rice/Coffee, chưa huấn luyện hoặc fine-tuning.

Phase 3B nhanh nhất: cố định bản app và checkpoint này, thu thập một bộ ảnh Cassava mới có nhãn và provenance độc lập, kiểm tra trùng ảnh với dữ liệu đã biết, rồi đánh giá macro-F1/confusion matrix trên bộ ảnh đó. Chỉ quyết định fine-tuning MobileNetV3-Small pretrained ImageNet sau khi có kết quả đánh giá đáng tin cậy; không dùng split cùng nguồn để tuyên bố chất lượng ngoài thực địa.
