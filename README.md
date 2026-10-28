# Cassava Leaf Disease Detection — Capstone

Ứng dụng Streamlit phân loại **lá sắn** bằng checkpoint MobileNetV3-Small đã kiểm chứng ở Phase 2. Ứng dụng hỗ trợ tải ảnh hoặc chụp bằng camera, xem ảnh, dự đoán một trong năm lớp, độ tin cậy softmax và ba khả năng cao nhất. Kết quả là dự đoán AI, không phải chẩn đoán nông nghiệp đã xác nhận.

## Chạy ứng dụng trên Windows PowerShell

Yêu cầu Python 3.13 trên Windows. Từ thư mục repository:

```powershell
cd C:\Users\Admin\Downloads\Leaf-Disease-Detection
py -3.13 -m venv .venv-phase2
.\.venv-phase2\Scripts\Activate.ps1
python -m pip install -r requirements-app.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

Mở [http://127.0.0.1:8501](http://127.0.0.1:8501). Nếu môi trường `.venv-phase2` đã có sẵn, bỏ qua lệnh tạo venv. Nếu PowerShell chặn activation, chạy trực tiếp `.\.venv-phase2\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1`.

Đặt checkpoint đã kiểm chứng tại `checkpoints/cassava_mobilenetv3_best.pth` trước khi dự đoán. Ứng dụng đối chiếu SHA-256 với `configs/cassava_app.json` và báo lỗi rõ nếu thiếu/sai trọng số. Checkpoint không được commit. Xem [hướng dẫn sử dụng](docs/CASSAVA_APP_GUIDE.md) để tải đúng bản và xử lý lỗi.

## Kiểm thử

```powershell
.\.venv-phase2\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp\phase3a tests\test_cassava_app.py tests\test_contract.py
```

Bài test so sánh năm dự đoán trên cùng ảnh với [smoke test Phase 2](reports/smoke_cassava/predictions.csv) khi checkpoint, dữ liệu và CSV có mặt. Những ảnh này có thể đã nằm trong dữ liệu huấn luyện checkpoint, nên **không** dùng các dự đoán đó để tính accuracy độc lập.

## Cấu trúc liên quan

- `app.py`: giao diện Cassava dùng Streamlit.
- `src/cassava_inference.py`: giải mã, tiền xử lý, nạp state_dict và suy luận CPU.
- `configs/cassava_app.json`: đường dẫn/hash checkpoint, thứ tự lớp, tiền xử lý và giới hạn ảnh.
- `requirements-app.txt`: các thư viện đã dùng trong môi trường thử nghiệm.
- `reports/10_phase3a_implementation.md`: kết quả triển khai và thử nghiệm thực tế.

Phase 1 và 2 được ghi trong `reports/`; các manifest nằm tại `manifests/`. Mã TorchScript cũ trong `src/inference.py` vẫn dành cho hợp đồng model riêng và không được ứng dụng Cassava hiện tại gọi. Rice và Coffee chưa tích hợp vào ứng dụng; không có fine-tuning trong Phase 3A.
