# Hướng dẫn ứng dụng nhận diện bệnh lá sắn

## Chuẩn bị

Chạy trên Windows PowerShell với Python 3.13:

```powershell
cd C:\Users\Admin\Downloads\Leaf-Disease-Detection
py -3.13 -m venv .venv-phase2
.\.venv-phase2\Scripts\Activate.ps1
python -m pip install -r requirements-app.txt
```

Nếu venv đã tồn tại, chỉ cần activate và cài yêu cầu. Đặt file `mobilenetv3_best.pth` từ [Space của tác giả](https://huggingface.co/spaces/ctibenkana/cassava-disease-classifier/tree/main) tại `checkpoints/cassava_mobilenetv3_best.pth`. Bản được kiểm chứng có SHA-256 `4043a480d7a48d636f4feefbe31df4c52eeadd8edb8404d31319eddd807ac4bb`. Kiểm tra bằng:

```powershell
Get-FileHash .\checkpoints\cassava_mobilenetv3_best.pth -Algorithm SHA256
```

Không cần tải lại checkpoint nếu file này đã có và hash khớp. Tên và thứ tự năm lớp ở `configs/cassava_app.json` là hợp đồng cố định của checkpoint.

## Sử dụng

```powershell
python -m streamlit run app.py --server.address 127.0.0.1
```

Mở [http://127.0.0.1:8501](http://127.0.0.1:8501), chọn **Tải ảnh lên** hoặc **Chụp bằng camera**, xem ảnh trước, rồi bấm **Dự đoán bệnh**. Tải ảnh JPEG, PNG hoặc WebP tối đa 20 MB và 24 triệu pixel. Ảnh cần là lá sắn đủ sáng, rõ nét, ít bị che khuất; để lá chiếm phần lớn khung hình. Khi dùng camera, trình duyệt cần quyền camera và thiết bị có camera hoạt động.

Ứng dụng hiển thị lớp dự đoán, xác suất softmax của lớp đó và ba lớp có xác suất cao nhất. Xác suất cao không bảo đảm chẩn đoán đúng. Kết quả chỉ hỗ trợ tham khảo; nên hỏi chuyên gia nông nghiệp trước khi xử lý cây.

Ứng dụng có hai lựa chọn trong **Chọn mô hình Cassava**. Checkpoint công bố là mặc định. Model nhóm fine-tune là tùy chọn thử nghiệm, đã được chấm internal test cùng nguồn Kaggle (accuracy 81,8%, macro-F1 0,701); chưa kiểm định ngoài thực địa. Để dùng tùy chọn này, cần có `checkpoints/experiments/full_backbone_smooth_20261028/best.pth` với SHA-256 trong `configs/cassava_finetuned_app.json`. Không dùng mức confidence để so trực tiếp độ chính xác giữa hai model.

## Lỗi thường gặp

- **Không tìm thấy checkpoint:** đặt file đúng đường dẫn trên rồi thử lại.
- **SHA-256 không khớp:** dùng đúng bản checkpoint Phase 2; không thay thế bằng trọng số khác mà không xác minh lại model, lớp và tiền xử lý.
- **Ảnh không đọc được:** kiểm tra định dạng và độ nguyên vẹn, chọn ảnh khác.
- **Camera bị chặn:** cấp quyền camera trong trình duyệt nếu bạn muốn chụp trực tiếp, hoặc dùng chức năng tải ảnh lên.
- **Không thể nạp model:** kiểm tra các phiên bản trong `requirements-app.txt` và nội dung lỗi; state_dict phải khớp nghiêm ngặt với MobileNetV3-Small năm đầu ra.

Ứng dụng chạy suy luận CPU. Các bộ dữ liệu Rice và Coffee chưa được hỗ trợ trong UI này.
