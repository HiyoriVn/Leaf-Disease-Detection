# Chạy huấn luyện Cassava trên PC Windows có NVIDIA GPU

Quy trình dùng MobileNetV3-Small, manifest đã cố định và ImageNet weights có SHA-256 trong `configs/cassava_training.yaml`. Script chỉ đọc split `train` và `val`; split `internal_test` không được mở ảnh hay dùng chọn checkpoint. Checkpoint tốt nhất theo validation macro-F1 được ghi thành `best.pth`; `last.pth` giữ trạng thái epoch gần nhất để tiếp tục.

## Chuyển repo và dữ liệu

Trên máy hiện tại, tạo bundle source (không chứa venv, ảnh hay checkpoint):

```powershell
cd C:\Users\Admin\Downloads\Leaf-Disease-Detection
.\scripts\prepare_training_bundle.ps1 -Destination E:\cassava-transfer
```

Chép `Leaf-Disease-Detection-training.zip` tới PC RTX 3060, giải nén, rồi chép dataset đã giải nén. Cấu trúc hiện tại mong đợi là thư mục `extracted` chứa ba bộ dữ liệu; vì vậy có thể đặt tại `D:\cassava\data\extracted`. Trên Windows có ổ đĩa chung, ví dụ:

```powershell
robocopy C:\Users\Admin\Downloads\data\extracted D:\cassava\data\extracted /E /MT:8 /R:2 /W:2
```

`robocopy` mã thoát 0–7 thường biểu thị hoàn tất (có thể có file được chép); 8 trở lên là lỗi cần xử lý. Dataset có thể lớn, nên kiểm tra dung lượng đĩa trước và tránh chép các file ZIP gốc nếu đã có cây `extracted`.

Chép riêng ImageNet weights `imagenet_mobilenetv3_small_100.safetensors` vào `checkpoints\imagenet_mobilenetv3_small_100.safetensors`. Script xác minh SHA-256 trước khi khởi tạo. Không cần chép model bệnh đã fine-tune hiện tại để train lần mới. Giữ nguyên `manifests\cassava.csv` trong bundle để split không thay đổi.

## Cài và chạy

RTX 3060 12 GB đủ cho MobileNetV3-Small; script chọn CUDA tự động nếu PyTorch CUDA hoạt động. Sau khi cài NVIDIA driver phù hợp, từ thư mục repo trên PC:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_train.ps1
.\scripts\run_train.ps1 -RunName cassava-mnv3-3060 -DataRoot D:\cassava\data\extracted
```

Đầu tiên có thể chạy pilot nhỏ để kiểm tra đường dẫn, CUDA và pipeline:

```powershell
.\scripts\run_train.ps1 -RunName cassava-pilot -DataRoot D:\cassava\data\extracted -Pilot
```

Pilot tạo checkpoint riêng và không nên dùng làm kết quả cuối. Lệnh train đầy đủ dùng mặc định trong config; có thể truyền cờ như `--head-epochs`, `--last-stage-epochs`, `--fine-stage`, `--fine-lr` trực tiếp cho Python nếu cần điều chỉnh có chủ đích.

`console.log` lưu output PowerShell, `epochs.jsonl` lưu một JSON record mỗi epoch, `run.json` lưu cấu hình/metrics/provenance, còn `best.pth` và `last.pth` là checkpoint. Nếu máy tắt hoặc job dừng, chạy lại cùng tên run với `-Resume`:

```powershell
.\scripts\run_train.ps1 -RunName cassava-mnv3-3060 -DataRoot D:\cassava\data\extracted -Resume
```

Resume yêu cầu cùng manifest và cùng ImageNet weights. Chạy thường và resume không được dùng chung cho hai thí nghiệm khác nhau; tạo `RunName` mới khi đổi cấu hình.

## Chuyển file không cần mang Codex sang máy

Bundle, dữ liệu và weights chỉ cần được chép một lần. Sau đó PC chạy độc lập bằng PowerShell, kể cả khi không có mạng nếu Python wheels và môi trường đã cài xong. Phương án SSH là tùy chọn: bật OpenSSH Server trên PC Windows, rồi từ máy điều khiển dùng `scp` để chép bundle/dữ liệu và `ssh user@PC ...` để gọi PowerShell runner. Không nên chạy training qua phiên SSH dễ mất kết nối; dùng `Start-Process`/Task Scheduler hoặc để terminal PC chạy trực tiếp, rồi SSH vào xem `console.log`. Mặc định quy trình này không cấu hình firewall, tài khoản hay quyền truy cập từ xa.

Sau khi chọn checkpoint trên validation, chỉ chạy đánh giá internal test theo quy trình đã ghi trong báo cáo Phase 3B; không dùng test để chọn hyperparameter.
