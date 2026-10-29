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

## EfficientNet-B3 trên RTX 3060 12 GB (thí nghiệm riêng)

`configs/cassava_efficientnet_b3.yaml` chọn `efficientnet_b3.ra2_in1k` của [timm trên Hugging Face](https://huggingface.co/timm/efficientnet_b3.ra2_in1k), khởi tạo từ ImageNet-1k. Đây là thí nghiệm mới, **không thay** config, checkpoint hay app MobileNetV3-Small. Input thí nghiệm là **300×300** (resize 343 rồi center crop 300, OpenCV cubic, ImageNet mean/std). Trọng số gốc được tác giả công bố với ảnh train 288 và test 320; 300 là lựa chọn thí nghiệm của nhóm, chưa có metric tác giả cho kích thước này. AMP bật khi CUDA hoạt động, batch CUDA mặc định **16**, head 3 epoch, hai block backbone cuối 10 epoch với early stopping 3 epoch, chọn `best.pth` theo validation macro-F1. `last.pth` dùng resume.

### 1. Trên máy nguồn: tạo bundle mới sau khi cập nhật code

```powershell
cd C:\Users\Admin\Downloads\Leaf-Disease-Detection
.\scripts\prepare_training_bundle.ps1 -Destination .\transfer
Get-FileHash .\manifests\cassava.csv -Algorithm SHA256
Get-FileHash .\checkpoints\imagenet_efficientnet_b3_ra2_in1k.safetensors -Algorithm SHA256
```

Chép **ba thành phần** vào ổ chuyển dữ liệu: (1) `Leaf-Disease-Detection-training.zip` mới tạo, (2) `checkpoints\imagenet_efficientnet_b3_ra2_in1k.safetensors` **49.335.454 byte**, (3) `C:\Users\Admin\Downloads\data\extracted\cassava-leaf-disease-classification\train_images` chứa **21.397 ảnh, khoảng 2,38 GiB**. Trong bundle đã có `manifests\cassava.csv`, `configs\cassava_efficientnet_b3.yaml`, các script, source, requirements và hướng dẫn này. Không cần chuyển `.venv-*`, các checkpoint Cassava cũ, ZIP gốc, `test_images` không nhãn, rice hay coffee để chạy thí nghiệm B3. Giữ nguyên cấu trúc `cassava-leaf-disease-classification\train_images` dưới `data\extracted`.

SHA-256 manifest cố định: `570b0d52ea950dec13de0eed792614b67c039b08b2cc28c8e588bb87b2f61082`. SHA-256 ImageNet B3: `279d2a53898aa89dab43fd6bd7df9f706aea4cb9cf916223988bbcb8e5850469`. Script từ chối chạy nếu hash weights hoặc số lượng split khác. Có thể dùng `robocopy` để chép ảnh (kiểm tra mã thoát 0–7 là thành công):

```powershell
$dst = 'E:\CassavaTransfer'
New-Item -ItemType Directory -Force -Path "$dst\checkpoints", "$dst\data\extracted\cassava-leaf-disease-classification\train_images" | Out-Null
Copy-Item .\transfer\Leaf-Disease-Detection-training.zip "$dst\"
Copy-Item .\checkpoints\imagenet_efficientnet_b3_ra2_in1k.safetensors "$dst\checkpoints\"
robocopy C:\Users\Admin\Downloads\data\extracted\cassava-leaf-disease-classification\train_images "$dst\data\extracted\cassava-leaf-disease-classification\train_images" /E /MT:8 /R:2 /W:2
if ($LASTEXITCODE -ge 8) { throw 'Copying Cassava images failed' }
```

Đổi `E:\CassavaTransfer` thành đường dẫn ổ chuyển thực tế nếu khác.

### 2. Trên PC RTX 3060: giải nén, cài và chạy pilot

Ví dụ ổ chuyển là `E:` và thư mục làm việc là `D:\CassavaWork`:

```powershell
$src = 'E:\CassavaTransfer'
$work = 'D:\CassavaWork'
New-Item -ItemType Directory -Force -Path "$work\Leaf-Disease-Detection", "$work\data\extracted\cassava-leaf-disease-classification\train_images" | Out-Null
Expand-Archive "$src\Leaf-Disease-Detection-training.zip" -DestinationPath "$work\Leaf-Disease-Detection" -Force
New-Item -ItemType Directory -Force -Path "$work\Leaf-Disease-Detection\checkpoints" | Out-Null
Copy-Item "$src\checkpoints\imagenet_efficientnet_b3_ra2_in1k.safetensors" "$work\Leaf-Disease-Detection\checkpoints\"
robocopy "$src\data\extracted\cassava-leaf-disease-classification\train_images" "$work\data\extracted\cassava-leaf-disease-classification\train_images" /E /MT:8 /R:2 /W:2
if ($LASTEXITCODE -ge 8) { throw 'Copying Cassava images failed' }
cd "$work\Leaf-Disease-Detection"
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_train.ps1 -Config configs\cassava_efficientnet_b3.yaml -VerifyWeights
.\scripts\run_train.ps1 -RunName cassava-b3-pilot -Config configs\cassava_efficientnet_b3.yaml -DataRoot "$work\data\extracted" -Pilot -RequireCuda
```

`setup_train.ps1` cài PyTorch 2.8 CUDA 12.8 vào `.venv-train` **trên PC mới**. Cần NVIDIA driver tương thích và mạng để cài wheel; lệnh `-VerifyWeights` kiểm tra SHA-256, strict state_dict và forward `[1,3,300,300] → [1,5]`. Pilot mặc định lấy 32 train + 16 validation mỗi lớp, một epoch head. Xem `checkpoints\experiments\cassava-b3-pilot\run.json`, `best.pth`, `last.pth` và `checkpoints\experiments\cassava-b3-pilot.console.log`. Điểm pilot chỉ xác minh vận hành, không dùng làm kết luận model.

### 3. Train chính, giảm batch nếu thiếu VRAM, resume

Chỉ chạy sau khi pilot thành công:

```powershell
.\scripts\run_train.ps1 -RunName cassava-b3-full -Config configs\cassava_efficientnet_b3.yaml -DataRoot "$work\data\extracted" -RequireCuda
```

Nếu CUDA báo thiếu VRAM **trước khi có checkpoint**, dùng run mới với `-BatchSize 8` (nếu cần thử 4). Không đổi batch giữa các lần resume cùng run vì script kiểm tra cấu hình. Trước khi giảm batch, có thể xem `gpu_peak_memory_bytes` trong `run.json` của pilot. Resume sau khi dừng:

```powershell
.\scripts\run_train.ps1 -RunName cassava-b3-full -Config configs\cassava_efficientnet_b3.yaml -DataRoot "$work\data\extracted" -Resume -RequireCuda
```

Muốn fine-tune **toàn bộ** backbone thay vì hai block cuối, dùng `-FineStage all` với **RunName khác** và chọn theo validation macro-F1. Cả pilot và train chính chỉ dùng `train`/`val` trong manifest; `internal_test` giữ ngoài lựa chọn model. **Sau khi khóa lựa chọn cuối cùng** trên validation, dùng evaluator đã mở rộng cho B3 đúng preprocessing 300×300 để đánh giá internal test đúng một lần vào thư mục output mới:

```powershell
.\.venv-train\Scripts\python.exe scripts\evaluate_cassava_new.py --checkpoint checkpoints\experiments\cassava-b3-full\best.pth --output reports\cassava_b3_internal_test --data-root "$work\data\extracted"
```

Không chạy lệnh này cho pilot hoặc để chọn batch, learning rate, epoch hay backbone. Evaluator từ chối ghi đè thư mục kết quả đã có.
