# 12 — Cassava fine-tuning from ImageNet weights

## Nguồn và kiểm soát dữ liệu

Mô hình `mobilenetv3_small_100` được khởi tạo từ trọng số ImageNet của [timm/mobilenetv3_small_100.lamb_in1k](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k), file `checkpoints/imagenet_mobilenetv3_small_100.safetensors` SHA-256 `46d2c063b18125884c48937afa4c49e18128869e52e8db96df48bf0a4d7ff697`. Trọng số nạp strict vào model 1.000 lớp, sau đó thay classifier bằng head 5 lớp. Không khởi tạo từ checkpoint Cassava của bên thứ ba.

Manifest Cassava SHA-256 `570b0d52ea950dec13de0eed792614b67c039b08b2cc28c8e588bb87b2f61082` giữ nguyên: train 14.978, validation 3.210, internal test 3.209. Kiểm tra group ID và SHA-256 không đi qua nhiều split. Không có plant ID để loại trừ toàn bộ nguy cơ cùng cây/ảnh gần trùng. Script huấn luyện chỉ đọc ảnh train và validation; internal test dành cho lần đánh giá cuối sau khi khóa lựa chọn.

## Môi trường và giao thức

GPU: NVIDIA GeForce GTX 1650 with Max-Q Design, 4.096 MiB VRAM; driver 610.62. Môi trường riêng `.venv-train`: Python 3.13, PyTorch 2.8.0+cu128, timm 1.0.20, OpenCV 4.12.0, NumPy 2.2.6. `torch.cuda.is_available()` trả về `True`, forward shape `[1,5]` thành công. Môi trường Streamlit `.venv-phase2` không bị thay đổi.

Seed 20261028. Ảnh RGB, resize OpenCV `INTER_AREA` về 256×256, crop tâm 224×224, float32 và ImageNet normalization. Kiểm thử xác nhận tensor validation khớp **chính xác** adapter của ứng dụng. Train có flip ngang/dọc, xoay bội số 90°, brightness/contrast ±10% trên ảnh đã resize; validation/test không augmentation. Weighted cross entropy dùng trọng số nghịch tần suất train theo `N/(5*n_class)`. Head-only trước, có thể mở hai stage cuối; AdamW, mixed precision CUDA, early stopping patience 3; chọn checkpoint bằng validation macro-F1. Cấu hình đầy đủ ở `configs/cassava_training.yaml`.

## Pilot đã chạy

Pilot CPU: 320 train + 160 validation (64/32 mỗi lớp), 1 epoch head-only; validation accuracy 0,3688, macro-F1 0,3171, 75,05 giây. Pilot GPU cùng kích thước: validation accuracy 0,3563, macro-F1 0,2711, 52,15 giây; batch 16, PyTorch peak allocated ~59,9 MiB. Đây là kiểm tra đường chạy, không dùng để chọn model cuối hoặc so với baseline đủ dữ liệu.

## Full run

Thí nghiệm weighted ban đầu ở `checkpoints/experiments/weighted_full_20261028/run.json` đã dừng sau 2 epoch vì preprocessing ảnh gốc khiến mỗi epoch mất ~350 giây; validation macro-F1 quan sát 0,5187 rồi 0,5700. Chỉ các epoch hoàn thành được lưu trong log, checkpoint không được chọn làm model cuối. Sau khi chuyển augmentation sang ảnh 256px, thời gian xử lý 100 ảnh train tuần tự giảm từ ~0,057 xuống ~0,025 giây/ảnh. Run weighted mới với 4 workers tại `checkpoints/experiments/weighted_full_fast_20261028/run.json` đã hoàn thành 6 epoch head-only và 1 epoch mở hai stage cuối. Best validation macro-F1 **0,5797** ở epoch 4; epoch 7 giảm còn 0,5544. Nhánh này được dừng để ưu tiên đối chứng standard cross entropy, chỉ dựa trên validation. Các run dừng sớm không có số đo internal test. Internal test chưa được mở.

Đối chứng standard cross entropy ở `checkpoints/experiments/standard_full_20261028/run.json` hoàn tất 4 epoch head-only + 6 epoch mở hai stage cuối (1.966,29 giây). Best epoch **7** có validation macro-F1 **0,68775**, accuracy **0,81121**, balanced accuracy **0,69111**. SHA-256 checkpoint tốt nhất: `2a221ebc6174964df6296d291ee7d8ead1e489e03b2e09eccbebc2e4a0cafdd3`. Train metric còn tăng sau epoch 7 nhưng validation giảm, nên giữ checkpoint epoch 7 theo giao thức.

Vì cả weighted và standard vẫn dưới baseline cùng tập validation, thí nghiệm bổ sung bắt đầu từ ImageNet: 2 epoch head-only, 6 epoch mở toàn bộ backbone, learning rate `3e-5`, standard CE với label smoothing 0,1. Run `checkpoints/experiments/full_backbone_smooth_20261028/` hoàn tất **8 epoch trong 1.358,46 giây** trên GTX 1650 Max-Q; PyTorch peak allocated **310,23 MiB**. Best epoch **6** theo validation macro-F1 **0,70441**, accuracy **0,82617**, balanced accuracy **0,70355**. Checkpoint SHA-256 `7cb54b7de90ec775eb346a0a5ade31dfcdac7fff2e03db5264c1aa8c6fc98cd0`.

Lệnh run được chọn: `.\.venv-train\Scripts\python.exe scripts\train_cassava.py --run-name full_backbone_smooth_20261028 --loss standard --head-epochs 2 --last-stage-epochs 6 --fine-stage all --fine-lr 0.00003 --label-smoothing 0.1`. Lệnh đánh giá một lần: `.\.venv-train\Scripts\python.exe scripts\evaluate_cassava_new.py --checkpoint checkpoints\experiments\full_backbone_smooth_20261028\best.pth --output reports\cassava_internal_test_20261028`. Script từ chối chạy lại nếu thư mục kết quả đã tồn tại.

Checkpoint được khóa sau khi so validation giữa các nhánh; **chỉ checkpoint best epoch 6** được chấm trên 3.209 ảnh internal test đúng một lần bằng `scripts/evaluate_cassava_new.py`. Kết quả thực tế: accuracy **0,81801**, macro-F1 **0,70112**, balanced accuracy **0,70386**; thời gian suy luận test **38,75 giây** trên GPU. [Metrics JSON](cassava_internal_test_20261028/metrics.json) và [dự đoán từng ảnh](cassava_internal_test_20261028/predictions.csv) là artifact của lần chạy này. Mục tiêu 85% accuracy **chưa đạt**. Không có phép đánh giá ngoài nguồn.

| Lớp | n test | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| CBB | 163 | 0,5729 | 0,6994 | 0,6298 |
| CBSD | 329 | 0,7118 | 0,6231 | 0,6645 |
| CGM | 358 | 0,6504 | 0,6704 | 0,6602 |
| CMD | 1.973 | 0,9148 | 0,9306 | 0,9226 |
| Healthy | 386 | 0,6647 | 0,5959 | 0,6284 |

Confusion matrix test: hàng = nhãn thật, cột = dự đoán; thứ tự CBB, CBSD, CGM, CMD, Healthy.

| Thật \\ Dự đoán | CBB | CBSD | CGM | CMD | Healthy |
|---|---:|---:|---:|---:|---:|
| CBB | 114 | 12 | 4 | 7 | 26 |
| CBSD | 27 | 205 | 20 | 51 | 26 |
| CGM | 12 | 12 | 240 | 62 | 32 |
| CMD | 11 | 26 | 68 | 1836 | 32 |
| Healthy | 35 | 33 | 37 | 51 | 230 |

Không có SHA-256 hoặc `group_id` xuất hiện ở nhiều split trong manifest. Điều đó không chứng minh không có ảnh gần trùng/cùng cây khác file; chưa có plant ID và bộ ảnh độc lập nên kết quả chỉ phản ánh **internal test cùng nguồn Kaggle**.

Log theo epoch của các run đã được chép vào [training_cassava/](training_cassava/) gồm hai pilot, hai run weighted dừng sớm, run standard và run toàn backbone được chọn. File checkpoint `.pth` và môi trường ảo không commit; cần giữ riêng file checkpoint có hash nêu trên để tái hiện ứng dụng.
