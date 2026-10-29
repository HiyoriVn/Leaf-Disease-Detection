# Cassava Leaf Disease Detection — Capstone

Ứng dụng Streamlit phân loại **lá sắn** bằng checkpoint MobileNetV3-Small đã kiểm chứng ở Phase 2. Ứng dụng hỗ trợ tải ảnh hoặc chụp bằng camera, xem ảnh, dự đoán một trong năm lớp của mô hình mặc định, độ tin cậy softmax và ba khả năng cao nhất. CropNet là lựa chọn thử nghiệm riêng với sáu lớp khi đã cài. Kết quả là dự đoán AI, không phải chẩn đoán nông nghiệp đã xác nhận.

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

Trong **Chọn mô hình Cassava**, checkpoint công bố vẫn là mặc định. Model do nhóm fine-tune từ ImageNet có thể chọn để thử nghiệm; checkpoint của nó nằm tại `checkpoints/experiments/full_backbone_smooth_20261028/best.pth`. Nếu chuyển máy, cần chuyển riêng hai file checkpoint (chúng được Git bỏ qua) và đối chiếu SHA-256 trong `configs/cassava_app.json` / `configs/cassava_finetuned_app.json`.

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

## Phase 3B: huấn luyện và đánh giá Cassava

Đã fine-tune MobileNetV3-Small khởi tạo từ ImageNet trên manifest cố định. Model tốt nhất theo validation macro-F1 đạt **81,80% accuracy, 0,7011 macro-F1** trên 3.209 ảnh internal test cùng nguồn Kaggle; mục tiêu 85% chưa đạt. Kết quả này không chứng minh hiệu năng trên ảnh thực địa độc lập. Xem [baseline](reports/11_cassava_baseline_evaluation.md), [huấn luyện](reports/12_cassava_finetuning.md) và [so sánh](reports/13_cassava_comparison.md).

Để tái lập trên máy có GPU NVIDIA và đủ dung lượng (wheel CUDA đã tải khoảng 3,5 GB), tạo `.venv-train` riêng. Trên máy này đã dùng các lệnh sau, theo [PyTorch CUDA 12.8](https://pytorch.org/get-started/previous-versions/):

```powershell
cd C:\Users\Admin\Downloads\Leaf-Disease-Detection
py -3.13 -m venv .venv-train
.\.venv-train\Scripts\Activate.ps1
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-train-extra.txt
```

Đặt ImageNet `model.safetensors` từ [timm](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k/tree/main) tại `checkpoints/imagenet_mobilenetv3_small_100.safetensors` và kiểm tra hash trong `configs/cassava_training.yaml`. Lệnh thí nghiệm được chọn:

```powershell
.\.venv-train\Scripts\python.exe scripts\train_cassava.py --run-name my-full-backbone --loss standard --head-epochs 2 --last-stage-epochs 6 --fine-stage all --fine-lr 0.00003 --label-smoothing 0.1
```

Script huấn luyện chỉ đọc ảnh train/validation. Không chạy `scripts/evaluate_cassava_new.py` lặp lại trên test để thử hyperparameter; lần đánh giá đã hoàn tất nằm trong `reports/cassava_internal_test_20261028/`.

Để chạy độc lập trên PC Windows NVIDIA, xem [hướng dẫn chuyển repo/dataset, cài CUDA, logging và resume](docs/WINDOWS_GPU_TRAINING.md). PowerShell helpers: `scripts/setup_train.ps1`, `scripts/prepare_training_bundle.ps1`, `scripts/run_train.ps1`.

### EfficientNet-B3 trên RTX 3060 (thí nghiệm mới)

Config riêng `configs/cassava_efficientnet_b3.yaml` dùng ImageNet pretrained `efficientnet_b3.ra2_in1k`, ảnh 300×300, AMP và batch CUDA khởi đầu 16. Trọng số safetensors 49.335.454 byte phải nằm ở `checkpoints/imagenet_efficientnet_b3_ra2_in1k.safetensors`; SHA-256 được kiểm tra trước khi train. Pilot kỹ thuật ban đầu đã chạy; sau đó full training B3 hoàn thành trên PC. Xem [báo cáo chuẩn bị](reports/15_efficientnet_b3_preparation.md) và [lệnh chuyển sang PC RTX 3060](docs/WINDOWS_GPU_TRAINING.md#efficientnet-b3-trên-rtx-3060-12-gb-thí-nghiệm-riêng). MobileNetV3-Small vẫn dùng config mặc định và checkpoint cũ.

Checkpoint B3 đã hoàn tất trên PC và được tích hợp để chọn trong ứng dụng. Validation GPU trên đủ 3.210 ảnh đã được trả về và audit: accuracy **83,12%**, macro-F1 **0,7052**, balanced accuracy **0,7001**; có 542 dự đoán sai. Đây là validation cùng nguồn Kaggle, không phải external test. Xem [báo cáo tích hợp](reports/16_efficientnet_b3_integration.md), [so sánh và lỗi theo lớp](reports/17_cassava_model_comparison.md) và artifact audit `reports/b3_validation_archive_audit.json`.

V2 (tempered weighted CE) và V3 (label smoothing 0.05) đã train trên PC; V3 hiện cao nhất theo metric tổng hợp validation nhưng per-class comparison đang chờ predictions từ PC. Xem [Phase 3E results](reports/19_phase3e_results.md) và [GPU evaluation handoff](docs/PHASE3E_GPU_EVALUATION.md). Chưa dùng internal test để chọn model.

Phase 3F diagnosis, preprocessing visual audit, configs, TTA/ensemble scripts và RTX 3060 plan: [report 20](reports/20_phase3f_diagnosis.md), [visual audit sheet](reports/phase3f_preprocessing_audit.png), [GPU plan](docs/PHASE3F_GPU_PLAN.md). Không có thí nghiệm Phase 3F nào tự chạy và Streamlit default chưa đổi.

Trạng thái tích hợp và so sánh: [report 16](reports/16_efficientnet_b3_integration.md), [report 17](reports/17_cassava_model_comparison.md). Tạo source-only bundle bằng `scripts/prepare_phase3_bundle.ps1`; dùng [Phase 3E GPU guide](docs/PHASE3E_GPU_EVALUATION.md) để chạy đánh giá trên PC.

## CropNet (Phase 3C, optional)

Google CropNet is available as a separate **exploratory** six-class option in the Cassava Streamlit app when `.venv-cropnet` and `checkpoints/cropnet` exist. Its `CGM` output means **Cassava Green Mite**; the local Kaggle 2020 label means **Cassava Green Mottle**. They have not been mapped to each other, so no local five-class accuracy is claimed. The two PyTorch choices and default remain unchanged. See [the evaluation report](reports/14_cropnet_evaluation.md).

To reproduce on Windows with Python 3.13:

```powershell
py -3.13 -m venv .venv-cropnet
.\.venv-cropnet\Scripts\python.exe -m pip install -r requirements-cropnet.txt
.\.venv-cropnet\Scripts\python.exe -c "import kagglehub; kagglehub.model_download('google/cropnet/tensorFlow2/classifier-cassava-disease-v1/1', output_dir='checkpoints/cropnet')"
.\.venv-cropnet\Scripts\python.exe scripts\smoke_cropnet.py
```

The app invokes CropNet through this separate interpreter; the first CropNet prediction can take several seconds while TensorFlow starts.
