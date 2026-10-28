# 11 — Cassava checkpoint baseline, diagnostic in-source evaluation

Ngày chạy: 28/10/2026. Checkpoint bên thứ ba `ctibenkana/cassava-disease-classifier`, SHA-256 `4043a480d7a48d636f4feefbe31df4c52eeadd8edb8404d31319eddd807ac4bb`. Chạy bằng PyTorch 2.8.0 CPU và adapter Phase 3A. **Đây là đánh giá chẩn đoán có thể bị nhiễm dữ liệu**: checkpoint tác giả đã huấn luyện trên chính nguồn Kaggle Cassava 21.397 ảnh. Split mới trong manifest không làm các ảnh này độc lập với checkpoint.

## Ảnh 1000015157.jpg

`train.csv` gốc ghi label `0`; manifest ghi `Cassava Bacterial Blight (CBB)`, split `train`. Checkpoint dự đoán CBB. Phân phối softmax trên đủ năm lớp:

| Lớp | Xác suất |
|---|---:|
| CBB | 0.30927357 |
| CBSD | 0.16383077 |
| CGM | 0.25369743 |
| CMD | 0.01487155 |
| Healthy | 0.25832659 |

Độ tin cậy 30,9% của một ảnh không dùng để suy ra accuracy của mô hình.

## Mẫu chẩn đoán validation

Lệnh: `.\.venv-phase2\Scripts\python.exe scripts\evaluate_cassava_baseline.py --split val --limit 500`. Chọn stratified ngẫu nhiên cố định seed `20261028` từ 3.210 ảnh validation, giữ xấp xỉ tỷ lệ lớp, không dùng ảnh test không nhãn. Manifest SHA-256: `570b0d52ea950dec13de0eed792614b67c039b08b2cc28c8e588bb87b2f61082`. Thời gian suy luận và đọc ảnh: 21,08 giây.

| Metric | Giá trị quan sát |
|---|---:|
| Accuracy | 0.8520 |
| Macro-F1 | 0.7898 |
| Balanced accuracy | 0.8074 |

| Lớp | n | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| CBB | 25 | 0.8261 | 0.7600 | 0.7917 |
| CBSD | 51 | 0.7321 | 0.8039 | 0.7664 |
| CGM | 56 | 0.7925 | 0.7500 | 0.7706 |
| CMD | 308 | 0.9648 | 0.8896 | 0.9257 |
| Healthy | 60 | 0.5952 | 0.8333 | 0.6944 |

Confusion matrix: hàng là nhãn thật, cột là dự đoán; thứ tự CBB, CBSD, CGM, CMD, Healthy.

| Thật \\ Dự đoán | CBB | CBSD | CGM | CMD | Healthy |
|---|---:|---:|---:|---:|---:|
| CBB | 19 | 1 | 0 | 0 | 5 |
| CBSD | 0 | 41 | 2 | 4 | 4 |
| CGM | 0 | 0 | 42 | 6 | 8 |
| CMD | 2 | 8 | 7 | 274 | 17 |
| Healthy | 2 | 6 | 2 | 0 | 50 |

Chi tiết máy đọc được: [metrics.json](baseline_cassava/metrics.json) và [predictions.csv](baseline_cassava/predictions.csv). Các con số này chỉ cho biết hành vi checkpoint trên mẫu cùng nguồn và không được so trực tiếp với internal-test của mô hình mới như hai phép đo tổng quát hóa tương đương.

## Toàn bộ validation để so sánh cùng tập ảnh

Chạy lại cùng script với `--limit 3210 --output reports\baseline_cassava_full_val`, bao gồm đúng 3.210 ảnh validation. Kết quả: **accuracy 0,8458; macro-F1 0,7688; balanced accuracy 0,7959** (103,81 giây CPU). Recall CBB/CBSD/CGM/CMD/Healthy lần lượt là **0,8160 / 0,7927 / 0,6648 / 0,8972 / 0,8088**. [JSON đầy đủ](baseline_cassava_full_val/metrics.json) và [dự đoán từng ảnh](baseline_cassava_full_val/predictions.csv) cho phép so sánh sau này trên cùng các ảnh. Rủi ro nhiễm nguồn huấn luyện của checkpoint cũ vẫn giữ nguyên.
