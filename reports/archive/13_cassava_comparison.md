# 13 — Cassava model comparison

## Cách đọc kết quả

Checkpoint công bố đã học từ cùng 21.397 ảnh Kaggle; mọi phép chấm của nó trên manifest Cassava local đều **có thể bị nhiễm dữ liệu**. Mô hình mới khởi tạo từ ImageNet và chỉ học trên `train` của manifest; validation dùng để chọn cấu hình/checkpoint, `internal_test` chỉ dùng sau khi khóa lựa chọn. Ngay cả internal test của mô hình mới vẫn là ảnh cùng nguồn Kaggle, không thay thế kiểm định ngoài thực địa.

## Kết quả đã quan sát

| Mô hình / tập | Accuracy | Macro-F1 | Balanced accuracy | Diễn giải |
|---|---:|---:|---:|---|
| Checkpoint công bố / 500 ảnh validation stratified | 0,8520 | 0,7898 | 0,8074 | Diagnostic cùng nguồn, có thể nhiễm |
| Checkpoint công bố / toàn bộ 3.210 ảnh validation | 0,8458 | 0,7688 | 0,7959 | Cùng tập validation để so sánh kỹ thuật, có thể nhiễm |
| ImageNet → Cassava, weighted CE / toàn bộ validation | 0,7477 | 0,5797 | 0,5833 | Best epoch 4; run dừng sau 7 epoch để ưu tiên đối chứng; chưa chấm internal test |
| ImageNet → Cassava, standard CE / toàn bộ validation | 0,8112 | 0,6877 | 0,6911 | Best epoch 7; 10 epoch đã chạy; chưa chấm internal test |
| ImageNet → Cassava, full backbone + label smoothing / toàn bộ validation | 0,8262 | 0,7044 | 0,7036 | Best epoch 6; được chọn trước test |
| ImageNet → Cassava, full backbone + label smoothing / 3.209 ảnh internal test | 0,8180 | 0,7011 | 0,7039 | Một lần test sau khi khóa checkpoint; cùng nguồn Kaggle, không phải external test |

Mô hình mới tốt nhất theo validation là run toàn backbone + label smoothing, nhưng **không vượt checkpoint công bố** trên cùng validation (macro-F1 0,7044 so với 0,7688). Checkpoint công bố có nguy cơ đã học từ chính các ảnh validation; vì thế chênh lệch này không đủ để xếp hạng khả năng tổng quát hóa ngoài nguồn. Mục tiêu 85% accuracy internal test của model mới không đạt (quan sát 81,8%). Streamlit giữ checkpoint công bố làm mặc định; model mới là lựa chọn thử nghiệm có ghi phạm vi metric. Không so trực tiếp metric tác giả tự báo cáo trên split khác với metric local.

## Kiểm tra ứng dụng sau tích hợp

Checkpoint mới nạp strict trong môi trường CPU của ứng dụng, nhận tensor `[1,3,224,224]` và trả `[1,5]`. 16 bài test đã qua, gồm kiểm tra tensor validation khớp adapter suy luận. Streamlit AppTest chọn được model mới, tải ảnh thật `1000015157.jpg`, render preview, bấm dự đoán ra CBB **40,3%** cùng ba thanh xác suất; không có exception. Chế độ không ảnh cho cảnh báo. Ứng dụng được khởi động lại tại `127.0.0.1:8501` và trả HTTP 200. Điều này kiểm tra luồng kỹ thuật; không phải metric đánh giá model.
