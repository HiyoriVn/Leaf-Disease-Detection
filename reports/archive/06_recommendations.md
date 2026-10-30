# 06 — Khuyến nghị và mức sẵn sàng nghiên cứu

## Thứ tự ưu tiên

1. **Đóng băng nguồn và provenance.** Ghi phiên bản, URL nguồn, điều khoản sử dụng, ngày tải, SHA-256 của ba ZIP, và quy tắc nhãn được công bố. Các archive hiện không đủ metadata để xác nhận license/điều kiện nghiên cứu.
2. **Thiết kế split không rò rỉ.** Với coffee, dùng `C#P#` làm `group_id`, giữ mọi ảnh cùng cây và mọi cặp trùng SHA-256 cùng split. Với rice/cassava, tìm nhóm chụp từ tên file hoặc metadata nguồn; trước khi có nhóm đáng tin, chạy near-duplicate có kiểm soát và review mẫu. Chốt một test độc lập, validation chỉ phục vụ chọn mô hình.
3. **Adjudicate nhãn coffee.** Rà 8 xung đột `classification`/`Leaf.state`, 27 cặp trùng byte khác nhãn; lưu quyết định và ảnh hưởng đến split. Không sửa dữ liệu gốc, tạo manifest dẫn xuất khi bước preprocessing được phê duyệt.
4. **Kiểm soát phân bố lớp.** Cassava cần báo macro-F1, balanced accuracy và recall từng lớp; áp dụng class weighting hoặc sampling trong train sau split. Coffee cần đánh giá lại mục tiêu 6 lớp vì rust_level_4 chỉ có 30 ảnh, khó chia test đủ đại diện. Rice dùng stratification nhưng vẫn ưu tiên group split nếu có nhóm.
5. **Chuẩn hóa pipeline riêng cho từng cây.** Manifest dẫn xuất nên có `dataset, path, raw_label, label_index, group_id, split, sha256, source`. Giữ ba ánh xạ lớp riêng, thống nhất RGB/EXIF, resize và normalization trên pipeline; không biến đổi file nguồn. Đồng bộ manifest với `models/manifest.json` khi có checkpoint thực tế. Repo hiện để `labels: []` và chưa có checkpoint, nên chưa thể đánh giá mô hình.

## Mức sẵn sàng

| Bộ | Mức sẵn sàng cho bước thiết kế thí nghiệm | Điều kiện trước huấn luyện |
|---|---|---|
| Cassava | Cao nhất trong ba bộ | Xác nhận nguồn/license, split chống leakage, xử lý lệch lớp và xác định vai trò TFRecord |
| Rice | Trung bình | Xác nhận provenance/taxonomy, nhóm ảnh và split; kiểm tra near-duplicate |
| Coffee | Thấp cho phân loại 6 lớp | Giải quyết xung đột nhãn/trùng ảnh, split theo cây, quyết định xử lý lớp rất hiếm |

## Khả thi kỹ thuật trên máy hiện tại

Ba ZIP chiếm 10,491 GB; dữ liệu giải nén ngoài chiếm 10,599 GB; tổng bản gốc + bản giải nén khoảng 21,09 GB. Ổ C còn khoảng 20,67 GB sau giải nén. Nếu tạo bản dẫn xuất có ảnh sao chép đầy đủ cho cả ba, cần thêm khoảng 10,6 GB; nên dùng manifest tham chiếu tới ảnh gốc để giảm dung lượng. TAR.GZ coffee đã nằm trong dữ liệu giải nén, cần thêm khoảng 0,70 GB nếu mở tiếp. Tránh lưu nhiều phiên bản ảnh resize hoặc cache TFRecord không cần thiết.

Máy có 12 logical CPU và NVIDIA GeForce GTX 1650 Max-Q 4 GiB VRAM. Dung lượng RAM vật lý chưa xác nhận do công cụ hệ thống không trả về thông tin; kiểm tra trước khi chọn số worker/batch. Với GPU 4 GiB, bắt đầu mô hình nhỏ, ảnh 224×224, batch nhỏ và mixed precision nếu framework hỗ trợ; đây là đề xuất, **chưa chạy benchmark**. Pillow và Python `zipfile` sẵn có; pandas/numpy cũng hiện diện. PyTorch/TensorFlow và openpyxl chưa có trong Python hiện tại, không cần cài cho giai đoạn khảo sát. Nút thắt dự kiến là giải mã ảnh rice độ phân giải cao, đọc nhiều file nhỏ và kiểm tra trùng gần.

## Hướng dẫn chạy lại khảo sát

Chạy từ thư mục repository, thay `DATA_ROOT` và đường dẫn output theo môi trường:

```powershell
python scripts/inspect_archives.py --data-root DATA_ROOT --output reports/archive_inventory.json
python scripts/inspect_archives.py --data-root DATA_ROOT --output reports/archive_inventory.json --extract
# Nếu bị ngắt giữa chừng: thêm --resume; file đã có được xác thực CRC trước khi bỏ qua.
python scripts/inspect_datasets.py --data-root DATA_ROOT --output reports/dataset_profile.json
python scripts/validate_data.py --data-root DATA_ROOT --output reports/data_validation.json --decode-images
```

Không chạy lại `--extract` trên dữ liệu đã hoàn thành nếu không cần. Các script không chứa đường dẫn dữ liệu hard-code. Kết quả JSON là bằng chứng cho sáu báo cáo Markdown; các phép kiểm tra bị giới hạn được ghi rõ ở từng báo cáo.
