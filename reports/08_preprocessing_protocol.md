# 08 — Giao thức manifest và chuẩn bị dữ liệu

## Đầu vào và tính bất biến

Ảnh nguồn tại `C:\Users\Admin\Downloads\data\extracted` và ba ZIP không bị sửa. [prepare_manifests.py](../scripts/prepare_manifests.py) tạo CSV dẫn xuất trong `manifests/`, mỗi hàng chỉ tham chiếu đường dẫn ảnh tương đối từ dataset root và SHA-256; không sao chép ảnh. Cấu hình cố định ở [dataset_config.yaml](../configs/dataset_config.yaml): seed **20261028**, tỷ lệ mục tiêu 70/15/15 (`train`, `val`, `internal_test`). `internal_test` là tập nội bộ mới chia, **không phải test độc lập với checkpoint học trên cùng nguồn**. Mọi biến đổi ảnh chỉ nên thực hiện lúc nạp trong Phase 3.

## Schema

Mỗi manifest có `dataset, image_path, raw_label, canonical_label, label_index, group_id, split, sha256, source, exclusion_reason`. `image_path` tương đối với `data_root` trong config; `source` là tên ZIP ban đầu. `raw_label` giữ nhãn gốc; `canonical_label` chỉ chuẩn hóa chữ hoa của `Healthy` rice thành `healthy`, không gộp bệnh khác nhau. `label_index` cassava giữ 0–4 của `label_num_to_disease_map.json`; rice dùng thứ tự alphabet của tên lớp chuẩn hóa; coffee dùng `[healthy, red_spider_mite, rust_level_1, rust_level_2, rust_level_3, rust_level_4]`. Khi xuất model mới, phải lưu đúng thứ tự này cùng checkpoint.

## Quy tắc theo bộ

| Bộ | Quy tắc đã áp dụng | Tình trạng cần xử lý |
|---|---|---|
| Rice | Một bản của cặp trùng SHA-256 bị đánh dấu `exact_duplicate` và không vào split sử dụng. pHash 64-bit phát hiện cặp nghi vấn (Hamming ≤6); mọi component nghi vấn được giữ cùng group/split một cách bảo thủ, **không tự xóa**. | Các cặp pHash cần người xem ảnh; nhóm chụp thật chưa có metadata. |
| Cassava | 21.397 ảnh train được chia stratified theo nhãn 0–4. Một ảnh trong `test_images` được ghi `unlabeled`, `exclusion_reason=unlabeled_test_no_metrics`. SHA-256 toàn bộ không thấy trùng byte trong corpus ảnh. | Không có plant ID/site ID; nguy cơ ảnh gần trùng chưa được loại trừ; CMD chiếm 61,5%. |
| Coffee | Group ban đầu lấy `C#P#` trong tên ảnh; các group có cùng SHA-256 được nối thành một component trước khi chia. 8 ảnh mâu thuẫn `Leaf.state`/`classification` và cả hai ảnh ở 27 cặp trùng byte khác nhãn vào `excluded`; một ảnh trong mỗi 40 cặp trùng cùng nhãn vào `excluded`. | Tổng 102 ảnh cần tránh dùng trực tiếp; 8 trường hợp mâu thuẫn và 27 cặp khác nhãn phải adjudicate; rust_level_4 còn rất ít mẫu mỗi split. |

Danh sách để rà soát: [coffee_label_disputes.csv](../manifests/coffee_label_disputes.csv), [coffee_exact_duplicates.csv](../manifests/coffee_exact_duplicates.csv), [rice_near_duplicates.csv](../manifests/rice_near_duplicates.csv). Ngưỡng pHash chỉ tạo nghi vấn; hai ảnh có cùng nội dung bệnh nhưng góc chụp/ánh sáng khác vẫn có thể không được phát hiện. Không dùng pHash để khẳng định nhãn sai.

## Kết quả manifest v1

| Bộ | Train | Val | Internal test | Ngoài split sử dụng |
|---|---:|---:|---:|---:|
| Rice | 4.766 | 973 | 972 | 1 trùng byte |
| Cassava | 14.978 | 3.210 | 3.209 | 1 ảnh test không nhãn |
| Coffee | 1.060 | 200 | 198 | 102: 8 mâu thuẫn state, 54 ảnh thuộc cặp trùng khác nhãn, 40 ảnh trùng cùng nhãn |

Rice có **1.344 cặp pHash Hamming ≤6**, trong đó 411 cặp khác tên lớp. Cả 1.344 cặp đã được giữ cùng split theo component; không thêm ảnh nào vào danh sách loại vì pHash. [50 cặp ưu tiên xem](../manifests/rice_near_duplicates_priority.csv) được sắp theo khoảng cách thấp và nhãn khác nhau. Coffee sau loại còn `rust_level_4` 18/3/3 ở train/val/internal test, quá ít để ước lượng ổn định.

## Tiền xử lý đề xuất cho mô hình

Ảnh nạp theo RGB, áp dụng EXIF transpose, sau đó resize/crop đúng giao thức của checkpoint. Smoke test MobileNetV3 cassava mô phỏng đúng mã Space: OpenCV `INTER_AREA` resize về **256×256**, crop tâm **224×224**, chia 255, chuẩn hóa mean `[0.485,0.456,0.406]` và std `[0.229,0.224,0.225]`, float32 NCHW. Lưu ý resize vuông làm thay đổi tỷ lệ khung hình 800×600; khi fine-tune có thể thử bảo toàn tỷ lệ, nhưng phải giữ cùng một giao thức giữa train và inference. Rice và coffee chưa có checkpoint 10/6 lớp tương thích nên **chưa chốt preprocessing mô hình**; đề xuất khởi tạo 224×224 với transform ImageNet của backbone, chỉ tính thống kê dataset nếu thí nghiệm yêu cầu.

Augmentation chỉ áp dụng train sau khi split, không lưu ảnh đã biến đổi vào nguồn. Mọi normalization, sampling, class weight phải được quyết định từ train/validation, không dùng `internal_test` để chọn cấu hình. Không sửa nhãn coffee cho đến khi có quyết định rà soát có log. Tập test từ nguồn khác cần được lập riêng nếu muốn khẳng định tổng quát hóa.

## Kiểm tra tính nhất quán

CSV manifest có đủ đường dẫn và SHA-256 cho từng ảnh nguồn. Kiểm tra tự động xác nhận không có SHA-256 hay `group_id` của hàng sử dụng xuất hiện ở nhiều split trong mỗi dataset. Kết quả split chi tiết ở [summary.json](../manifests/summary.json); các ảnh `excluded` và `unlabeled` không được tính metric.
