# 03 — Hồ sơ dữ liệu

Phạm vi: ba thư mục đã giải nén. Số file, byte, nhãn, kích thước ảnh và SHA-256 là thống kê toàn bộ. Ảnh được đọc header, toàn bộ byte để băm, rồi giải mã pixel đầy đủ trong lượt kiểm tra riêng. **29.670/29.670 ảnh giải mã được, không có lỗi**. Chi tiết máy đọc được ở [dataset_profile.json](dataset_profile.json) và [data_validation.json](data_validation.json).

| Bộ | File / thư mục | Dung lượng | Ảnh | Kích thước ảnh | Nhãn / split |
|---|---:|---:|---:|---|---|
| Rice | 6.712 / 11 | 2,143 GB | 6.712 | 339 kích thước; phổ biến 1600×1600 (2.984), 1600×1200 (2.175), 3120×3120 (677) | 10 thư mục lớp; không có split |
| Cassava | 21.418 / 4 | 6,185 GB | 21.398 | 800×600 cho toàn bộ | `train.csv`: 21.397 ảnh train, 5 lớp; 1 ảnh test, không có validation |
| Coffee | 1.565 / 3 | 2,271 GB | 1.560 | 2048×1152 (768), 1280×720 (479), 4128×2322 (313) | JSON/CSV annotation: 1.560 ảnh, 6 lớp; không có split |

## Rice

Ảnh: 6.709 JPG, 2 JPEG, 1 PNG. Cấu trúc `A Rice Leaf Disease Image Dataset for Multi-Class/<tên lớp>/<ảnh>`. Không có README, metadata, annotation ngoài tên thư mục, hoặc thời gian/địa điểm/plant ID. Số ảnh theo lớp: bacterial_leaf_blight 785; brown_spot 761; Healthy 753; leaf_blast 521; leaf_scald 514; narrow_brown_spot 767; neck_blast 677; rice_hispa 770; sheath_blight 608; tungro 556. Tỷ lệ lớp từ 7,7% đến 11,7%.

## Cassava

`train_images` chứa 21.397 JPEG; `test_images` chứa 1 JPEG. `train.csv` có hai cột `image_id,label`, 21.397 hàng, không thiếu giá trị, không trùng hàng hoặc `image_id`. Mọi hàng đều có ảnh tương ứng, không có ảnh train thiếu nhãn. `label_num_to_disease_map.json` ánh xạ 0=Cassava Bacterial Blight, 1=Cassava Brown Streak Disease, 2=Cassava Green Mottle, 3=Cassava Mosaic Disease, 4=Healthy. Phân bố: 0: 1.087 (5,1%); 1: 2.189 (10,2%); 2: 2.386 (11,2%); 3: 13.158 (61,5%); 4: 2.577 (12,0%). Có 16 TFRecord train và 1 TFRecord test; chưa parse nội dung TFRecord, có khả năng là bản đóng gói của ảnh chứ không nên cộng vào số mẫu. `sample_submission.csv` có 1 hàng, không phải nhãn ground truth.

## Coffee

`Photos` chứa 1.560 JPEG. `Annotations` chứa JSON kiểu Labelbox, COCO JSON, CSV, XLSX và TAR.GZ kiểu VOC. JSON Labelbox có 1.560 hàng, `External ID` ghép khớp toàn bộ 1.560 ảnh, không thiếu hay trùng ID. Sáu nhãn `classification`: healthy 791 (50,7%); rust_level_1 344 (22,1%); rust_level_2 166 (10,6%); rust_level_3 62 (4,0%); rust_level_4 30 (1,9%); red_spider_mite 167 (10,7%). Trường `Leaf.state`: healthy 791, unhealthy 769. COCO JSON có 1.560 ảnh và 1.560 polygon lá; `categories` mô tả đối tượng Leaf, **không phải sáu nhãn bệnh**. CSV có 1.560 hàng, không trùng hàng hoặc `External ID`; các cột Agreement/Benchmark chủ yếu hoặc hoàn toàn trống, nhưng không phải nhãn mục tiêu.

TAR.GZ lồng bên trong chứa 1.560 XML và 1.560 JPEG của bản xuất VOC, chưa giải nén hoặc kiểm tra tính tương đương nội dung với `Photos`. Tên ảnh theo dạng `C…P…`, tạo 390 nhóm cây (`C#P#`) trong annotation; nhiều ảnh từ cùng nhóm.

## Các kiểu dữ liệu khác

Không thấy chuỗi thời gian hoặc dữ liệu không gian/viễn thám có CRS/band. Timestamp trong metadata Labelbox là thời điểm gán nhãn, không đủ căn cứ xem là thời gian thu ảnh. Không có tọa độ hoặc split nghiên cứu sẵn trong ba bộ. Chưa xác minh nguồn công bố, license và địa điểm thu thập từ nội dung các archive.
