# 05 — Phân tích giữa ba bộ dữ liệu

| Đặc điểm | Rice | Cassava | Coffee |
|---|---|---|---|
| Cây trồng | Lúa | Sắn | Cà phê robusta |
| Mẫu có nhãn phân loại | 6.712 | 21.397 | 1.560 |
| Số lớp | 10 | 5 | 6 |
| Nguồn nhãn | Tên thư mục | CSV + JSON mapping | JSON Labelbox (`classification`) |
| Bài toán bổ sung | Không thấy annotation vùng | TFRecord đóng gói | Polygon lá / VOC |
| Rủi ro chính | Thiếu nhóm nguồn, kích thước không đồng nhất | CMD 61,5%, thiếu split | Nhãn mâu thuẫn, trùng byte, lớp hiếm, nhiều ảnh cùng cây |

Ba bộ cùng thuộc họ bài toán nhận diện tình trạng/bệnh cây từ ảnh, nhưng **không cùng không gian nhãn**. `Healthy` có thể quy về một khái niệm cấp cao, song bệnh của lúa, sắn và cà phê không thể ghép thành một lớp chung theo tên. Với mục tiêu hiện tại của repository (ba bộ phân loại theo từng cây), nên giữ ba nhánh dữ liệu và ba ánh xạ nhãn riêng. Một mô hình đa cây chỉ hợp lý khi định nghĩa lại taxon nhãn và giao thức đánh giá theo từng cây; chưa có bằng chứng để ghép trực tiếp.

Khác biệt về miền quan sát đã xác nhận ở kích thước ảnh: rice có 339 kích thước, cassava toàn bộ 800×600, coffee ba kích thước lớn. Khác biệt về cây trồng và nguồn annotation cũng rõ. Khác biệt ánh sáng, camera, địa điểm và mùa vụ chưa đo được. Không phát hiện SHA-256 trùng **giữa** ba bộ trên 29.670 ảnh; kết quả này không loại trừ ảnh trùng gần hoặc ảnh crop/resize. Trong mỗi bộ vẫn có rủi ro leakage, đặc biệt coffee có 67 cặp trùng byte và 390 nhóm cây cần giữ cùng split.

Có thể chuẩn hóa **cấu trúc quản lý** về manifest `dataset, image_path, crop, raw_label, canonical_label, group_id, split, source, checksum`, còn taxonomy bệnh và quy tắc đánh giá phải giữ riêng. TFRecord của cassava và VOC của coffee cần được xem là biểu diễn phụ, tránh đếm như mẫu mới cho đến khi đối chiếu nội dung.

**Ưu tiên cho nghiên cứu phân loại hiện tại:** cassava là điểm bắt đầu tốt nhất vì có 21.397 ảnh train, nhãn rõ ràng và ảnh đồng kích thước, sau khi xử lý mất cân bằng/split. Rice là nhánh thứ hai vì phân bố lớp tốt nhưng thiếu provenance và nhóm ảnh. Coffee là nhánh sau cùng cho phân loại 6 lớp vì 27 cặp ảnh trùng byte khác nhãn, 8 mâu thuẫn nhãn nhị phân và lớp rust_level_4 chỉ có 30 ảnh. Nếu nghiên cứu segmentation lá, coffee có annotation polygon và cần đánh giá riêng.
