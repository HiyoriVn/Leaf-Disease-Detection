# 04 — Đánh giá chất lượng

## Phát hiện đã xác nhận

| Mức độ | Bộ | Tiêu chí | Bằng chứng | Hướng xử lý đề xuất |
|---|---|---|---|---|
| Cao | Coffee | Nhất quán nhãn | 8 ảnh có `classification` mâu thuẫn với `Leaf.state` sau quy về healthy/unhealthy. | Kiểm tra thủ công annotation gốc, xác lập thứ tự ưu tiên nguồn nhãn; không tự sửa. |
| Cao | Coffee | Duy nhất / nhãn | 67 cặp JPEG trùng SHA-256; 27 cặp có nhãn 6 lớp khác nhau, 1 cặp khác cả healthy/unhealthy. | Rà soát cặp trùng cùng ảnh và adjudicate nhãn trước khi chia tập. |
| Cao | Coffee | Rò rỉ dữ liệu | 1.560 ảnh thuộc 390 nhóm tên cây `C#P#`; 379 nhóm có nhiều nhãn. Chia ngẫu nhiên theo ảnh sẽ đặt ảnh cùng cây ở nhiều split. | Chia theo nhóm cây, kiểm tra chéo SHA-256 trước khi cố định split. |
| Cao | Cassava | Mất cân bằng lớp | CMD chiếm 13.158/21.397 = 61,5%; CBB chỉ 5,1%. | Stratify, báo macro-F1/recall từng lớp, dùng weighting/sampling chỉ trên train. |
| Trung bình | Coffee | Mất cân bằng lớp | rust_level_4 có 30 ảnh (1,9%), healthy 791 (50,7%). | Đánh giá khả năng giữ lớp hiếm trong các split; cân nhắc gom mức độ chỉ khi mục tiêu nghiên cứu cho phép. |
| Trung bình | Rice | Duy nhất | 1 cặp ảnh trùng byte trong `brown_spot` (`… - Copy.jpg` và bản gốc). | Chỉ giữ một bản trong tập huấn luyện sau khi quyết định chuẩn hóa. |
| Trung bình | Cả ba | Completeness cho đánh giá | Không có validation/test độc lập có nhãn đáng dùng: rice/coffee không có split; cassava chỉ 1 ảnh test và không có ground truth test. | Xây dựng split cố định theo nhóm nguồn, giữ test chỉ sử dụng một lần. |

## Tính hợp lệ và đầy đủ

Toàn bộ 29.670 file ảnh mở được header, đọc được byte và giải mã pixel đầy đủ, không ghi nhận lỗi. Ba ZIP đã được đọc hết khi giải nén nên CRC của thành viên được xác thực. Cassava không có giá trị thiếu ở `train.csv`, không trùng hàng/ID và không có ảnh train thiếu nhãn. Coffee không có ảnh thiếu annotation hoặc annotation thiếu ảnh theo `External ID` trong JSON. Rice chỉ có nhãn suy từ thư mục; thiếu manifest độc lập để kiểm định lớp thực tế của từng ảnh.

## Tính đại diện và khả năng sử dụng

Rice cân bằng tương đối nhưng có 339 kích thước ảnh và không có nguồn/nhóm cây để kiểm soát trùng gần. Cassava có kích thước đồng nhất, ánh xạ nhãn rõ ràng, song tập lệch mạnh về CMD và thiếu metadata nhóm/địa điểm. Coffee có polygon và phân loại chi tiết nhưng số mẫu lớp hiếm nhỏ và xung đột annotation. Chưa có bằng chứng về phạm vi mùa vụ, vùng trồng, giống cây, thiết bị và mức nhiễu ngoài đồng; **không thể kết luận tính đại diện ngoài thực địa**.

## Kiểm tra chưa thực hiện

Near-duplicate cần giải mã pixel của 29.670 ảnh và so perceptual hash/embedding theo ngưỡng có hiệu chỉnh; chi phí đọc và rà soát false positive lớn, nên chưa chạy ở vòng khảo sát này. Các ảnh trùng chính xác đã được phát hiện bằng SHA-256 toàn bộ. Chưa kiểm tra tính đúng của nhãn bằng chuyên gia bệnh cây, chưa parse TFRecord/VOC và chưa xác minh license. Những mục này là nghi vấn hoặc phạm vi chưa kiểm tra, không được xem là lỗi đã xác nhận.
