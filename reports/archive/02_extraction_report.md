# 02 — Giải nén an toàn

Đã giải nén ba ZIP vào ba thư mục riêng dưới `C:\Users\Admin\Downloads\data\extracted`. Các ZIP gốc được giữ nguyên. Mỗi thành viên được kiểm tra đường dẫn trước khi ghi; tạo file theo chế độ không ghi đè, đọc bằng `zipfile` để kiểm tra CRC, rồi đối chiếu số byte của từng file. Không có lỗi được báo.

| Bộ dữ liệu | Đích dưới `data/extracted` | File thực tế | Byte thực tế | Trạng thái |
|---|---|---:|---:|---|
| Rice | `A Rice Leaf Disease Image Dataset for Multi-Class` | 6.712 | 2.142.647.017 | Hoàn tất, khớp archive |
| Cassava | `cassava-leaf-disease-classification` | 21.418 | 6.185.151.170 | Hoàn tất, khớp archive |
| Coffee | `RoCoLe A robusta coffee leaf images dataset` | 1.565 | 2.270.913.636 | Hoàn tất, khớp archive |

Ổ C còn **20.672 GB** tại lần kiểm tra sau giải nén. Dữ liệu được đọc tuần tự theo khối 1 MiB; không nạp toàn bộ archive vào RAM. [archive_inventory.json](archive_inventory.json) ghi kết quả từng archive.

Trong bộ coffee còn `Annotations/RoCoLe-voc.tar.gz` (697.591.506 byte), đã kiểm kê bằng stream: 3.121 thành viên gồm một thư mục, 1.560 XML và 1.560 JPEG, tổng nội dung file 700.315.790 byte; không phát hiện đường dẫn nguy hiểm hoặc link. Chưa giải nén TAR.GZ lồng này vì đây là một bản xuất annotation kèm bản JPEG bổ sung; dữ liệu ảnh gốc và annotation JSON/CSV đã đủ để khảo sát bài toán phân loại. Nếu cần sử dụng định dạng VOC cho segmentation, giải nén riêng sau khi xác nhận quy tắc ghép ảnh và annotation.

Lưu ý khả năng tiếp tục: script không xóa dữ liệu giải nén dở. Nếu bị ngắt, chạy lại với `--extract --resume`; script kiểm tra kích thước và CRC của từng file đã có rồi tiếp tục, từ chối file khác nội dung. Không có `--resume`, script từ chối thư mục đích đã có nội dung.
