# 01 — Kiểm kê archive

Nguồn: `C:\Users\Admin\Downloads\data`. Kiểm kê ZIP bằng central directory, chưa giải nén ở giai đoạn này. Số liệu chính xác theo metadata của ZIP; dung lượng trống là ảnh chụp tại thời điểm kiểm kê.

| Archive | Dung lượng nén | File | Dung lượng giải nén | Tỷ số | Nội dung chính |
|---|---:|---:|---:|---:|---|
| A Rice Leaf Disease Image Dataset for Multi-Class.zip | 2.060 GB | 6.712 | 2.143 GB | 1,04 | 6.712 ảnh trong một thư mục gốc |
| cassava-leaf-disease-classification.zip | 6.186 GB | 21.418 | 6.185 GB | 1,00 | 21.398 JPEG, 17 TFRecord, 2 CSV, 1 JSON |
| RoCoLe A robusta coffee leaf images dataset.zip | 2.246 GB | 1.565 | 2.271 GB | 1,01 | 1.560 JPEG, CSV/JSON/XLSX và một TAR.GZ lồng bên trong |

Tổng dự kiến giải nén archive ngoài: **10.599 GB** (hệ thập phân). Ổ C còn **31.091 GB** trước khi giải nén; sau khi giải nén dự kiến còn khoảng **20,5 GB**. Trường hợp giải nén thêm TAR.GZ bên trong bộ coffee sẽ cần đánh giá riêng.

Kiểm tra tên thành viên: không phát hiện đường dẫn tuyệt đối, `..`, symlink, file đặc biệt, mục mã hóa, hoặc tên trùng (so sánh không phân biệt hoa/thường). Tỷ số nén cao nhất của một thành viên lần lượt là 2,9; 3,1; 5,4; chưa có dấu hiệu ZIP bomb. Chưa chạy bài kiểm tra CRC riêng; bộ giải nén Python xác minh CRC cho từng thành viên khi đọc.

Công cụ có sẵn: Python 3.13 `zipfile` và Windows `tar.exe`; không thấy 7-Zip/UnZip trong PATH. Ba tệp đều là ZIP hợp lệ theo `zipfile.is_zipfile`. Danh sách chi tiết, kích thước thành viên, đuôi file và metadata candidate nằm trong [archive_inventory.json](archive_inventory.json).

**Quyết định:** đủ điều kiện giải nén ZIP ngoài vào `data/extracted/<tên archive>` với kiểm tra đường dẫn và không ghi đè. Giữ nguyên các ZIP gốc. TAR.GZ lồng bên trong chưa nằm trong quyết định này.
