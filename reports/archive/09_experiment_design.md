# 09 — Thiết kế thí nghiệm Phase 3

## Mục tiêu và giới hạn

Deadline 30/10/2026: ưu tiên một kết quả tái lập được với MobileNetV3-Small hoặc ResNet18 khởi tạo ImageNet, **không huấn luyện từ đầu**. Phase 2 chỉ làm manifest và smoke test; chưa fine-tune hoặc tính metric. Checkpoint cassava của bên thứ ba dùng cho demo kỹ thuật, không dùng `internal_test` mới chia để tuyên bố kiểm định độc lập vì checkpoint đã học từ tập Kaggle 21.397 ảnh.

## Thứ tự triển khai ngắn nhất

1. Chốt manifest v1 và review các cặp pHash có ảnh hưởng tới split. Giữ danh sách ảnh bị loại và quyết định nhãn riêng; không thay dữ liệu gốc.
2. Cassava: fine-tune head của MobileNetV3-Small ImageNet với 5 lớp và split stratified hiện có. Báo class counts, macro-F1, balanced accuracy, recall từng lớp và confusion matrix. So sánh class weight/sampler trên validation; không chấm lại `internal_test` trong quá trình lựa chọn.
3. Rice: fine-tune cùng backbone cho 10 lớp sau rà soát near-duplicate. Nếu chưa có plant ID, ghi rõ split chỉ kiểm soát exact/pHash, chưa bảo đảm độc lập lá/cây; có thể giới hạn tuyên bố ở internal split.
4. Coffee: chỉ fine-tune 6 lớp sau khi adjudicate 8 annotation mâu thuẫn và 27 cặp trùng ảnh khác nhãn. Group theo cây và nối nhóm trùng byte. Với `rust_level_4` chỉ có **3 ảnh ở validation và 3 ảnh ở internal test**, báo độ bất định và tránh khẳng định hiệu năng lớp hiếm. Nếu không kịp review, demo kỹ thuật tách khỏi kết quả khoa học.

## Giao thức đánh giá

Seed và tỷ lệ split trong [dataset_config.yaml](../configs/dataset_config.yaml). Lưu hash manifest và mã commit với mỗi lần chạy. Lựa chọn checkpoint theo macro-F1 validation; `internal_test` chỉ dùng một lần sau khi khóa kiến trúc, preprocessing, ngưỡng và chính sách class weight. Báo thêm accuracy, balanced accuracy, recall/F1 từng lớp, support, confusion matrix, latency đo trên máy thực tế, và số mẫu loại. Không so trực tiếp điểm tự báo cáo của checkpoint khác với điểm của nhóm khi split/dataset khác nhau.

Chỉ xem test là độc lập với checkpoint khi xác minh được ảnh/cây/nguồn không nằm trong dữ liệu train hoặc chọn mô hình của checkpoint đó. Local split rice/cassava/coffee hiện chưa đáp ứng điều này cho checkpoint disease-specific đã học từ cùng nguồn; muốn tuyên bố tổng quát hóa cần bộ ảnh ngoài nguồn, nhãn chuyên gia và quy trình thu thập rõ ràng. Với coffee, nhóm theo `C#P#` dựa trên tên file là proxy, cần xác minh ý nghĩa tên từ tác giả dữ liệu.

## Ngân sách và điều kiện dừng

GPU GTX 1650 Max-Q 4 GiB: thử batch nhỏ và mixed precision nếu môi trường CUDA hỗ trợ; dùng CPU cho smoke test hiện tại. Chưa benchmark thời gian fine-tune. Không cài thêm framework hoặc tải checkpoint lớn nếu không có lợi ích rõ ràng trước deadline. Nếu thiếu thời gian, hoàn thành một mô hình cassava có báo cáo đúng phạm vi trước, sau đó rice; không công bố metric coffee 6 lớp khi nhãn còn tranh chấp.
