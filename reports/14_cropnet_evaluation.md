# 14 — Google CropNet Cassava, Phase 3C (29/10/2026)

## Kết luận

**CropNet chạy được để suy luận, nhưng chưa có cơ sở khẳng định cải thiện hiệu năng phân loại Cassava local.** Đã thêm như lựa chọn *thử nghiệm* trong Streamlit, giữ checkpoint công bố làm mặc định và giữ nguyên cả hai checkpoint PyTorch. Không fine-tune, không dùng internal test để chọn model, không báo accuracy trên nhãn không tương thích.

## Artifact, license và môi trường đã kiểm chứng

- Tải đúng Kaggle `google/cropnet/tensorFlow2/classifier-cassava-disease-v1/1` bằng `kagglehub==1.0.3` vào `checkpoints/cropnet`. [Tutorial chính thức](https://www.tensorflow.org/hub/tutorials/cropnet_cassava) dùng TF Hub `.../cassava_disease_V1/2`; liên kết TF Hub chuyển sang **Kaggle variation version 1**. Hai số phiên bản thuộc hai hệ URL khác nhau.
- Ba file SavedModel thực tải: `saved_model.pb` **1.515.889 B**, SHA-256 `aac5d946061721786ff2cd3e8e49515d1cf660f9982a10266b519cd679410992`; `variables/variables.data-00000-of-00001` **16.971.880 B**, SHA-256 `dd6919af91cf984853e8c57a30a8bd08adbad99d424fda57474e871904a7a1e0`; `variables/variables.index` **5.607 B**, SHA-256 `54749e7c840c85fb08409ec1911128675e743498735e08f12f50ee8d27bd9199`. Bản tải không kèm file LICENSE riêng; [Kaggle model listing](https://www.kaggle.com/models/google/cropnet/tensorFlow1/classifier-cassava-disease-v1/1) ghi **Apache 2.0** cho model và variation classifier. Model card mô tả kiến trúc **MobileNet V3**, nhưng không chỉ rõ biến thể Small/Large của artifact TF2; không suy đoán thêm.
- Môi trường riêng `.venv-cropnet`: Windows, Python 3.13, TensorFlow **2.21.0**, Keras **3.15.1**, KaggleHub **1.0.3**. SavedModel metadata ghi TensorFlow **2.4.0** lúc xuất; `tf.saved_model.load` và inference hoạt động với TF 2.21. Không cài TensorFlow vào `.venv-phase2` hoặc `.venv-train`. Native Windows TensorFlow 2.21 không có GPU; quan sát `tf.config.list_physical_devices('GPU') == []`. Chạy CPU.
- `model.signatures` rỗng, vì vậy **không có `serving_default`**. Chữ ký `__call__` thực tế gồm các biến thể `training=True/False` và `batch_norm_momentum=0.99` hoặc scalar tensor. Với `training=False`: đầu vào `image: float32 [None,224,224,3]`, đầu ra `float32 [None,6]`. Đã nạp và gọi thành công; giá trị đầu ra không âm và cộng thành 1 trên các ảnh thử.
- Preprocessing khớp [tutorial chính thức](https://www.tensorflow.org/hub/tutorials/cropnet_cassava): giải mã RGB, cast `float32`, chia 255, `tf.image.resize(...,(224,224))` bilinear, **không** ImageNet normalization hay center crop. Mã tại `src/cropnet_inference.py`.

## Sáu lớp và ranh giới nhãn

| Index | CropNet / TFDS 2019 | Manifest Kaggle 2020 local |
|---:|---|---|
| 0 | Cassava Bacterial Blight (CBB) | CBB |
| 1 | Cassava Brown Streak Disease (CBSD) | CBSD |
| 2 | **Cassava Green Mite (CGM)** | **Cassava Green Mottle (CGM)** |
| 3 | Cassava Mosaic Disease (CMD) | CMD |
| 4 | Healthy | Healthy |
| 5 | Unknown | Không có lớp tương ứng |

[TFDS Cassava catalog](https://www.tensorflow.org/datasets/catalog/cassava) và tutorial nói rõ **Green Mite** của bộ 2019 gồm 9.430 ảnh; file `label_num_to_disease_map.json` của bộ local 2020 ghi **Green Mottle** trong 21.397 ảnh. Hai tên này chưa được chuyên gia dữ liệu xác nhận tương đương. Không đổi tên index 2 hay bỏ logit Unknown để tính metric 5 lớp. Tutorial báo cáo **0,88 accuracy trên 1.000 ảnh đầu của TFDS 2019 test** (`MAX_EXAMPLES=1000`); đó là kết quả tutorial trên nguồn 2019, không phải kết quả nhóm. Quan hệ trùng ảnh giữa nguồn 2019 và 2020 chưa được kiểm chứng, nên ngay cả metric cùng nguồn sau này cũng không mặc định độc lập.

## Smoke test đã chạy

`scripts/smoke_cropnet.py` lấy **5 ảnh cho mỗi nhãn local**, tổng **25 ảnh**, theo thứ tự đầu tiên trong CSV dự đoán validation 3.210 hàng đã có. Đây là tập khám phá có nhãn từ cùng nguồn Kaggle 2020, không phải tập test độc lập. `reports/cropnet_smoke/predictions.csv` lưu đường dẫn ảnh, nhãn local, dự đoán CropNet và **đủ sáu xác suất**; `summary.json` lưu hash artifact, shape và thống kê.

| Quan sát | Kết quả |
|---|---:|
| Tensor vào / ra | `[1,224,224,3]` / `[1,6]` |
| Số dự đoán CBB / CBSD / Green Mite / CMD / Healthy / Unknown | 4 / 2 / 4 / 7 / 8 / 0 |
| Median thời gian mỗi ảnh sau khi model đã nạp, gồm preprocessing (CPU) | 27,77 ms |
| Xác suất Unknown lớn nhất trong 25 ảnh | 0,003689 |

Ví dụ ảnh local `1000015157.jpg` có ground truth **CBB**: CropNet dự đoán **Healthy 0,733142**, CBB **0,220859**, Green Mite **0,025316**, CBSD **0,018563**, CMD **0,002090**, Unknown **0,000030**. Đây là một lỗi dự đoán quan sát được, không phải ước lượng accuracy.

`scripts/compare_cropnet.py` lưu `comparison.csv` cho **chính 25 ảnh** trên. Dự đoán khớp tên chính xác với MobileNetV3 công bố ở 16/25 ảnh, với MobileNetV3 nhóm fine-tune ở 12/25 ảnh; hai MobileNetV3 khớp 17/25 ảnh. Đây chỉ là **mức đồng thuận dự đoán**, không phải độ đúng: lớp Green Mite và Green Mottle được giữ thành hai chuỗi khác nhau, và Unknown không có nhãn local. Các con số Phase 3B (0,8458 accuracy trên validation của checkpoint công bố có nguy cơ nhiễm; 0,8180 internal test của model nhóm) dùng giao thức và taxonomy khác, **không so trực tiếp** với 25 dự đoán CropNet.

## Ứng dụng và việc còn mở

Streamlit thêm mục `Google CropNet (thử nghiệm, 6 lớp)` chỉ khi môi trường và đủ ba file SavedModel có mặt. Mục này chạy worker bằng `.venv-cropnet` qua stdin, trả về sáu xác suất và ba lớp cao nhất; UI ghi rõ Green Mite/Green Mottle và không nêu metric. Worker kiểm tra SHA-256 ba file trước khi nạp. Đã chạy worker từ môi trường PyTorch hiện tại trên ảnh thật `1000015157.jpg`, nhận `[1,6]` và đủ sáu xác suất. Hai checkpoint cũ, preprocessing và tùy chọn mặc định không đổi. Khởi động TensorFlow cho mỗi dự đoán mất thêm vài giây; 27,77 ms ở trên **không gồm** thời gian khởi động worker. Streamlit AppTest không kết thúc trong lần kiểm tra tại môi trường này; vì vậy xác minh tự động luồng chọn/upload trong UI vẫn còn mở, dù đường suy luận worker thực tế đã chạy.

**Khuyến nghị:** giữ CropNet làm công cụ đối chiếu, chưa thay model mặc định. Muốn đánh giá hiệu năng hợp lệ cần (1) xác minh taxonomy Green Mite/Green Mottle cùng chuyên gia hoặc chọn bộ nhãn đúng 2019, (2) có tập ảnh ngoài nguồn huấn luyện đã kiểm tra trùng lặp, và (3) quy định cách xử lý Unknown trước khi chấm. Không đủ căn cứ để nói CropNet tăng accuracy trên Cassava 2020 trong Phase 3C.
