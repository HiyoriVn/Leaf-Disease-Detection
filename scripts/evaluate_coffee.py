import os
import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 1. Xác định đường dẫn tương đối từ thư mục scripts/ lùi về thư mục gốc dự án
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "coffee_dataset")
MODEL_PATH = os.path.join(BASE_DIR, "models", "coffee_model.pth")

# Kiểm tra xem có file mô hình chưa
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy file mô hình tại: {MODEL_PATH}. Vui lòng train mô hình trước!")

# Tải lại mô hình và checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
class_names = checkpoint['class_names']

# Khởi tạo lại kiến trúc mô hình và nạp trọng số
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Chuẩn bị dữ liệu kiểm thử
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu tại: {DATA_DIR}")

full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

# Sửa lại đoạn chia dữ liệu cho đúng cú pháp Python (khắc phục lỗi dấu - trước đó)
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
_, val_dataset = random_split(full_dataset, [train_size, val_size])

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Tiến hành đánh giá
all_preds = []
all_labels = []

print("Đang tiến hành đánh giá mô hình trên tập validation...")
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# In báo cáo chi tiết (Precision, Recall, F1-Score cho từng bệnh)
print("\n--- BÁO CÁO ĐÁNH GIÁ CHI TIẾT (Classification Report) ---")
print(classification_report(all_labels, all_preds, target_names=class_names))

# In ma trận nhầm lẫn
print("\n--- MA TRẬN NHẦM LẪN (Confusion Matrix) ---")
print(confusion_matrix(all_labels, all_preds))