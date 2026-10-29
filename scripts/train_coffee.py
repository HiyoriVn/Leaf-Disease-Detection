import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split

def main():
    # Xác định đường dẫn tự động dựa trên vị trí thư mục hiện tại
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "dataset", "coffee_dataset") # Thư mục chứa ảnh cà phê của bạn
    SAVE_PATH = os.path.join(BASE_DIR, "models", "coffee_model.pth")

    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001

    # Tiền xử lý và tăng cường ảnh
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(DATA_DIR):
        print(f"Lỗi: Không tìm thấy thư mục dữ liệu tại {DATA_DIR}")
        return

    # Đọc dữ liệu
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transforms)
    class_names = full_dataset.classes
    print(f"Các lớp bệnh cà phê phát hiện được: {class_names}")

    # Chia train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Khởi tạo mô hình Transfer Learning (ResNet18)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang huấn luyện trên thiết bị: {device}")
    
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Vòng lặp train
    print("--- BẮT ĐẦU HUẤN LUYỆN ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {running_loss/len(train_dataset):.4f} | Accuracy: {correct/total*100:.2f}%")

    # Lưu model vào thư mục models/
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }, SAVE_PATH)
    print(f"Đã lưu mô hình thành công tại: {SAVE_PATH}")

if __name__ == '__main__':
    main()