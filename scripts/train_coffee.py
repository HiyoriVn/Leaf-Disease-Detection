"""Train the coffee ResNet18 model and save a reproducible data split."""
from __future__ import annotations

from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "coffee_dataset"
SAVE_PATH = ROOT / "models" / "coffee_model.pth"
SEED = 42
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TRAIN_TRANSFORM = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), NORMALIZE]
)
EVAL_TRANSFORM = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(), NORMALIZE]
)


def main() -> None:
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {DATA_DIR}")

    train_source = datasets.ImageFolder(root=DATA_DIR, transform=TRAIN_TRANSFORM)
    class_names = train_source.classes
    targets = train_source.targets
    indices = list(range(len(train_source)))
    if min(targets.count(class_id) for class_id in range(len(class_names))) < 2:
        raise ValueError("Mỗi lớp cần ít nhất 2 ảnh để tạo split train/validation.")
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=SEED, stratify=targets
    )
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    loader_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        Subset(train_source, train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=loader_generator,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | Loss: {running_loss / total:.4f} "
            f"| Accuracy: {correct / total:.2%}"
        )

    model = model.to("cpu").eval()
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "sample_paths": [
                Path(path).relative_to(DATA_DIR).as_posix()
                for path, _ in train_source.samples
            ],
            "split_seed": SEED,
        },
        SAVE_PATH,
    )
    print(f"Saved checkpoint: {SAVE_PATH}")


if __name__ == "__main__":
    main()
