"""Evaluate only the fixed, held-out validation indices saved during training."""
from __future__ import annotations

from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "coffee_dataset"
MODEL_PATH = ROOT / "models" / "coffee_model.pth"
EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {DATA_DIR}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    required = {"class_names", "model_state_dict", "val_indices", "sample_paths"}
    missing = required - checkpoint.keys()
    if missing:
        raise RuntimeError(
            "Checkpoint cũ không lưu split validation cố định. Hãy huấn luyện lại "
            "bằng scripts/train_coffee.py; thiếu: " + ", ".join(sorted(missing))
        )

    dataset = datasets.ImageFolder(root=DATA_DIR, transform=EVAL_TRANSFORM)
    current_paths = [
        Path(path).relative_to(DATA_DIR).as_posix() for path, _ in dataset.samples
    ]
    if dataset.classes != checkpoint["class_names"] or current_paths != checkpoint["sample_paths"]:
        raise RuntimeError("Dataset/lớp đã thay đổi từ lúc train; không thể dùng split checkpoint.")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    val_loader = DataLoader(
        Subset(dataset, checkpoint["val_indices"]), batch_size=32, shuffle=False
    )

    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.inference_mode():
        for inputs, labels in val_loader:
            all_preds.extend(model(inputs).argmax(1).tolist())
            all_labels.extend(labels.tolist())

    class_ids = list(range(len(checkpoint["class_names"])))
    print("Validation split cố định; preprocessing deterministic.")
    print(
        classification_report(
            all_labels,
            all_preds,
            labels=class_ids,
            target_names=checkpoint["class_names"],
            zero_division=0,
        )
    )
    print(confusion_matrix(all_labels, all_preds, labels=class_ids))


if __name__ == "__main__":
    main()
