import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os


def predict_coffee(image: Image.Image):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "models", "coffee_model.pth")
    
    if not os.path.exists(model_path):
        return "Chưa có file mô hình .pth trong thư mục models/", 0.0, []
        
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    class_names = checkpoint['class_names']
    
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # Lấy ra Top 3 xác suất cao nhất
        topk_prob, topk_indices = torch.topk(probabilities, k=min(3, len(class_names)))
        
        top3_list = []
        for i in range(len(topk_prob)):
            top3_list.append({
                "label": class_names[topk_indices[i].item()],
                "probability": topk_prob[i].item()
            })
            
    result_label = top3_list[0]["label"]
    score = top3_list[0]["probability"] * 100
    
    return result_label, score, top3_list