import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
import pandas as pd

IMAGE_SIZE = 256
CLASS_NAMES = ["normal", "benign", "malignant"]
MODEL_PATH = "breast_cancer_resnet18.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def predict_one(model, path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(t)
        probs = F.softmax(outputs, dim=1)[0]
        top_idx = torch.argmax(probs).item()
    return CLASS_NAMES[top_idx], probs.cpu().numpy()

if __name__ == "__main__":
    FOLDER = "data/images"  # or any folder of test images
    model = load_model()
    rows = []

    for fname in os.listdir(FOLDER):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        path = os.path.join(FOLDER, fname)
        cls, probs = predict_one(model, path)
        rows.append({
            "filename": fname,
            "pred_class": cls,
            "prob_normal": probs[0],
            "prob_benign": probs[1],
            "prob_malignant": probs[2]
        })

    df = pd.DataFrame(rows)
    df.to_csv("batch_predictions.csv", index=False)
    print("Saved batch_predictions.csv with", len(df), "rows")
