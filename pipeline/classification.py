# pipeline/classification.py

import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import os
import cv2
from PIL import Image


class RegionClassifier:
    def __init__(self, model_name: str = "resnet50", custom_model_path: str = None):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_name, custom_model_path)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, name, path):
        if path:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Custom model file not found: {path}")
            model = torch.load(path, map_location=self.device)
        else:
            model = getattr(models, name)(pretrained=True)
            model.fc = nn.Identity()  # replace final layer if needed
        return model.to(self.device)

    def classify(self, image_bgr, box):
        x1, y1, x2, y2 = map(int, box)
        roi = image_bgr[y1:y2, x1:x2]
        if roi.shape[2] == 1:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
        return output.cpu().numpy().tolist()