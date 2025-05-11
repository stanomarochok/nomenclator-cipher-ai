import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class Classifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.preprocess = self._get_preprocess()

    def _get_preprocess(self):
        """Define common preprocessing for all models."""
        return transforms.Compose([
            transforms.Resize((224, 224)),  # Standard input size for most models
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_model(self, model_name):
        """Load a specified pre-trained model."""
        if model_name not in self.models:
            if model_name == "alexnet":
                model = models.alexnet(pretrained=True)
            elif model_name == "densenet201":
                model = models.densenet201(pretrained=True)
            elif model_name == "efficientnet_b7":
                model = models.efficientnet_b7(pretrained=True)
            elif model_name == "inception_v3":
                model = models.inception_v3(pretrained=True)
            elif model_name == "resnet50":
                model = models.resnet50(pretrained=True)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            model.eval()
            model = model.to(self.device)
            self.models[model_name] = model
        return self.models[model_name]

    def classify_region(self, image, region, model_name):
        """Classify a region of the image."""
        model = self.load_model(model_name)
        x1, y1, x2, y2 = region
        region_img = image[y1:y2, x1:x2]

        # Convert to RGB and PIL Image
        if len(region_img.shape) == 2:  # Grayscale to RGB
            region_img = cv2.cvtColor(region_img, cv2.COLOR_GRAY2RGB)
        else:  # BGR to RGB
            region_img = cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(region_img)

        # Preprocess and classify
        input_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = model(input_tensor)

        # Get the predicted class (using ImageNet labels for simplicity)
        _, predicted = torch.max(output, 1)
        # Note: This assumes ImageNet class indices; for custom classes, you'd need a mapping
        class_id = predicted.item()
        class_names = {  # Simplified ImageNet top-level categories
            0: "tench", 1: "goldfish", 2: "great white shark", 3: "tiger shark", 4: "hammerhead shark"
            # Add more classes or use a full ImageNet label file for 1000 classes
        }
        return class_names.get(class_id, f"Class {class_id}")


# Example usage (to be called from gui.py)
if __name__ == "__main__":
    classifier = Classifier()
    # Test with a dummy image and region
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    region = (10, 10, 100, 100)
    print(classifier.classify_region(dummy_img, region, "resnet50"))
