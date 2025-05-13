# pipeline/page_segmentation.py

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH_PAGE = "pipeline/trained_models/detection/components/YOLOv11/best.pt"


class PageSegmenter:
    def __init__(self, model_path=MODEL_PATH_PAGE):
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise Exception(f"Failed to load page segmentation model: {e}")

    def segment(self, image: np.ndarray):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        try:
            results = self.model.predict(image)
            return results[0].boxes.xyxy.cpu().numpy()
        except Exception as e:
            print(f"Segmentation error: {e}")
            return None
