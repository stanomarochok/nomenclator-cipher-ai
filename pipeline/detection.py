# pipeline/detection.py

import cv2
import numpy as np
from ultralytics import YOLO
import os


class DetectionMethod:
    def detect(self, image: np.ndarray) -> list:
        raise NotImplementedError("Each detection method must implement 'detect'.")


class YOLOv11Detector(DetectionMethod):
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise Exception(f"Failed to load YOLOv11 model: {e}")

    def detect(self, image: np.ndarray) -> list:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        try:
            results = self.model.predict(image)
            return results[0].boxes.xyxy.cpu().numpy().tolist()
        except Exception as e:
            print(f"Detection error: {e}")
            return []


class ContourDetector(DetectionMethod):
    def __init__(self, padding: int = 5):
        self.padding = padding

    def detect(self, image: np.ndarray) -> list:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        threshold = 128
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)

        if self.padding > 0:
            kernel = np.ones((self.padding * 2 + 1, self.padding * 2 + 1), np.uint8)
            binary = cv2.dilate(binary, kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = image.shape[0] * image.shape[1]
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < image_area * 0.5:
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append([x, y, x + w, y + h])
        return boxes
