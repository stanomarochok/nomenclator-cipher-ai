# pipeline/preprocessing.py

import cv2
import numpy as np


def apply_preprocessing(image, binarization_value=0, gaussian_value=0, contrast_value=0, sharpen_value=0):
    img = image.copy()

    if binarization_value > 0:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img, binarization_value, 255, cv2.THRESH_BINARY)

    if gaussian_value > 0:
        kernel_size = int(gaussian_value) * 2 + 1
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    if contrast_value > 0:
        img = cv2.convertScaleAbs(img, alpha=contrast_value, beta=0)

    if sharpen_value > 0:
        kernel = np.array([[0, -sharpen_value, 0],
                           [-sharpen_value, 1 + 4 * sharpen_value, -sharpen_value],
                           [0, -sharpen_value, 0]])
        img = cv2.filter2D(img, -1, kernel)

    return img
