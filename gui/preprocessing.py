# gui/preprocessing.py

from pipeline.preprocessing import apply_preprocessing as core_apply_preprocessing
import cv2


def apply_preprocessing(self):
    if self.current_image is None or self.is_grid_view:
        return

    bin_val = self.binarization_scale.get() if hasattr(self, 'binarization_scale') else 0
    gauss_val = self.gaussian_scale.get() if hasattr(self, 'gaussian_scale') else 0
    cont_val = self.contrast_scale.get() if hasattr(self, 'contrast_scale') else 0
    sharp_val = self.sharpen_scale.get() if hasattr(self, 'sharpen_scale') else 0

    img = core_apply_preprocessing(
        self.current_image,
        binarization_value=bin_val,
        gaussian_value=gauss_val,
        contrast_value=cont_val,
        sharpen_value=sharp_val,
    )

    self.processed_image = img
    self.display_image = img.copy() if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    self.update_image()


def update_binarization(self, value):
    self.apply_preprocessing()


def update_gaussian_blur(self, value):
    self.apply_preprocessing()


def update_contrast(self, value):
    self.apply_preprocessing()


def update_sharpening(self, value):
    self.apply_preprocessing()


def update_mask_offset(self, value):
    if self.current_image is not None and not self.is_grid_view:
        self.step_all_objects_detection()
