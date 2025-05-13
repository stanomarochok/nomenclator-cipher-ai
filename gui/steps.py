# gui/steps.py (refactored to use pipeline modules + reset support)

import cv2
from tkinter import messagebox, filedialog
from pipeline.page_segmentation import PageSegmenter
from pipeline.detection import YOLOv11Detector, ContourDetector
from pipeline.classification import RegionClassifier

PAGE_SEGMENTER = PageSegmenter()
CONTOUR_DETECTOR = ContourDetector()
YOLO_DETECTOR = YOLOv11Detector("pipeline/trained_models/detection/words_symbols/YOLOv11/best.pt")


def start_selection(self, event):
    if self.is_grid_view or self.display_image is None:
        return
    self.is_selecting = True

    # Convert canvas to image-space coordinates using canvas.canvasx()
    self.start_x = (self.canvas.canvasx(event.x) - self.offset_x) / self.scale_factor
    self.start_y = (self.canvas.canvasy(event.y) - self.offset_y) / self.scale_factor

    self.canvas.delete("temp_box")


def update_selection(self, event):
    if self.is_grid_view or not self.is_selecting:
        return

    # Convert start to canvas coords
    x1 = self.start_x * self.scale_factor + self.offset_x
    y1 = self.start_y * self.scale_factor + self.offset_y

    # Use actual canvas cursor position
    x2 = self.canvas.canvasx(event.x)
    y2 = self.canvas.canvasy(event.y)

    self.canvas.delete("temp_box")
    self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2, tags="temp_box")


def end_selection(self, event):
    if self.is_grid_view or not self.is_selecting:
        return
    self.is_selecting = False

    end_x = (self.canvas.canvasx(event.x) - self.offset_x) / self.scale_factor
    end_y = (self.canvas.canvasy(event.y) - self.offset_y) / self.scale_factor

    x1 = min(self.start_x, end_x)
    y1 = min(self.start_y, end_y)
    x2 = max(self.start_x, end_x)
    y2 = max(self.start_y, end_y)

    if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
        self.bounding_boxes = [[x1, y1, x2, y2]]
        self.selected_box_index = 0
        self.update_image()

    self.canvas.delete("temp_box")


def remove_selected_box(self):
    if self.selected_box_index is not None:
        del self.bounding_boxes[self.selected_box_index]
        self.selected_box_index = None
        self.update_image()


def step_segment_page(self):
    if not self.image_paths or self.is_grid_view:
        messagebox.showerror("Error", "Please select an image from the grid!")
        return

    img = self.processed_image if self.processed_image is not None else self.current_image
    processed_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    self.segmentation = PAGE_SEGMENTER.segment(processed_bgr)
    self.display_image = processed_bgr.copy()
    self.update_image()


def reset_segment_page(self):
    self.segmentation = None
    self.display_image = self.current_image.copy()
    self.update_image()


def step_detect_words_symbols(self):
    if not self.image_paths or self.is_grid_view:
        messagebox.showerror("Error", "Please select an image from the grid!")
        return

    img = self.processed_image if self.processed_image is not None else self.current_image
    self.detections = YOLO_DETECTOR.detect(img)

    display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    for box in self.detections:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    self.display_image = display_img
    self.update_image()


def reset_detect_words_symbols(self):
    self.detections = None
    self.display_image = self.current_image.copy()
    self.update_image()


def step_all_objects_detection(self):
    if not self.image_paths or self.is_grid_view:
        messagebox.showerror("Error", "Please select an image from the grid!")
        return

    img = self.processed_image if self.processed_image is not None else self.current_image
    padding = self.mask_offset_scale.get() if hasattr(self, 'mask_offset_scale') else 5
    boxes = CONTOUR_DETECTOR.detect(img)

    display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    overlay = display_img.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)

    cv2.addWeighted(overlay, 0.4, display_img, 0.6, 0, display_img)

    self.detections = boxes
    self.display_image = display_img
    self.update_image()


def reset_all_objects_detection(self):
    self.detections = None
    self.display_image = self.current_image.copy()
    self.update_image()


def classify_selected_region(self):
    if not self.bounding_boxes or self.current_image is None or self.is_grid_view:
        messagebox.showerror("Error", "Please select a region on an image!")
        return

    if self.selected_box_index is None:
        messagebox.showerror("Error", "Please select a bounding box to classify!")
        return

    box = self.bounding_boxes[self.selected_box_index]
    x1, y1, x2, y2 = map(int, [coord * self.scale_factor for coord in box])
    x1, y1 = max(0, min(x1, x2)), max(0, min(y1, y2))
    x2, y2 = min(self.current_image.shape[1], max(x1, x2)), min(self.current_image.shape[0], max(y1, y2))

    if x1 >= x2 or y1 >= y2:
        messagebox.showerror("Error", "Invalid region selection!")
        return

    model_name = self.model_var.get()
    if model_name == 'custom' and not hasattr(self, 'custom_model_path'):
        self.load_custom_model()
        if not hasattr(self, 'custom_model_path'):
            messagebox.showerror("Error", "No custom model loaded!")
            return

    classifier = RegionClassifier(model_name=model_name, custom_model_path=getattr(self, 'custom_model_path', None))
    result = classifier.classify(self.current_image, (x1, y1, x2, y2))

    messagebox.showinfo("Classification Result", f"Model: {model_name}\nOutput: {result}")


def load_custom_model(self):
    path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pt")])
    if path:
        self.custom_model_path = path
        messagebox.showinfo("Custom Model Loaded", f"Selected: {path}")


def save_results(self):
    import json
    from tkinter import filedialog

    if not self.image_paths or self.current_image is None:
        messagebox.showerror("Error", "No image loaded to save results!")
        return

    save_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
    if not save_path:
        return

    results = {
        "image": self.image_paths[self.current_image_index],
        "segmentation": self.segmentation.tolist() if self.segmentation is not None else None,
        "detections": self.detections if self.detections is not None else None,
        "mapping": self.mapping if self.mapping is not None else None,
        "htr_results": self.htr_results if self.htr_results is not None else None,
    }

    try:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        messagebox.showinfo("Success", f"Results saved to {save_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save results: {e}")


def zoom_with_scroll(self, event):
    if event.delta > 0:
        self.zoom_in()
    else:
        self.zoom_out()


def start_pan(self, event):
    self.is_panning = True
    self.pan_start_x = event.x
    self.pan_start_y = event.y


def do_pan(self, event):
    if self.is_panning:
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.offset_x += dx
        self.offset_y += dy
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.update_image()


def end_pan(self, event):
    self.is_panning = False
