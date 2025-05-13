# gui/app.py

import os
import cv2
import numpy as np
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from processor import CipherKeyProcessor
from classifier import Classifier

from .layout import setup_gui, setup_subcategories
from .events import (
    display_image_grid, display_single_image, select_images_or_folder,
    show_previous_image, show_next_image, update_filepath_label,
    toggle_category
)

from .preprocessing import apply_preprocessing, update_binarization, update_gaussian_blur, update_contrast, update_sharpening, update_mask_offset
from .drawing import update_image, zoom_in, zoom_out
from .steps import (
    step_segment_page, step_detect_words_symbols, step_all_objects_detection,
    remove_selected_box, classify_selected_region, load_custom_model, save_results,
    start_selection, update_selection, end_selection,
    zoom_with_scroll, start_pan, do_pan, end_pan,
    reset_segment_page
)


class CipherKeyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nomenclator Cipher Key Processor")
        self.root.geometry("1100x900")
        self.root.resizable(True, True)

        self.processor = CipherKeyProcessor()
        self.classifier = Classifier()

        # Shared state variables
        self.image_paths = []
        self.current_image = None
        self.display_image = None
        self.processed_image = None
        self.segmentation = None
        self.detections = None
        self.tables = None
        self.mapping = None
        self.htr_results = None

        self.photo = None
        self.step_buttons = {}
        self.thumbnail_photos = []
        self.is_grid_view = True
        self.current_image_index = 0

        self.bounding_boxes = []
        self.selected_box_index = None
        self.is_selecting = False
        self.start_x = self.start_y = 0
        self.scale_factor = 1.0

        self.offset_x = 0
        self.offset_y = 0
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0

        self.custom_model_path = None  # For user-selected classifier model

        self.toggle_category = lambda category: toggle_category(self, category)

        # Setup UI layout and bindings
        setup_gui(self)
        setup_subcategories(self)

    # Public methods to bind actions from layout
    def step_segment_page(self): step_segment_page(self)
    def reset_segment_page(self): reset_segment_page(self)

    def step_detect_words_symbols(self): step_detect_words_symbols(self)

    def step_all_objects_detection(self): step_all_objects_detection(self)

    def classify_selected_region(self): classify_selected_region(self)

    def load_custom_model(self): load_custom_model(self)

    def update_image(self, event=None): update_image(self, event)
    def zoom_in(self): zoom_in(self)
    def zoom_out(self): zoom_out(self)

    def apply_preprocessing(self): apply_preprocessing(self)
    def update_binarization(self, value): update_binarization(self, value)
    def update_gaussian_blur(self, value): update_gaussian_blur(self, value)
    def update_contrast(self, value): update_contrast(self, value)
    def update_sharpening(self, value): update_sharpening(self, value)
    def update_mask_offset(self, value): update_mask_offset(self, value)

    def display_image_grid(self): display_image_grid(self)
    def display_single_image(self, path, index): display_single_image(self, path, index)
    def select_images_or_folder(self): select_images_or_folder(self)
    def show_previous_image(self): show_previous_image(self)
    def show_next_image(self): show_next_image(self)
    def update_filepath_label(self): update_filepath_label(self)

    def remove_selected_box(self): remove_selected_box(self)
    def save_results(self): save_results(self)

    def start_selection(self, event): start_selection(self, event)
    def update_selection(self, event): update_selection(self, event)
    def end_selection(self, event): end_selection(self, event)

    def zoom_with_scroll(self, event): zoom_with_scroll(self, event)

    def start_pan(self, event): start_pan(self, event)
    def do_pan(self, event): do_pan(self, event)
    def end_pan(self, event): end_pan(self, event)

