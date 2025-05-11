import os
import cv2
import numpy as np
from tkinter import Tk, Frame, Button, Scale, Label, filedialog, messagebox, Canvas, Scrollbar, StringVar, OptionMenu
from PIL import Image, ImageTk
from processor import CipherKeyProcessor
from classifier import Classifier

class CipherKeyApp:
    def __init__(self, root):
        """Initialize the GUI application with a fixed window size."""
        self.root = root
        self.root.title("Nomenclator Cipher Key Processor")
        self.root.resizable(True, True)
        self.processor = CipherKeyProcessor()
        self.classifier = Classifier()
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
        self.bounding_boxes = []  # List to store multiple bounding boxes
        self.selected_box_index = None  # Currently selected box for editing
        self.is_selecting = False
        self.start_x = self.start_y = 0
        self.scale_factor = 1.0

        self.setup_gui()

        # Set fixed window size
        self.root.geometry("1100x900")

    def setup_gui(self):
        """Set up the GUI with file selection, categorized filters, navigation, and filepath controls."""
        main_frame = Frame(self.root, padx=5, pady=5)
        main_frame.pack(fill="both", expand=True)

        self.control_frame = Frame(main_frame, width=300)
        self.control_frame.pack(side="left", fill="y", padx=5)
        self.control_frame.pack_propagate(False)

        self.right_frame = Frame(main_frame)
        self.right_frame.pack(side="right", fill="both", expand=True)

        self.nav_frame = Frame(self.right_frame)
        self.nav_frame.pack(fill="x")
        self.filepath_label = Label(self.nav_frame, text="No images selected", wraplength=500)
        self.filepath_label.pack(side="left", padx=5, pady=2)
        self.prev_button = Button(self.nav_frame, text="Previous", command=self.show_previous_image)
        self.prev_button.pack(side="left", padx=5)
        self.next_button = Button(self.nav_frame, text="Next", command=self.show_next_image)
        self.next_button.pack(side="left", padx=5)
        self.back_button = Button(self.nav_frame, text="Back to Grid", command=self.display_image_grid)
        self.back_button.pack(side="left", padx=5)
        self.back_button.pack_forget()

        self.zoom_frame = Frame(self.right_frame)
        self.zoom_frame.pack(fill="x")
        Button(self.zoom_frame, text="Zoom In", command=self.zoom_in).pack(side="left", padx=5)
        Button(self.zoom_frame, text="Zoom Out", command=self.zoom_out).pack(side="left", padx=5)
        Button(self.zoom_frame, text="Remove Box", command=self.remove_selected_box).pack(side="left", padx=5)

        self.image_frame = Frame(self.right_frame)
        self.image_frame.pack(fill="both", expand=True)

        self.canvas = Canvas(self.image_frame, bg="gray")
        self.canvas.pack(fill="both", expand=True)

        Button(self.control_frame, text="Select Images/Folder", command=self.select_images_or_folder).pack(fill="x", pady=20)

        self.categories = {
            "Preprocessing": ["Binarization", "Gaussian Blur", "Contrast Adjustment", "Sharpening"],
            "Processing Steps": ["Segment Page", "Detect Words Symbols", "All Objects Detection", "Detect Table Structure",
                              "Map Plaintext to Ciphertext", "HTR", "Classification"]
        }
        self.subcategory_frames = {}
        self.active_subcategory = None

        for category in self.categories:
            btn = Button(self.control_frame, text=category, command=lambda c=category: self.toggle_category(c))
            btn.pack(fill="x", pady=2)
            frame = Frame(self.control_frame)
            frame.pack(fill="x", padx=5)
            frame.pack_forget()
            self.subcategory_frames[category] = frame

        self.setup_subcategories()

        Button(self.control_frame, text="Save Results", command=self.save_results).pack(fill="x", pady=5, side="bottom")

    def setup_subcategories(self):
        """Set up controls for each subcategory, nesting properties and substeps."""
        for category, subcategories in self.categories.items():
            frame = self.subcategory_frames[category]
            for subcat in subcategories:
                normalized_subcat = subcat.lower().replace(' ', '_')
                btn = Button(frame, text=f"Step: {subcat}", command=getattr(self, f"step_{normalized_subcat}"), width=15)
                btn.pack(fill="x", padx=30, pady=1)

                properties_frame = Frame(frame)
                properties_frame.pack(fill="x", padx=40)

                if subcat == "Binarization":
                    binarization_frame = Frame(properties_frame)
                    binarization_frame.pack(fill="x", pady=2)
                    Label(binarization_frame, text="Binarization").pack(side="left")
                    self.binarization_scale = Scale(binarization_frame, from_=0, to=255, orient="horizontal", command=self.update_binarization)
                    self.binarization_scale.set(0)
                    self.binarization_scale.pack(side="left", padx=5, expand=True)
                    Button(binarization_frame, text="Reset", command=lambda: self.reset_slider(self.binarization_scale, 0)).pack(side="left")

                elif subcat == "Gaussian Blur":
                    gaussian_frame = Frame(properties_frame)
                    gaussian_frame.pack(fill="x", pady=2)
                    Label(gaussian_frame, text="Gaussian Blur").pack(side="left")
                    self.gaussian_scale = Scale(gaussian_frame, from_=0, to=5, orient="horizontal", command=self.update_gaussian_blur)
                    self.gaussian_scale.set(0)
                    self.gaussian_scale.pack(side="left", padx=5, expand=True)
                    Button(gaussian_frame, text="Reset", command=lambda: self.reset_slider(self.gaussian_scale, 0)).pack(side="left")

                elif subcat == "Contrast Adjustment":
                    contrast_frame = Frame(properties_frame)
                    contrast_frame.pack(fill="x", pady=2)
                    Label(contrast_frame, text="Contrast").pack(side="left")
                    self.contrast_scale = Scale(contrast_frame, from_=0, to=2, orient="horizontal", command=self.update_contrast, resolution=0.1)
                    self.contrast_scale.set(0)
                    self.contrast_scale.pack(side="left", padx=5, expand=True)
                    Button(contrast_frame, text="Reset", command=lambda: self.reset_slider(self.contrast_scale, 0)).pack(side="left")

                elif subcat == "Sharpening":
                    sharpen_frame = Frame(properties_frame)
                    sharpen_frame.pack(fill="x", pady=2)
                    Label(sharpen_frame, text="Sharpening").pack(side="left")
                    self.sharpen_scale = Scale(sharpen_frame, from_=0, to=2, orient="horizontal", command=self.update_sharpening, resolution=0.1)
                    self.sharpen_scale.set(0)
                    self.sharpen_scale.pack(side="left", padx=5, expand=True)
                    Button(sharpen_frame, text="Reset", command=lambda: self.reset_slider(self.sharpen_scale, 0)).pack(side="left")

                elif subcat == "All Objects Detection":
                    detection_frame = Frame(properties_frame)
                    detection_frame.pack(fill="x", pady=2)
                    Label(detection_frame, text="Algorithm:").pack(side="left")
                    self.detection_algo_var = StringVar(self.root)
                    self.detection_algo_var.set("Contour Masks")
                    detection_options = ["Contour Masks"]
                    OptionMenu(detection_frame, self.detection_algo_var, *detection_options).pack(side="left")

                    mask_frame = Frame(properties_frame)
                    mask_frame.pack(fill="x", pady=2)
                    Label(mask_frame, text="Contour Mask Offset").pack(side="left")
                    self.mask_offset_scale = Scale(mask_frame, from_=0, to=20, orient="horizontal", command=self.update_mask_offset)
                    self.mask_offset_scale.set(5)
                    self.mask_offset_scale.pack(side="left", padx=5, expand=True)
                    Button(mask_frame, text="Reset", command=lambda: self.reset_slider(self.mask_offset_scale, 5)).pack(side="left")

                elif subcat == "Classification":
                    class_frame = Frame(properties_frame)
                    class_frame.pack(fill="x", pady=2)
                    Label(class_frame, text="Model:").pack(side="left")
                    self.model_var = StringVar(self.root)
                    self.model_var.set("resnet50")
                    model_options = ["alexnet", "densenet201", "efficientnet_b7", "inception_v3", "resnet50"]
                    OptionMenu(class_frame, self.model_var, *model_options).pack(side="left")
                    Button(class_frame, text="Classify Region", command=self.classify_selected_region).pack(side="left", padx=5)

    def select_images_or_folder(self):
        """Handle selection of individual images or a folder and display them in a grid."""
        files = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if files:
            self.image_paths = list(files)
        else:
            folder = filedialog.askdirectory()
            if folder:
                self.image_paths = [os.path.join(folder, f) for f in os.listdir(folder)
                                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            else:
                return

        if self.image_paths:
            self.is_grid_view = True
            self.current_image_index = 0
            self.display_image_grid()
            self.update_filepath_label()
        else:
            messagebox.showerror("Error", "No images found!")

    def display_image_grid(self):
        """Display all images in a scrollable grid layout with dynamic column count."""
        self.is_grid_view = True
        self.back_button.pack_forget()
        self.prev_button.pack()
        self.next_button.pack()

        # Clear the main canvas without destroying it
        self.canvas.delete("all")

        # Create a new canvas for the grid view inside self.canvas
        grid_canvas = Canvas(self.image_frame)
        grid_canvas.pack(side="top", fill="both", expand=True)
        scrollbar = Scrollbar(self.image_frame, orient="vertical", command=grid_canvas.yview)
        scrollbar.pack(side="right", fill="y")
        scrollable_frame = Frame(grid_canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: grid_canvas.configure(scrollregion=grid_canvas.bbox("all"))
        )

        grid_canvas.configure(yscrollcommand=scrollbar.set)
        grid_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        self.thumbnail_photos = []
        thumbnail_size = (150, 150)

        window_width = self.image_frame.winfo_width()
        if window_width <= 1:
            window_width = 800
        cols = max(1, window_width // (thumbnail_size[0] + 10))

        for idx, img_path in enumerate(self.image_paths):
            try:
                img = Image.open(img_path)
                img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.thumbnail_photos.append(photo)

                row = idx // cols
                col = idx % cols
                label = Label(scrollable_frame, image=photo)
                label.grid(row=row, column=col, padx=5, pady=5)
                label.bind("<Button-1>", lambda event, path=img_path, index=idx: self.display_single_image(path, index))
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue

        # Ensure the grid canvas is destroyed when switching to single image view
        self.grid_canvas = grid_canvas  # Store reference to destroy later

    def display_single_image(self, img_path, index):
        """Display a single image, fitting it to the canvas."""
        self.is_grid_view = False
        self.current_image_index = index
        self.back_button.pack()
        self.prev_button.pack()
        self.next_button.pack()
        self.update_filepath_label()

        # Destroy the grid canvas if it exists
        if hasattr(self, 'grid_canvas'):
            self.grid_canvas.destroy()
            delattr(self, 'grid_canvas')

        self.current_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if self.current_image is None:
            messagebox.showerror("Error", f"Failed to load image: {img_path}")
            return
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2BGR)

        self.display_image = self.current_image.copy()
        self.processed_image = None
        self.segmentation = None
        self.detections = None
        self.tables = None
        self.mapping = None
        self.htr_results = None
        self.bounding_boxes = []
        self.selected_box_index = None

        canvas_width = self.canvas.winfo_width() or 800
        canvas_height = self.canvas.winfo_height() or 1200
        img_height, img_width = self.current_image.shape[:2]
        self.scale_factor = min(canvas_width / img_width, canvas_height / img_height)

        self.canvas.delete("all")
        self.canvas.bind("<Configure>", self.update_image)
        self.canvas.bind("<Button-1>", self.start_selection)
        self.canvas.bind("<B1-Motion>", self.update_selection)
        self.canvas.bind("<ButtonRelease-1>", self.end_selection)
        self.canvas.bind("<Button-3>", self.adjust_region)
        self.canvas.bind("<MouseWheel>", self.zoom_with_scroll)
        self.canvas.bind("<Button-4>", self.zoom_with_scroll)
        self.canvas.bind("<Button-5>", self.zoom_with_scroll)

        self.update_image()

    def start_selection(self, event):
        """Start selecting a new region, clearing existing boxes."""
        self.is_selecting = True
        self.start_x = self.canvas.canvasx(event.x) / self.scale_factor
        self.start_y = self.canvas.canvasy(event.y) / self.scale_factor
        self.bounding_boxes = []  # Clear all existing boxes
        self.selected_box_index = None
        self.bounding_boxes.append([self.start_x, self.start_y, self.start_x, self.start_y])
        self.selected_box_index = 0
        self.update_image()

    def update_selection(self, event):
        """Update the region during selection and draw the bounding box on the fly."""
        if self.is_selecting:
            self.bounding_boxes[self.selected_box_index][2] = self.canvas.canvasx(event.x) / self.scale_factor
            self.bounding_boxes[self.selected_box_index][3] = self.canvas.canvasy(event.y) / self.scale_factor
            x1, y1, x2, y2 = map(int, [coord * self.scale_factor for coord in self.bounding_boxes[self.selected_box_index]])
            x1, y1 = max(0, min(x1, x2)), max(0, min(y1, y2))
            x2, y2 = min(self.display_image.shape[1] * self.scale_factor, max(x1, x2)), min(self.display_image.shape[0] * self.scale_factor, max(y1, y2))
            self.canvas.delete("temp_rect")
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2, tags="temp_rect")
            self.canvas.update()

    def end_selection(self, event):
        """End region selection."""
        self.is_selecting = False
        if self.selected_box_index is not None:
            self.bounding_boxes[self.selected_box_index][2] = self.canvas.canvasx(event.x) / self.scale_factor
            self.bounding_boxes[self.selected_box_index][3] = self.canvas.canvasy(event.y) / self.scale_factor
            x1, y1, x2, y2 = map(int, self.bounding_boxes[self.selected_box_index])
            self.bounding_boxes[self.selected_box_index] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        self.update_image()

    def adjust_region(self, event):
        """Adjust the selected region with right-click (move or resize)."""
        x, y = self.canvas.canvasx(event.x) / self.scale_factor, self.canvas.canvasy(event.y) / self.scale_factor
        self.selected_box_index = None

        # Find the closest box to the click
        for idx, box in enumerate(self.bounding_boxes):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            if (x1 - 10 <= x <= x2 + 10) and (y1 - 10 <= y <= y2 + 10):
                self.selected_box_index = idx
                break

        if self.selected_box_index is not None:
            box = self.bounding_boxes[self.selected_box_index]
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            if abs(x - cx) < w/4 and abs(y - cy) < h/4:  # Move
                dx = x - cx
                dy = y - cy
                box[0] += dx
                box[1] += dy
                box[2] += dx
                box[3] += dy
            else:  # Resize
                if x < x1: box[0] = x
                elif x > x2: box[2] = x
                if y < y1: box[1] = y
                elif y > y2: box[3] = y
            box[0], box[1] = max(0, min(box[0], box[2])), max(0, min(box[1], box[3]))
            box[2], box[3] = min(self.current_image.shape[1], max(box[0], box[2])), min(self.current_image.shape[0], max(box[1], box[3]))
            self.update_image()

    def remove_selected_box(self):
        """Remove the currently selected bounding box."""
        if self.selected_box_index is not None:
            self.bounding_boxes.pop(self.selected_box_index)
            self.selected_box_index = None
            self.update_image()

    def classify_selected_region(self):
        """Classify the selected region using the chosen model."""
        if not self.bounding_boxes or self.current_image is None or self.is_grid_view:
            messagebox.showerror("Error", "Please select a region on an image!")
            return
        if self.selected_box_index is None:
            messagebox.showerror("Error", "Please select a bounding box to classify!")
            return
        x1, y1, x2, y2 = map(int, [coord * self.scale_factor for coord in self.bounding_boxes[self.selected_box_index]])
        x1, y1 = max(0, min(x1, x2)), max(0, min(y1, y2))
        x2, y2 = min(self.current_image.shape[1], max(x1, x2)), min(self.current_image.shape[0], max(y1, y2))
        if x1 >= x2 or y1 >= y2:
            messagebox.showerror("Error", "Invalid region selection!")
            return

        result = self.classifier.classify_region(self.current_image, (x1, y1, x2, y2), self.model_var.get())
        messagebox.showinfo("Classification Result", f"Predicted class: {result}")

    def show_previous_image(self):
        """Show the previous image in the list."""
        if self.is_grid_view or not self.image_paths:
            return
        self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
        self.display_single_image(self.image_paths[self.current_image_index], self.current_image_index)

    def show_next_image(self):
        """Show the next image in the list."""
        if self.is_grid_view or not self.image_paths:
            return
        self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
        self.display_single_image(self.image_paths[self.current_image_index], self.current_image_index)

    def update_filepath_label(self):
        """Update the filepath label based on the current view."""
        if not self.image_paths:
            self.filepath_label.config(text="No images selected")
        elif self.is_grid_view:
            self.filepath_label.config(text=f"Selected {len(self.image_paths)} images")
        else:
            self.filepath_label.config(text=self.image_paths[self.current_image_index])

    def toggle_category(self, category):
        """Toggle the visibility of the subcategory frame for the selected category."""
        if self.active_subcategory:
            self.subcategory_frames[self.active_subcategory].pack_forget()
        if self.active_subcategory != category:
            self.subcategory_frames[category].pack(fill="x")
            self.active_subcategory = category
        else:
            self.active_subcategory = None

    def apply_preprocessing(self):
        """Apply all active preprocessing steps in order."""
        if self.current_image is None or self.is_grid_view:
            return
        img = self.current_image.copy()

        binarization_value = self.binarization_scale.get() if self.binarization_scale else 0
        if binarization_value > 0:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, img = cv2.threshold(img, binarization_value, 255, cv2.THRESH_BINARY)

        gaussian_value = self.gaussian_scale.get() if self.gaussian_scale else 0
        if gaussian_value > 0:
            kernel_size = int(gaussian_value) * 2 + 1
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        contrast_value = self.contrast_scale.get() if self.contrast_scale else 0
        if contrast_value > 0:
            alpha = contrast_value
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)

        sharpen_value = self.sharpen_scale.get() if self.sharpen_scale else 0
        if sharpen_value > 0:
            kernel = np.array([[0, -sharpen_value, 0],
                             [-sharpen_value, 1 + 4*sharpen_value, -sharpen_value],
                             [0, -sharpen_value, 0]])
            img = cv2.filter2D(img, -1, kernel)

        self.processed_image = img
        self.display_image = img.copy() if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self.update_image()

    def update_binarization(self, value):
        """Update binarization on-the-fly."""
        if self.current_image is not None and not self.is_grid_view:
            self.apply_preprocessing()

    def update_gaussian_blur(self, value):
        """Update Gaussian blur on-the-fly."""
        if self.current_image is not None and not self.is_grid_view:
            self.apply_preprocessing()

    def update_contrast(self, value):
        """Update contrast on-the-fly."""
        if self.current_image is not None and not self.is_grid_view:
            self.apply_preprocessing()

    def update_sharpening(self, value):
        """Update sharpening on-the-fly."""
        if self.current_image is not None and not self.is_grid_view:
            self.apply_preprocessing()

    def update_mask_offset(self, value):
        """Update contour mask offset on-the-fly."""
        if self.current_image is not None and not self.is_grid_view:
            self.step_all_objects_detection()

    def reset_slider(self, scale, default_value):
        """Reset the slider to its default value."""
        scale.set(default_value)
        if scale == self.binarization_scale:
            self.update_binarization(default_value)
        elif scale == self.gaussian_scale:
            self.update_gaussian_blur(default_value)
        elif scale == self.contrast_scale:
            self.update_contrast(default_value)
        elif scale == self.sharpen_scale:
            self.update_sharpening(default_value)
        elif scale == self.mask_offset_scale:
            self.update_mask_offset(default_value)

    def update_image(self, event=None):
        """Update the displayed image, resizing to fit the canvas and draw bounding boxes."""
        if self.display_image is None or self.is_grid_view:
            self.canvas.delete("all")
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 800, 1200

        img_height, img_width = self.display_image.shape[:2]
        new_width = int(img_width * self.scale_factor)
        new_height = int(img_height * self.scale_factor)
        resized_image = cv2.resize(self.display_image, (new_width, new_height))

        if len(resized_image.shape) == 2:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
        pil_image = Image.fromarray(resized_image)
        self.photo = ImageTk.PhotoImage(pil_image)

        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.photo, anchor="center")

        if self.segmentation and not self.is_selecting:
            for box in self.segmentation:
                x1, y1, x2, y2 = map(int, [b * self.scale_factor for b in box])
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="green", width=2)

        # Draw all bounding boxes
        for idx, box in enumerate(self.bounding_boxes):
            x1, y1, x2, y2 = map(int, [coord * self.scale_factor for coord in box])
            x1, y1 = max(0, min(x1, x2)), max(0, min(y1, y2))
            x2, y2 = min(new_width, max(x1, x2)), min(new_height, max(y1, y2))
            color = "blue" if idx == self.selected_box_index else "red"
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags=f"box_{idx}")

        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def zoom_with_scroll(self, event):
        """Zoom the image with Ctrl + Scroll."""
        if event.state & 0x4:
            if event.delta > 0 or event.num == 4:
                self.zoom_in()
            elif event.delta < 0 or event.num == 5:
                self.zoom_out()

    def zoom_in(self):
        """Zoom in the image."""
        self.scale_factor = min(3.0, self.scale_factor + 0.1)
        self.update_image()

    def zoom_out(self):
        """Zoom out the image."""
        self.scale_factor = max(0.1, self.scale_factor - 0.1)
        self.update_image()

    def step_binarization(self):
        """Execute the binarization step and update the display."""
        if not self.image_paths or self.is_grid_view:
            messagebox.showerror("Error", "Please select an image from the grid!")
            return
        self.apply_preprocessing()

    def step_gaussian_blur(self):
        """Execute the Gaussian blur step and update the display."""
        if not self.image_paths or self.is_grid_view:
            messagebox.showerror("Error", "Please select an image from the grid!")
            return
        self.apply_preprocessing()

    def step_contrast_adjustment(self):
        """Execute the contrast adjustment step and update the display."""
        if not self.image_paths or self.is_grid_view:
            messagebox.showerror("Error", "Please select an image from the grid!")
            return
        self.apply_preprocessing()

    def step_sharpening(self):
        """Execute the sharpening step and update the display."""
        if not self.image_paths or self.is_grid_view:
            messagebox.showerror("Error", "Please select an image from the grid!")
            return
        self.apply_preprocessing()

    def step_segment_page(self):
        """Execute the page segmentation step and update the display."""
        if not self.image_paths or self.is_grid_view:
            messagebox.showerror("Error", "Please select an image from the grid!")
            return
        img = self.processed_image if self.processed_image is not None else self.current_image
        processed_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self.segmentation = self.processor.segment_page(processed_bgr)
        self.display_image = processed_bgr.copy()
        self.update_image()

    def step_detect_words_symbols(self):
        """Execute the words/symbols detection step and update the display."""
        if not self.image_paths or self.is_grid_view:
            messagebox.showerror("Error", "Please select an image from the grid!")
            return
        img = self.processed_image if self.processed_image is not None else self.current_image
        processed_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self.detections = self.processor.detect_words_symbols(processed_bgr)
        self.display_image = processed_bgr.copy()
        if self.detections is not None:
            for box in self.detections:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        self.update_image()

    def step_all_objects_detection(self):
        """Execute all objects detection with selected algorithm."""
        if not self.image_paths or self.is_grid_view:
            messagebox.showerror("Error", "Please select an image from the grid!")
            return
        img = self.processed_image if self.processed_image is not None else self.current_image
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        algo = self.detection_algo_var.get()
        if algo == "Contour Masks":
            threshold = 128
            blurred = cv2.GaussianBlur(img, (5, 5), 0)
            _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            padding = self.mask_offset_scale.get()

            if padding > 0:
                kernel = np.ones((padding * 2 + 1, padding * 2 + 1), np.uint8)
                dilated = cv2.dilate(binary, kernel, iterations=1)
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            image_area = img.shape[0] * img.shape[1]
            overlay = display_img.copy()
            self.detections = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < image_area * 0.5:
                    epsilon = 0.01 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    cv2.fillPoly(overlay, [approx], (0, 0, 255))
                    x, y, w, h = cv2.boundingRect(contour)
                    self.detections.append([x, y, x + w, y + h])

            cv2.addWeighted(overlay, 0.4, display_img, 0.6, 0, display_img)

            self.display_image = display_img
            self.update_image()

    def step_detect_table_structure(self):
        """Detect table structure using detected objects with non-straight lines."""
        if not self.image_paths or self.is_grid_view:
            messagebox.showerror("Error", "Please select an image from the grid!")
            return
        if self.detections is None:
            messagebox.showerror("Error", "Run All Objects Detection step first!")
            return
        self.tables = []
        display_img = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR) if self.processed_image is not None else self.display_image.copy()

        boxes = self.detections
        if not boxes:
            return

        boxes.sort(key=lambda b: b[1])
        rows = []
        current_row = [boxes[0]]
        for box in boxes[1:]:
            if abs(box[1] - current_row[0][1]) < 50:
                current_row.append(box)
            else:
                rows.append(current_row)
                current_row = [box]
        rows.append(current_row)

        for row in rows:
            row.sort(key=lambda b: b[0])

        for i in range(len(rows)):
            for j in range(len(rows[i]) - 1):
                x1, y1, x2, y2 = map(int, rows[i][j])
                x3, y3, x4, y4 = map(int, rows[i][j + 1])
                cv2.line(display_img, ((x1+x2)//2, (y1+y2)//2), ((x3+x4)//2, (y3+y4)//2), (0, 255, 0), 2)

            if i < len(rows) - 1:
                next_row = rows[i + 1]
                for obj1, obj2 in zip(rows[i], next_row):
                    x1, y1, x2, y2 = map(int, obj1)
                    x3, y3, x4, y4 = map(int, obj2)
                    cv2.line(display_img, ((x1+x2)//2, (y1+y2)//2), ((x3+x4)//2, (y3+y4)//2), (0, 0, 255), 2)

        self.display_image = display_img
        self.update_image()

    def step_map_plaintext_to_ciphertext(self):
        """Execute the mapping step."""
        if not self.image_paths or self.is_grid_view:
            messagebox.showerror("Error", "Please select an image from the grid!")
            return
        if not self.tables:
            messagebox.showerror("Error", "Run Detect Table Structure step first!")
            return
        self.mapping = self.processor.map_plaintext_ciphertext(self.tables)
        messagebox.showinfo("Info", f"Mapping completed: {self.mapping}")
        self.display_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)
        self.update_image()

    def step_htr(self):
        """Execute the HTR step."""
        if not self.image_paths or self.is_grid_view:
            messagebox.showerror("Error", "Please select an image from the grid!")
            return
        if self.detections is None:
            messagebox.showerror("Error", "Run All Objects Detection step first!")
            return
        self.htr_results = self.processor.htr(self.processed_image, self.detections)
        messagebox.showinfo("Info", f"HTR Results: {self.htr_results}")
        self.display_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)
        self.update_image()

    def step_classification(self):
        """Placeholder for classification step (handled by classify_selected_region button)."""
        pass

    def save_results(self):
        """Save the processed image and results."""
        if not self.image_paths or self.is_grid_view:
            messagebox.showerror("Error", "Please select an image from the grid!")
            return
        if self.display_image is None:
            messagebox.showerror("Error", "No processed image to save!")
            return
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, f"processed_{os.path.basename(self.image_paths[self.current_image_index])}"), self.display_image)
        if self.mapping:
            with open(os.path.join(output_dir, f"mapping_{os.path.basename(self.image_paths[self.current_image_index])}.txt"), "w") as f:
                f.write(str(self.mapping))
        if self.htr_results:
            with open(os.path.join(output_dir, f"htr_{os.path.basename(self.image_paths[self.current_image_index])}.txt"), "w") as f:
                f.write(str(self.htr_results))
        messagebox.showinfo("Info", "Results saved successfully!")