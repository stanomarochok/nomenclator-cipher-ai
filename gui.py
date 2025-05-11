import os
import cv2
import numpy as np
from tkinter import Tk, Frame, Button, Scale, Label, filedialog, messagebox, Canvas, Scrollbar
from PIL import Image, ImageTk
from processor import CipherKeyProcessor

class CipherKeyApp:
    def __init__(self, root):
        """Initialize the GUI application with a dynamically sized window."""
        self.root = root
        self.root.title("Nomenclator Cipher Key Processor")
        self.root.resizable(True, True)
        self.processor = CipherKeyProcessor()
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
        self.binarization_scale = None
        self.mask_offset_scale = None
        self.row_threshold_scale = None
        self.col_threshold_scale = None
        self.thumbnail_photos = []
        self.is_grid_view = True
        self.current_image_index = 0  # Track the current image in single-image view

        # Set up the GUI
        self.setup_gui()

        # Set initial window size with a default height of 800px
        control_width = 300
        control_height = self.control_frame.winfo_reqheight()
        initial_width = control_width + 400
        initial_height = max(control_height + 50, 800)
        self.root.geometry(f"{initial_width}x{initial_height}")

    def setup_gui(self):
        """Set up the GUI with file selection, categorized filters, properties, navigation, and filepath display."""
        main_frame = Frame(self.root, padx=5, pady=5)
        main_frame.pack(fill="both", expand=True)

        # Left frame for controls (fixed width)
        self.control_frame = Frame(main_frame, width=300)
        self.control_frame.pack(side="left", fill="y", padx=5)
        self.control_frame.pack_propagate(False)

        # Right frame for image display or grid
        self.right_frame = Frame(main_frame)
        self.right_frame.pack(side="right", fill="both", expand=True)

        # Navigation and filepath display at the top of the right frame
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
        self.back_button.pack_forget()  # Hidden initially

        # Image frame for grid or single image
        self.image_frame = Frame(self.right_frame)
        self.image_frame.pack(fill="both", expand=True)

        # File selection (top of left panel)
        Button(self.control_frame, text="Select Images/Folder", command=self.select_images_or_folder).pack(fill="x", pady=20)

        # Categories and subcategories
        self.categories = {
            "Preprocessing": ["Binarization", "Other Preprocessing"],
            "Table Detection": ["Row Threshold", "Column Threshold"],
            "Processing Steps": ["Segment Page", "Detect Words Symbols", "Detect Table Structure",
                              "Map Plaintext to Ciphertext", "HTR"]
        }
        self.subcategory_frames = {}
        self.active_subcategory = None

        # Create category buttons
        for category in self.categories:
            btn = Button(self.control_frame, text=category, command=lambda c=category: self.toggle_category(c))
            btn.pack(fill="x", pady=2)
            frame = Frame(self.control_frame)
            frame.pack(fill="x", padx=5)
            frame.pack_forget()
            self.subcategory_frames[category] = frame

        # Populate subcategory frames
        self.setup_subcategories()

        # Properties at the bottom of the left panel
        self.properties_frame = Frame(self.control_frame)
        self.properties_frame.pack(fill="x", side="bottom", pady=5)

        # Binarization properties
        binarization_frame = Frame(self.properties_frame)
        binarization_frame.pack(fill="x", pady=2)
        Label(binarization_frame, text="Binarization Threshold").pack(side="left")
        self.binarization_scale = Scale(binarization_frame, from_=0, to=255, orient="horizontal", command=self.update_binarization)
        self.binarization_scale.set(210)
        self.binarization_scale.pack(side="left", padx=5, expand=True)
        Button(binarization_frame, text="Reset", command=lambda: self.reset_slider(self.binarization_scale, 210)).pack(side="left")

        # Contour Mask Offset properties
        mask_offset_frame = Frame(self.properties_frame)
        mask_offset_frame.pack(fill="x", pady=2)
        Label(mask_offset_frame, text="Contour Mask Offset").pack(side="left")
        self.mask_offset_scale = Scale(mask_offset_frame, from_=0, to=20, orient="horizontal", command=self.update_mask_offset)
        self.mask_offset_scale.set(5)
        self.mask_offset_scale.pack(side="left", padx=5, expand=True)
        Button(mask_offset_frame, text="Reset", command=lambda: self.reset_slider(self.mask_offset_scale, 5)).pack(side="left")

        # Row Threshold properties
        row_threshold_frame = Frame(self.properties_frame)
        row_threshold_frame.pack(fill="x", pady=2)
        Label(row_threshold_frame, text="Table Row Threshold").pack(side="left")
        self.row_threshold_scale = Scale(row_threshold_frame, from_=10, to=100, orient="horizontal", command=self.update_row_threshold)
        self.row_threshold_scale.set(50)
        self.row_threshold_scale.pack(side="left", padx=5, expand=True)
        Button(row_threshold_frame, text="Reset", command=lambda: self.reset_slider(self.row_threshold_scale, 50)).pack(side="left")

        # Column Threshold properties
        col_threshold_frame = Frame(self.properties_frame)
        col_threshold_frame.pack(fill="x", pady=2)
        Label(col_threshold_frame, text="Table Col Threshold").pack(side="left")
        self.col_threshold_scale = Scale(col_threshold_frame, from_=50, to=300, orient="horizontal", command=self.update_col_threshold)
        self.col_threshold_scale.set(150)
        self.col_threshold_scale.pack(side="left", padx=5, expand=True)
        Button(col_threshold_frame, text="Reset", command=lambda: self.reset_slider(self.col_threshold_scale, 150)).pack(side="left")

        # Save Results button (below properties)
        Button(self.control_frame, text="Save Results", command=self.save_results).pack(fill="x", pady=5)

    def setup_subcategories(self):
        """Set up controls for each subcategory."""
        for category, subcategories in self.categories.items():
            frame = self.subcategory_frames[category]
            for subcat in subcategories:
                if subcat in ["Binarization", "Other Preprocessing", "Row Threshold", "Column Threshold"]:
                    continue
                normalized_subcat = subcat.lower().replace(' ', '_')
                btn = Button(frame, text=f"Step: {subcat}", command=getattr(self, f"step_{normalized_subcat}"), width=15)
                btn.pack(fill="x", padx=30, pady=1)

    def select_images_or_folder(self):
        """Handle selection of individual images or a folder and display them in a grid."""
        # First, try selecting individual images
        files = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if files:
            self.image_paths = list(files)
        else:
            # If no files selected, try selecting a folder
            folder = filedialog.askdirectory()
            if folder:
                self.image_paths = [os.path.join(folder, f) for f in os.listdir(folder)
                                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            else:
                return  # No selection made

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
        self.back_button.pack_forget()  # Hide back button in grid view
        self.prev_button.pack()
        self.next_button.pack()

        # Clear the right panel
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        # Create scrollable canvas
        canvas = Canvas(self.image_frame)
        scrollbar = Scrollbar(self.image_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        self.thumbnail_photos = []
        thumbnail_size = (150, 150)

        # Calculate number of columns based on window width
        window_width = self.image_frame.winfo_width()
        if window_width <= 1:  # If not yet rendered, use an estimate
            window_width = 1200  # Based on default window width (1500 - 300 control panel)
        cols = max(1, window_width // (thumbnail_size[0] + 10))  # 10 for padding

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

        self.adjust_window_size()

    def display_single_image(self, img_path, index):
        """Display a single image in full quality."""
        self.is_grid_view = False
        self.current_image_index = index
        self.back_button.pack()
        self.prev_button.pack()
        self.next_button.pack()
        self.update_filepath_label()

        # Clear the right panel
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        # Load and display the full-quality image
        self.current_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if self.current_image is None:
            messagebox.showerror("Error", f"Failed to load image: {img_path}")
            return
        self.display_image = self.current_image.copy()
        self.processed_image = None
        self.segmentation = None
        self.detections = None
        self.tables = None
        self.mapping = None
        self.htr_results = None

        # Create canvas for single image
        self.canvas = Canvas(self.image_frame, bg="gray")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self.update_image)

        self.adjust_window_size()
        self.update_image()

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

    def adjust_window_size(self):
        """Adjust the window size based on the image dimensions with a cap to prevent it from being too large."""
        control_width = 300
        control_height = self.control_frame.winfo_reqheight()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        max_window_width = int(screen_width * 0.8)
        max_window_height = int(screen_height * 0.8)

        padding = 20
        if self.is_grid_view:
            window_width = min(control_width + 500, max_window_width)
            window_height = min(control_height + 500, max_window_height)
        else:
            img_height = self.current_image.shape[0]
            window_width = min(1500, max_window_width)
            desired_height = max(img_height, control_height) + padding
            window_height = min(desired_height, max_window_height)

        min_window_width = control_width + 400
        min_window_height = control_height + 50

        window_width = max(window_width, min_window_width)
        window_height = max(window_height, min_window_height)

        self.root.geometry(f"{window_width}x{window_height}")

    def toggle_category(self, category):
        """Toggle the visibility of the subcategory frame for the selected category."""
        if self.active_subcategory:
            self.subcategory_frames[self.active_subcategory].pack_forget()
        if self.active_subcategory != category:
            self.subcategory_frames[category].pack(fill="x")
            self.active_subcategory = category
        else:
            self.active_subcategory = None

        self.root.update()
        control_height = self.control_frame.winfo_reqheight()
        current_width = self.root.winfo_width()
        self.root.geometry(f"{current_width}x{max(control_height + 50, self.root.winfo_height())}")

    def update_binarization(self, value):
        """Update binarization threshold on-the-fly."""
        if self.current_image is not None and not self.is_grid_view:
            self.processed_image = self.processor.preprocess_image(
                self.image_paths[self.current_image_index],
                int(value),
                self.mask_offset_scale.get() if self.mask_offset_scale else 5
            )
            self.display_image = self.processed_image.copy()
            self.update_image()

    def update_mask_offset(self, value):
        """Update contour mask offset on-the-fly."""
        if self.current_image is not None and not self.is_grid_view:
            self.processed_image = self.processor.preprocess_image(
                self.image_paths[self.current_image_index],
                self.binarization_scale.get() if self.binarization_scale else 210,
                int(value)
            )
            self.display_image = self.processed_image.copy()
            self.update_image()

    def update_row_threshold(self, value):
        """Update row threshold on-the-fly (placeholder)."""
        if self.detections is not None and not self.is_grid_view:
            self.tables = self.processor.detect_table_structure(
                self.detections,
                int(value),
                self.col_threshold_scale.get() if self.col_threshold_scale else 150
            )
            self.display_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)
            if self.tables:
                for box1, box2 in self.tables:
                    x1, y1, x2, y2 = map(int, box1)
                    x3, y3, x4, y4 = map(int, box2)
                    cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.rectangle(self.display_image, (x3, y3), (x4, y4), (255, 0, 0), 2)
                    cv2.line(self.display_image, ((x1+x2)//2, (y1+y2)//2), ((x3+x4)//2, (y3+y4)//2), (0, 0, 255), 2)
            self.update_image()

    def update_col_threshold(self, value):
        """Update column threshold on-the-fly (placeholder)."""
        if self.detections is not None and not self.is_grid_view:
            self.tables = self.processor.detect_table_structure(
                self.detections,
                self.row_threshold_scale.get() if self.row_threshold_scale else 50,
                int(value)
            )
            self.display_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)
            if self.tables:
                for box1, box2 in self.tables:
                    x1, y1, x2, y2 = map(int, box1)
                    x3, y3, x4, y4 = map(int, box2)
                    cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.rectangle(self.display_image, (x3, y3), (x4, y4), (255, 0, 0), 2)
                    cv2.line(self.display_image, ((x1+x2)//2, (y1+y2)//2), ((x3+x4)//2, (y3+y4)//2), (0, 0, 255), 2)
            self.update_image()

    def reset_slider(self, scale, default_value):
        """Reset the slider to its default value."""
        scale.set(default_value)
        if scale == self.binarization_scale:
            self.update_binarization(default_value)
        elif scale == self.mask_offset_scale:
            self.update_mask_offset(default_value)
        elif scale == self.row_threshold_scale:
            self.update_row_threshold(default_value)
        elif scale == self.col_threshold_scale:
            self.update_col_threshold(default_value)

    def update_image(self, event=None):
        """Update the displayed image, resizing to fit the canvas."""
        if self.display_image is None or self.is_grid_view:
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 400, 400

        img_height, img_width = self.display_image.shape[:2]
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        resized_image = cv2.resize(self.display_image, (new_width, new_height))

        if len(resized_image.shape) == 2:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
        pil_image = Image.fromarray(resized_image)
        self.photo = ImageTk.PhotoImage(pil_image)

        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.photo, anchor="center")
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def step_segment_page(self):
        """Execute the page segmentation step and update the display."""
        if not self.image_paths or self.is_grid_view:
            messagebox.showerror("Error", "Please select an image from the grid!")
            return
        if self.processed_image is None:
            messagebox.showerror("Error", "Run Pre-process step first!")
            return
        processed_bgr = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)
        self.segmentation = self.processor.segment_page(processed_bgr)
        self.display_image = processed_bgr.copy()
        if self.segmentation is not None:
            for box in self.segmentation:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        self.update_image()

    def step_detect_words_symbols(self):
        """Execute the words/symbols detection step and update the display."""
        if not self.image_paths or self.is_grid_view:
            messagebox.showerror("Error", "Please select an image from the grid!")
            return
        if self.processed_image is None:
            messagebox.showerror("Error", "Run Pre-process step first!")
            return
        processed_bgr = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)
        self.detections = self.processor.detect_words_symbols(processed_bgr)
        self.display_image = processed_bgr.copy()
        if self.detections is not None:
            for box in self.detections:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        self.update_image()

    def step_detect_table_structure(self):
        """Execute the table structure detection step and update the display."""
        if not self.image_paths or self.is_grid_view:
            messagebox.showerror("Error", "Please select an image from the grid!")
            return
        if self.detections is None:
            messagebox.showerror("Error", "Run Detect Words Symbols step first!")
            return
        self.tables = self.processor.detect_table_structure(
            self.detections,
            self.row_threshold_scale.get() if self.row_threshold_scale else 50,
            self.col_threshold_scale.get() if self.col_threshold_scale else 150
        )
        self.display_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)
        if self.tables:
            for box1, box2 in self.tables:
                x1, y1, x2, y2 = map(int, box1)
                x3, y3, x4, y4 = map(int, box2)
                cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.rectangle(self.display_image, (x3, y3), (x4, y4), (255, 0, 0), 2)
                cv2.line(self.display_image, ((x1+x2)//2, (y1+y2)//2), ((x3+x4)//2, (y3+y4)//2), (0, 0, 255), 2)
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
            messagebox.showerror("Error", "Run Detect Words Symbols step first!")
            return
        self.htr_results = self.processor.htr(self.processed_image, self.detections)
        messagebox.showinfo("Info", f"HTR Results: {self.htr_results}")
        self.display_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)
        self.update_image()

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