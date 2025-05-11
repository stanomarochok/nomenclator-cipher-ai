import os
import cv2
import numpy as np
from tkinter import Tk, Frame, Button, Scale, Label, filedialog, messagebox, Canvas
from PIL import Image, ImageTk
from processor import CipherKeyProcessor

class CipherKeyApp:
    def __init__(self, root):
        """Initialize the GUI application with a properly sized window."""
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

        # Set up the GUI first to calculate control panel height
        self.setup_gui()

        # Calculate initial window size based on control panel height
        control_width = 300
        control_height = 7 * 40 + 4 * 60 + 20  # 7 buttons (40px each), 4 sliders+labels (60px each), padding (20px)
        initial_width = control_width + 400  # Control panel + initial canvas width
        initial_height = control_height + 50  # Control panel height + padding
        self.root.geometry(f"{initial_width}x{initial_height}")

    def setup_gui(self):
        """Set up the GUI with file selection, parameter sliders, and image display."""
        main_frame = Frame(self.root, padx=5, pady=5)
        main_frame.pack(fill="both", expand=True)

        # Left frame for controls (fixed width)
        self.control_frame = Frame(main_frame, width=300)
        self.control_frame.pack(side="left", fill="y", padx=5)
        self.control_frame.pack_propagate(False)

        # Right frame for image display
        self.image_frame = Frame(main_frame)
        self.image_frame.pack(side="right", fill="both", expand=True)
        self.canvas = Canvas(self.image_frame, bg="gray")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self.update_image)

        # File selection
        Button(self.control_frame, text="Select Files/Folder", command=self.select_files).pack(fill="x")

        # Parameter sliders
        params_frame = Frame(self.control_frame)
        params_frame.pack(fill="x")
        Label(params_frame, text="Binarization Threshold").pack()
        self.binarization_scale = Scale(params_frame, from_=0, to=255, orient="horizontal", command=self.update_preprocess)
        self.binarization_scale.set(210)
        self.binarization_scale.pack(fill="x")
        Label(params_frame, text="Contour Mask Offset").pack()
        self.mask_offset_scale = Scale(params_frame, from_=0, to=20, orient="horizontal", command=self.update_preprocess)
        self.mask_offset_scale.set(5)
        self.mask_offset_scale.pack(fill="x")
        Label(params_frame, text="Table Row Threshold").pack()
        self.row_threshold_scale = Scale(params_frame, from_=10, to=100, orient="horizontal")
        self.row_threshold_scale.set(50)
        self.row_threshold_scale.pack(fill="x")
        Label(params_frame, text="Table Col Threshold").pack()
        self.col_threshold_scale = Scale(params_frame, from_=50, to=300, orient="horizontal")
        self.col_threshold_scale.set(150)
        self.col_threshold_scale.pack(fill="x")

        # Step-by-step processing buttons
        buttons_frame = Frame(self.control_frame)
        buttons_frame.pack(fill="x")
        Button(buttons_frame, text="Step 1: Pre-process", command=self.step_preprocess).pack(fill="x")
        Button(buttons_frame, text="Step 2: Segment Page", command=self.step_segment).pack(fill="x")
        Button(buttons_frame, text="Step 3: Detect Words/Symbols", command=self.step_detect).pack(fill="x")
        Button(buttons_frame, text="Step 4: Detect Table Structure", command=self.step_table).pack(fill="x")
        Button(buttons_frame, text="Step 5: Map Plaintext to Ciphertext", command=self.step_map).pack(fill="x")
        Button(buttons_frame, text="Step 6: HTR", command=self.step_htr).pack(fill="x")
        Button(buttons_frame, text="Save Results", command=self.save_results).pack(fill="x")

    def select_files(self):
        """Handle file or folder selection and display the first image immediately with window adjustment."""
        files = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if files:
            self.image_paths = list(files)
        else:
            folder = filedialog.askdirectory()
            if folder:
                self.image_paths = [os.path.join(folder, f) for f in os.listdir(folder)
                                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if self.image_paths:
            self.current_image = cv2.imread(self.image_paths[0], cv2.IMREAD_GRAYSCALE)
            if self.current_image is None:
                messagebox.showerror("Error", f"Failed to load image: {self.image_paths[0]}")
                self.image_paths = []
                return
            self.display_image = self.current_image.copy()
            self.processed_image = None
            self.segmentation = None
            self.detections = None
            self.tables = None
            self.mapping = None
            self.htr_results = None

            # Adjust window size based on image dimensions with a cap
            self.adjust_window_size()
            self.update_image()

    def adjust_window_size(self):
        """Adjust the window size based on the image dimensions with a cap to prevent it from being too large."""
        if self.current_image is None:
            return

        # Get image dimensions
        img_height, img_width = self.current_image.shape[:2]

        # Control panel width (fixed)
        control_width = 300

        # Estimate control panel height
        control_height = 7 * 40 + 4 * 60 + 20  # 7 buttons, 4 sliders+labels, padding

        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Set maximum window size (e.g., 80% of screen dimensions)
        max_window_width = int(screen_width * 0.8)
        max_window_height = int(screen_height * 0.8)

        # Calculate desired window size to fit image
        padding = 20
        desired_width = img_width + control_width + padding
        desired_height = max(img_height, control_height) + padding

        # Cap the window size to the maximum
        window_width = min(desired_width, max_window_width)
        window_height = min(desired_height, max_window_height)

        # Set minimum window size to ensure controls are visible
        min_window_width = control_width + 400
        min_window_height = control_height + 50

        window_width = max(window_width, min_window_width)
        window_height = max(window_height, min_window_height)

        self.root.geometry(f"{window_width}x{window_height}")

    def update_preprocess(self, *args):
        """Update the preprocessing step in real-time."""
        if self.current_image is not None:
            self.processed_image = self.processor.preprocess_image(
                self.image_paths[0],
                self.binarization_scale.get(),
                self.mask_offset_scale.get()
            )
            self.display_image = self.processed_image.copy()
            self.update_image()

    def update_image(self, event=None):
        """Update the displayed image, resizing to fit the canvas."""
        if self.display_image is None:
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

    def step_preprocess(self):
        """Execute the pre-processing step and update the display."""
        if not self.image_paths:
            messagebox.showerror("Error", "No images selected!")
            return
        self.processed_image = self.processor.preprocess_image(
            self.image_paths[0],
            self.binarization_scale.get(),
            self.mask_offset_scale.get()
        )
        self.display_image = self.processed_image.copy()
        self.update_image()

    def step_segment(self):
        """Execute the page segmentation step and update the display."""
        if not self.image_paths:
            messagebox.showerror("Error", "No images selected!")
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

    def step_detect(self):
        """Execute the words/symbols detection step and update the display."""
        if not self.image_paths:
            messagebox.showerror("Error", "No images selected!")
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

    def step_table(self):
        """Execute the table structure detection step and update the display."""
        if not self.image_paths:
            messagebox.showerror("Error", "No images selected!")
            return
        if self.detections is None:
            messagebox.showerror("Error", "Run Detect Words/Symbols step first!")
            return
        self.tables = self.processor.detect_table_structure(
            self.detections,
            self.row_threshold_scale.get(),
            self.col_threshold_scale.get()
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

    def step_map(self):
        """Execute the mapping step."""
        if not self.image_paths:
            messagebox.showerror("Error", "No images selected!")
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
        if not self.image_paths:
            messagebox.showerror("Error", "No images selected!")
            return
        if self.detections is None:
            messagebox.showerror("Error", "Run Detect Words/Symbols step first!")
            return
        self.htr_results = self.processor.htr(self.processed_image, self.detections)
        messagebox.showinfo("Info", f"HTR Results: {self.htr_results}")
        self.display_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)
        self.update_image()

    def save_results(self):
        """Save the processed image and results."""
        if not self.image_paths:
            messagebox.showerror("Error", "No images selected!")
            return
        if self.display_image is None:
            messagebox.showerror("Error", "No processed image to save!")
            return
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, f"processed_{os.path.basename(self.image_paths[0])}"), self.display_image)
        if self.mapping:
            with open(os.path.join(output_dir, f"mapping_{os.path.basename(self.image_paths[0])}.txt"), "w") as f:
                f.write(str(self.mapping))
        if self.htr_results:
            with open(os.path.join(output_dir, f"htr_{os.path.basename(self.image_paths[0])}.txt"), "w") as f:
                f.write(str(self.htr_results))
        messagebox.showinfo("Info", "Results saved successfully!")