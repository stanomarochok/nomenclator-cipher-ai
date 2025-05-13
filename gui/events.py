# gui/events.py

from tkinter import Canvas, Scrollbar, Frame, Label, messagebox
from PIL import Image, ImageTk
import cv2
import os


def display_image_grid(self):
    self.is_grid_view = True
    self.back_button.pack_forget()
    self.prev_button.pack()
    self.next_button.pack()

    self.canvas.delete("all")

    grid_canvas = Canvas(self.image_frame)
    grid_canvas.pack(side="top", fill="both", expand=True)
    scrollbar = Scrollbar(self.image_frame, orient="vertical", command=grid_canvas.yview)
    scrollbar.pack(side="right", fill="y")
    scrollable_frame = Frame(grid_canvas)

    scrollable_frame.bind("<Configure>", lambda e: grid_canvas.configure(scrollregion=grid_canvas.bbox("all")))
    grid_canvas.configure(yscrollcommand=scrollbar.set)
    grid_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    self.thumbnail_photos = []
    thumbnail_size = (150, 150)
    window_width = self.image_frame.winfo_width() or 800
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

    self.grid_canvas = grid_canvas


def display_single_image(self, img_path, index):
    self.is_grid_view = False
    self.current_image_index = index
    self.back_button.pack()
    self.prev_button.pack()
    self.next_button.pack()
    self.update_filepath_label()

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

    self.canvas.delete("all")
    bind_canvas_events(self)

    # 🛠 Postpone update_image until canvas has its final size
    self.root.after(50, self.update_image)


def bind_canvas_events(self):
    self.canvas.bind("<Configure>", self.update_image)
    self.canvas.bind("<Button-1>", self.start_selection)
    self.canvas.bind("<B1-Motion>", self.update_selection)
    self.canvas.bind("<ButtonRelease-1>", self.end_selection)
    self.canvas.bind("<MouseWheel>", self.zoom_with_scroll)
    self.canvas.bind("<Button-4>", self.zoom_with_scroll)
    self.canvas.bind("<Button-5>", self.zoom_with_scroll)
    self.canvas.bind("<ButtonPress-2>", self.start_pan)
    self.canvas.bind("<B2-Motion>", self.do_pan)
    self.canvas.bind("<ButtonRelease-2>", self.end_pan)


def select_images_or_folder(self):
    from tkinter import filedialog

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


def show_previous_image(self):
    if self.is_grid_view or not self.image_paths:
        return
    self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
    self.display_single_image(self.image_paths[self.current_image_index], self.current_image_index)


def show_next_image(self):
    if self.is_grid_view or not self.image_paths:
        return
    self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
    self.display_single_image(self.image_paths[self.current_image_index], self.current_image_index)


def update_filepath_label(self):
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
