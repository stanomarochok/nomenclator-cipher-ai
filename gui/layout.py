# gui/layout.py

from tkinter import Frame, Button, Scale, Label, StringVar, OptionMenu, Canvas


def setup_gui(self):
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
        "Processing Steps": ["Segment Page", "Detect Words Symbols", "All Objects Detection",
                             "Detect Table Structure", "Map Plaintext to Ciphertext", "HTR", "Classification"]
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

    Button(self.control_frame, text="Save Results", command=self.save_results).pack(fill="x", pady=5, side="bottom")


def setup_subcategories(self):
    for category, subcategories in self.categories.items():
        frame = self.subcategory_frames[category]
        for subcat in subcategories:
            normalized_subcat = subcat.lower().replace(' ', '_')
            method_name = f"step_{normalized_subcat}"
            if hasattr(self, method_name):
                btn = Button(frame, text=f"Step: {subcat}", command=getattr(self, method_name), width=15)
                btn.pack(fill="x", padx=30, pady=1)

            properties_frame = Frame(frame)
            properties_frame.pack(fill="x", padx=40)

            if subcat == "Binarization":
                self.binarization_scale = Scale(properties_frame, from_=0, to=255, orient="horizontal",
                                                command=self.update_binarization)
                self.binarization_scale.set(0)
                Label(properties_frame, text="Binarization").pack(side="left")
                self.binarization_scale.pack(side="left", padx=5, expand=True)
                Button(properties_frame, text="Reset",
                       command=lambda: [self.binarization_scale.set(0), self.update_binarization(0)]).pack(side="left")

            elif subcat == "Gaussian Blur":
                self.gaussian_scale = Scale(properties_frame, from_=0, to=5, orient="horizontal",
                                            command=self.update_gaussian_blur)
                self.gaussian_scale.set(0)
                Label(properties_frame, text="Gaussian Blur").pack(side="left")
                self.gaussian_scale.pack(side="left", padx=5, expand=True)
                Button(properties_frame, text="Reset",
                       command=lambda: [self.gaussian_scale.set(0), self.update_gaussian_blur(0)]).pack(side="left")

            elif subcat == "Contrast Adjustment":
                self.contrast_scale = Scale(properties_frame, from_=0, to=2, orient="horizontal",
                                            command=self.update_contrast, resolution=0.1)
                self.contrast_scale.set(0)
                Label(properties_frame, text="Contrast").pack(side="left")
                self.contrast_scale.pack(side="left", padx=5, expand=True)
                Button(properties_frame, text="Reset",
                       command=lambda: [self.contrast_scale.set(0), self.update_contrast(0)]).pack(side="left")

            elif subcat == "Sharpening":
                self.sharpen_scale = Scale(properties_frame, from_=0, to=2, orient="horizontal",
                                           command=self.update_sharpening, resolution=0.1)
                self.sharpen_scale.set(0)
                Label(properties_frame, text="Sharpening").pack(side="left")
                self.sharpen_scale.pack(side="left", padx=5, expand=True)
                Button(properties_frame, text="Reset",
                       command=lambda: [self.sharpen_scale.set(0), self.update_sharpening(0)]).pack(side="left")

            elif subcat == "All Objects Detection":
                Label(properties_frame, text="Algorithm:").pack(side="left")
                self.detection_algo_var = StringVar(self.root)
                self.detection_algo_var.set("Contour Masks")
                OptionMenu(properties_frame, self.detection_algo_var, "Contour Masks").pack(side="left")

                Label(properties_frame, text="Contour Mask Offset").pack(side="left")
                self.mask_offset_scale = Scale(properties_frame, from_=0, to=20, orient="horizontal",
                                               command=self.update_mask_offset)
                self.mask_offset_scale.set(5)
                self.mask_offset_scale.pack(side="left", padx=5, expand=True)
                Button(properties_frame, text="Reset",
                       command=lambda: [self.mask_offset_scale.set(5), self.update_mask_offset(5)]).pack(side="left")

            elif subcat == "Classification":
                Label(properties_frame, text="Model:").pack(side="left")
                self.model_var = StringVar(self.root)
                self.model_var.set("resnet50")
                model_options = ["alexnet", "densenet201", "efficientnet_b7", "inception_v3", "resnet50", "custom"]
                OptionMenu(properties_frame, self.model_var, *model_options).pack(side="left")
                Button(properties_frame, text="Classify Region", command=self.classify_selected_region).pack(side="left", padx=5)
