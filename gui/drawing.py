# gui/drawing.py

import cv2
from PIL import Image, ImageTk


def update_image(self, event=None):
    if self.display_image is None or self.is_grid_view:
        self.canvas.delete("all")
        return

    canvas_width = self.canvas.winfo_width() or 800
    canvas_height = self.canvas.winfo_height() or 1200
    img_height, img_width = self.display_image.shape[:2]

    # Only initialize zoom once (when image is first shown)
    if not hasattr(self, "_zoom_initialized") or not self._zoom_initialized:
        self.scale_factor = min(canvas_width / img_width, canvas_height / img_height) * 0.9
        self._zoom_initialized = True

    new_width = int(img_width * self.scale_factor)
    new_height = int(img_height * self.scale_factor)
    resized_image = cv2.resize(self.display_image, (new_width, new_height))

    if len(resized_image.shape) == 2:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
    pil_image = Image.fromarray(resized_image)
    self.photo = ImageTk.PhotoImage(pil_image)

    self.canvas.delete("all")
    self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.photo)

    # Draw segmentation boxes
    if self.segmentation is not None and len(self.segmentation) > 0 and not self.is_selecting:
        for box in self.segmentation:
            x1, y1, x2, y2 = [b * self.scale_factor for b in box]
            x1 += self.offset_x
            y1 += self.offset_y
            x2 += self.offset_x
            y2 += self.offset_y
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="green", width=2)

    # Draw classification box (only one)
    for idx, box in enumerate(self.bounding_boxes):
        x1, y1, x2, y2 = [coord * self.scale_factor for coord in box]
        x1 += self.offset_x
        y1 += self.offset_y
        x2 += self.offset_x
        y2 += self.offset_y
        color = "blue" if idx == self.selected_box_index else "red"
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags=f"box_{idx}")

    self.canvas.config(scrollregion=self.canvas.bbox("all"))


def zoom_with_scroll(self, event):
    if event.state & 0x4:
        if event.delta > 0 or event.num == 4:
            self.zoom_in()
        elif event.delta < 0 or event.num == 5:
            self.zoom_out()


def zoom_in(self):
    old_scale = self.scale_factor
    self.scale_factor = min(3.0, self.scale_factor * 1.1)

    # Adjust offset to maintain zoom center
    canvas_center_x = self.canvas.winfo_width() / 2
    canvas_center_y = self.canvas.winfo_height() / 2
    self.offset_x = canvas_center_x - (canvas_center_x - self.offset_x) * (self.scale_factor / old_scale)
    self.offset_y = canvas_center_y - (canvas_center_y - self.offset_y) * (self.scale_factor / old_scale)

    self.update_image()


def zoom_out(self):
    old_scale = self.scale_factor
    self.scale_factor = max(0.1, self.scale_factor / 1.1)

    canvas_center_x = self.canvas.winfo_width() / 2
    canvas_center_y = self.canvas.winfo_height() / 2
    self.offset_x = canvas_center_x - (canvas_center_x - self.offset_x) * (self.scale_factor / old_scale)
    self.offset_y = canvas_center_y - (canvas_center_y - self.offset_y) * (self.scale_factor / old_scale)

    self.update_image()


