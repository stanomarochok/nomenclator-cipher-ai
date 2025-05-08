import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

class ImageBinarizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Binarizer and Mask Drawer")
        self.image = None
        self.processed_image = None
        self.photo = None

        # GUI elements
        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()

        self.btn_load = tk.Button(root, text="Load Image", command=self.load_image)
        self.btn_load.pack()

        self.threshold_label = tk.Label(root, text="Binarization Threshold: 128")
        self.threshold_label.pack()
        self.threshold_slider = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, length=300, command=self.update_image)
        self.threshold_slider.set(128)
        self.threshold_slider.pack()

        self.padding_label = tk.Label(root, text="Mask Padding: 0")
        self.padding_label.pack()
        self.padding_slider = tk.Scale(root, from_=0, to=50, orient=tk.HORIZONTAL, length=300, command=self.update_image)
        self.padding_slider.set(0)
        self.padding_slider.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.image)
            self.update_image()

    def display_image(self, img):
        # Convert numpy array to PIL Image
        if len(img.shape) == 2:  # Grayscale
            img_pil = Image.fromarray(img, mode='L')
        else:  # Color (BGR)
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # Resize for display while maintaining aspect ratio
        max_size = (800, 600)
        img_pil.thumbnail(max_size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img_pil)
        self.canvas.config(width=self.photo.width(), height=self.photo.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def update_image(self, *args):
        if self.image is None:
            return

        # Binarize image
        threshold = self.threshold_slider.get()
        self.threshold_label.config(text=f"Binarization Threshold: {threshold}")
        # Add noise reduction
        blurred = cv2.GaussianBlur(self.image, (5, 5), 0)
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)  # Inverted for dark objects

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert binary to color for display
        display_img = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)  # Use original grayscale for display
        padding = self.padding_slider.get()
        self.padding_label.config(text=f"Mask Padding: {padding}")

        # Dilate contours to simulate padding and merge nearby contours
        if padding > 0:
            kernel = np.ones((padding * 2 + 1, padding * 2 + 1), np.uint8)
            dilated = cv2.dilate(binary, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw polygonal masks as filled shapes with transparency
        image_area = self.image.shape[0] * self.image.shape[1]
        overlay = display_img.copy()
        for contour in contours:
            # Filter contours by area to avoid noise
            area = cv2.contourArea(contour)
            if 50 < area < image_area * 0.5:  # Adjusted for smaller objects
                # Simplify contour to polygon
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                # Draw filled polygon in red (BGR: 0, 0, 255)
                cv2.fillPoly(overlay, [approx], (0, 0, 255))

        # Apply transparency to the overlay
        cv2.addWeighted(overlay, 0.4, display_img, 0.6, 0, display_img)

        self.processed_image = display_img
        self.display_image(self.processed_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageBinarizerApp(root)
    root.mainloop()