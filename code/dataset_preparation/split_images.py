import os
import cv2
from glob import glob


def split_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = [f for ext in extensions for f in glob(os.path.join(input_dir, f"*{ext}"))]

    print(f"Found {len(image_files)} images in '{input_dir}' to split.\n")

    for image_path in image_files:
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)

        image = cv2.imread(image_path)
        if image is None:
            print(f"✗ Could not read image: {filename}")
            continue

        h, w = image.shape[:2]
        mid = w // 2

        left = image[:, :mid]
        right = image[:, mid:]

        left_path = os.path.join(output_dir, f"{name}_left{ext}")
        right_path = os.path.join(output_dir, f"{name}_right{ext}")

        success_left = cv2.imwrite(left_path, left)
        success_right = cv2.imwrite(right_path, right)

        if success_left and success_right:
            print(f"✓ Split {filename} → {name}_left{ext}, {name}_right{ext}")
        else:
            print(f"✗ Failed to save split for {filename}")

    print("\n✅ Image splitting complete.")
