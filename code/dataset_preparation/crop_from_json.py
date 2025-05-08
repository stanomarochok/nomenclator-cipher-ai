"""
------------------------------------------------------------------------------
Function: crop_from_json
------------------------------------------------------------------------------

Description:
  Crops labeled rectangular regions from input images using LabelMe-style
  JSON annotations. Each cropped region is saved in a label-named subfolder
  inside `output_dir`.

Inputs:
  - annotations_dir: directory containing .json files with region annotations
  - images_dir     : directory containing corresponding images
  - output_dir     : directory where cropped regions will be saved

Outputs:
  - Cropped regions saved as individual JPEG images into:
      output_dir/<label>/<original_image>_<region_index>.jpg

Notes:
  - Automatically creates label subfolders inside output_dir.
  - Skips empty crops or missing images.
  - Saves JPEGs with quality=50 (for size efficiency).
------------------------------------------------------------------------------
"""

import os
import json
import cv2
from glob import glob


def crop_from_json(annotations_dir, images_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    annotation_files = glob(os.path.join(annotations_dir, "*.json"))
    print(f"📦 Found {len(annotation_files)} annotation files in '{annotations_dir}'")

    for annotation_path in annotation_files:
        with open(annotation_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        image_name = annotations.get("imagePath")
        if not image_name:
            print(f"⚠️  Skipping {annotation_path}: no imagePath field")
            continue

        image_path = os.path.join(images_dir, image_name)
        if not os.path.exists(image_path):
            print(f"⚠️  Image not found: {image_path}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️  Failed to load image: {image_path}")
            continue

        shapes = annotations.get("shapes", [])
        print(f"📂 Processing '{image_name}' with {len(shapes)} regions")

        for idx, shape in enumerate(shapes):
            label = shape.get("label", "unknown")
            points = shape.get("points", [])

            if len(points) < 2:
                print(f"  ⚠️  Invalid points for region {idx} in {image_name}")
                continue

            try:
                x1, y1 = map(int, points[0])
                x2, y2 = map(int, points[1])
            except Exception as e:
                print(f"  ⚠️  Error reading coordinates for region {idx}: {e}")
                continue

            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])
            crop = image[y_min:y_max, x_min:x_max]

            if crop.size == 0:
                print(f"  ⚠️  Empty crop for region {idx} (label: {label})")
                continue

            label_folder = os.path.join(output_dir, label)
            os.makedirs(label_folder, exist_ok=True)

            output_filename = f"{os.path.splitext(image_name)[0]}_{idx}.jpg"
            output_path = os.path.join(label_folder, output_filename)

            success = cv2.imwrite(output_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 50])
            if success:
                print(f"  ✅ Saved: {output_path}")
            else:
                print(f"  ❌ Failed to save: {output_path}")

    print("\n✅ Cropping complete.")
