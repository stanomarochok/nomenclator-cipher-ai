"""
------------------------------------------------------------------------------
Function: visualize_yolo_annotations
------------------------------------------------------------------------------

Description:
  Loads a YOLO-style dataset and overlays bounding boxes on images using
  corresponding `.txt` label files and a class list defined in dataset.yaml.
  Annotated images are saved to `output_dir`.

Inputs:
  - images_dir    : folder with original images
  - labels_dir    : folder with YOLO-format `.txt` annotation files
  - dataset_yaml  : path to a .yaml file containing class names under 'names'
  - output_dir    : directory where annotated images will be saved

Output:
  - Each image saved to output_dir with bounding boxes and class labels drawn

------------------------------------------------------------------------------
"""

import os
import cv2
import random
import yaml
import matplotlib
matplotlib.use('TkAgg')  # For PyCharm or local GUI compatibility
import matplotlib.pyplot as plt


def visualize_yolo_annotations(images_dir, labels_dir, dataset_yaml, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(dataset_yaml):
        print(f"❌ dataset.yaml not found at {dataset_yaml}")
        return

    with open(dataset_yaml, 'r') as f:
        data = yaml.safe_load(f)

    class_names = data.get("names", [])
    print(f"✅ Loaded {len(class_names)} class names.")

    colors = {}

    def get_color(cls_id):
        if cls_id not in colors:
            colors[cls_id] = [random.randint(0, 255) for _ in range(3)]
        return colors[cls_id]

    def get_label(cls_id):
        return class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"

    def draw_boxes(img_path, label_path):
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ Warning: could not read image {img_path}")
            return None

        height, width, _ = image.shape

        if not os.path.exists(label_path):
            print(f"⚠️ Warning: missing label file for {img_path}")
            return image

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            x_center = float(parts[1]) * width
            y_center = float(parts[2]) * height
            box_width = float(parts[3]) * width
            box_height = float(parts[4]) * height

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            color = get_color(cls_id)
            label = get_label(cls_id)

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=4)

            # Draw label background + text
            font_scale = 2.0
            font_thickness = 3
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        return image

    all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for img_file in all_images:
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')

        annotated_img = draw_boxes(img_path, label_path)

        if annotated_img is not None:
            output_path = os.path.join(output_dir, img_file)
            cv2.imwrite(output_path, annotated_img)
            print(f"💾 Saved annotated image: {output_path}")

    print(f"\n✅ All annotated images saved to: {output_dir}")
