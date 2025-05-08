#!/usr/bin/env python3
import glob
import json
import os
import random
import shutil
import cv2


def to_yolo_box(xmin, ymin, xmax, ymax, img_w, img_h):
    """Convert box to YOLO format: x_center, y_center, width, height (normalized)"""
    x_c = (xmin + xmax) / 2.0 / img_w
    y_c = (ymin + ymax) / 2.0 / img_h
    w   = (xmax - xmin) / img_w
    h   = (ymax - ymin) / img_h
    return x_c, y_c, w, h


def prepare_yolo(json_dir, img_dir, out_dir, train_frac, val_frac, exts):
    # Validate split fractions
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be less than 1.0")
    test_frac = 1.0 - train_frac - val_frac

    os.makedirs(out_dir, exist_ok=True)
    # 1) Collect class names
    classes = []
    json_paths = glob.glob(os.path.join(json_dir, "*.json"))
    for jpath in json_paths:
        data = json.load(open(jpath, 'r', encoding='utf-8'))
        for shape in data.get("shapes", []):
            lbl = shape["label"]
            if lbl not in classes:
                classes.append(lbl)

    # Write classes.txt
    with open(os.path.join(out_dir, "classes.txt"), "w") as f:
        f.write("\n".join(classes))

    # 2) Prepare folder structure
    for subset in ("train", "val", "test"):
        os.makedirs(os.path.join(out_dir, f"images/{subset}"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, f"labels/{subset}"), exist_ok=True)

    # 3) Split dataset
    random.shuffle(json_paths)
    total = len(json_paths)
    n_train = int(total * train_frac)
    n_val   = int(total * val_frac)

    subsets = {
        "train": json_paths[:n_train],
        "val":   json_paths[n_train:n_train + n_val],
        "test":  json_paths[n_train + n_val:]
    }

    # 4) Process each subset
    for subset, paths in subsets.items():
        for jpath in paths:
            base = os.path.splitext(os.path.basename(jpath))[0]
            # Find matching image
            img_path = None
            for ext in exts:
                cand = os.path.join(img_dir, base + ext)
                if os.path.isfile(cand):
                    img_path = cand
                    break
            if img_path is None:
                print(f"[WARN] No image for {base}, skipping")
                continue

            # Load image for size
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Failed to load {img_path}, skipping")
                continue
            h, w = img.shape[:2]

            # Read JSON and write YOLO label file
            data = json.load(open(jpath, 'r', encoding='utf-8'))
            label_lines = []
            for shape in data.get("shapes", []):
                pts = shape["points"]
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                x_c, y_c, bw, bh = to_yolo_box(xmin, ymin, xmax, ymax, w, h)
                cls_id = classes.index(shape["label"])
                label_lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

            lbl_out = os.path.join(out_dir, f"labels/{subset}", base + ".txt")
            with open(lbl_out, "w") as lf:
                lf.write("\n".join(label_lines))

            # Copy image to subset folder
            dst_img = os.path.join(out_dir, f"images/{subset}", os.path.basename(img_path))
            if not os.path.exists(dst_img):
                shutil.copy(img_path, dst_img)
            print(f"[OK] {subset}: {base}")

    # 5) Generate dataset YAML for YOLO
    yaml_path = os.path.join(out_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as yf:
        yf.write(f"train: {os.path.abspath(os.path.join(out_dir, 'images/train'))}\n")
        yf.write(f"val:   {os.path.abspath(os.path.join(out_dir, 'images/val'))}\n")
        yf.write(f"test:  {os.path.abspath(os.path.join(out_dir, 'images/test'))}\n")
        yf.write(f"nc: {len(classes)}\n")
        # Write names as YAML list
        yf.write("names: ")
        yf.write(json.dumps(classes))
        yf.write("\n")
    print(f"[OK] dataset.yaml created at {yaml_path}")


if __name__ == "__main__":
    prepare_yolo("../../materials/dataset/subparts_annotations_yolo",
                 "../../materials/dataset/images",
                 "../../materials/dataset/subparts_annotations_yolo/dataset",
                 0.7, 0.2, [".jpg", ".jpeg", ".png"])

