import json
import os
from copy import deepcopy
from PIL import Image


def clamp(val, lo, hi):
    """Clamp val between lo and hi."""
    return max(lo, min(val, hi))


def split_annotation(json_path, output_dir, crop_images=False):
    """
    Split a single LabelMe JSON annotation into left/right halves and write outputs to output_dir.

    Args:
        json_path (str): Path to the LabelMe JSON annotation file.
        output_dir (str): Directory to save the split JSON files (and optionally images).
        crop_images (bool): Whether to crop and save the corresponding image halves.
    """
    # Load JSON annotation
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Get image dimensions
    width = data.get('imageWidth')
    height = data.get('imageHeight')
    image_path = data.get('imagePath')

    # If width or height missing, try to load image to get size
    if (width is None or height is None) and image_path:
        img_full_path = os.path.join(os.path.dirname(json_path), image_path)
        img_full = Image.open(img_full_path)
        width, height = img_full.size
    else:
        img_full = None

    if width is None or height is None:
        raise ValueError(f"Couldn't determine dimensions for {json_path}")

    mid = width // 2
    halves = {'left': (0, mid), 'right': (mid, width)}

    base_name = os.path.splitext(os.path.basename(json_path))[0]
    img_root, img_ext = (os.path.splitext(os.path.basename(image_path))
                         if image_path else (None, None))

    for side, (xmin, xmax) in halves.items():
        new = deepcopy(data)
        new_shapes = []

        # Update image metadata
        if image_path:
            new['imagePath'] = f"{img_root}_{side}{img_ext}"
        new['imageWidth'] = xmax - xmin
        new['imageHeight'] = height

        # Process shapes: keep only those that intersect the half
        for shape in data.get('shapes', []):
            points = shape.get('points', [])
            # Check if any point is inside the half (xmin <= x < xmax)
            if any(x >= xmin and x < xmax for x, y in points):
                shp = deepcopy(shape)
                # Clamp points to the half and shift x coordinates
                shp['points'] = [[clamp(x, xmin, xmax) - xmin, y] for x, y in points]
                new_shapes.append(shp)

        new['shapes'] = new_shapes

        # Write split JSON annotation
        out_file = os.path.join(output_dir, f"{base_name}_{side}.json")
        with open(out_file, 'w') as f:
            json.dump(new, f, indent=2)
        print(f"Wrote annotation: {out_file}")

        # Optionally crop and save image half
        if crop_images and img_full:
            cropped = img_full.crop((xmin, 0, xmax, height))
            out_img = os.path.join(output_dir, f"{img_root}_{side}{img_ext}")
            cropped.save(out_img)
            print(f"Wrote image slice: {out_img}")


def split_annotations_in_dir(annotations_dir, output_dir, crop_images=False):
    """
    Split all LabelMe JSON annotations in a directory into left/right halves.

    Args:
        annotations_dir (str): Directory containing JSON annotation files.
        output_dir (str): Directory to save split JSON files (and optionally images).
        crop_images (bool): Whether to crop and save image halves.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for fname in os.listdir(annotations_dir):
        if not fname.lower().endswith('.json'):
            continue
        file_path = os.path.join(annotations_dir, fname)
        try:
            split_annotation(file_path, output_dir, crop_images=crop_images)
            # Do NOT remove original annotation file
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
