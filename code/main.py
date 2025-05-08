"""
===============================================================================
Main Script - Dataset Task Runner
===============================================================================

Supported Datasets (selected via `selected_dataset`):
- "components"               : Region segmentation using JSON and YOLO annotations
- "words_symbols"         : Word/Symbol detection using JSON and YOLO
- "classification_subparts": Image classification dataset (subparts only, no labels)

Supported Tasks (selected via `task`):
- "split"      : Splits each image in half (left/right) and saves to output_dir
- "split_annotation": Splits LabelMe JSON annotations into left/right halves
- "crop"       : Crops regions from images based on Labelme-style JSON annotations
- "visualize"  : Draws YOLO bounding boxes on images using YOLO .txt labels and dataset.yaml
- "train"      : Trains a YOLO model using a dataset.yaml configuration (Ultralytics YOLO)

Structure:
- Each dataset config contains:
  - paths: {
      images_dir      → path to input images,
      labels_dir      → path to YOLO labels (if needed),
      annotations_dir → path to JSON labels (if needed),
      dataset_yaml    → path to YOLO dataset.yaml file (for visualize/train),
      output_dir      → folder to save output files
    }

  - params: {
      model_name, epochs, imgsz, device, etc. (used for training)
    }
===============================================================================
"""

from dataset_preparation.split_images import split_images
from dataset_preparation.crop_from_json import crop_from_json
from dataset_preparation.dataset_visualization import visualize_yolo_annotations
from dataset_preparation.split_annotation import split_annotations_in_dir
from dataset_preparation.rename_labels import rename_labels_in_annotations  # NEW import

# Dataset configurations
datasets = {
    "components": {  # renamed from "regions"
        "paths": {
            "images_dir": "../data/images/pages",  # updated images path
            "annotations_dir": "../data/annotations/components",  # as per previous update
            "labels_dir": "../data/annotations/components",  # assuming labels_dir same as annotations_dir
            "dataset_yaml": "../data/dataset_yaml/page_segmentation.yaml",  # updated dataset yaml path
            "output_dir": "../data/images/components_cropped"
        },
        "params": {
            "model_name": "yolo11n.pt",
            "epochs": 50,
            "imgsz": 960,
            "device": "0",
            "save": True,
            "save_period": 1
        }
    },
    "words_symbols": {
        "paths": {
            "images_dir": "../data/images",  # updated images path
            "annotations_dir": "../data/annotations/words_symbols",  # updated annotations path
            "labels_dir": "../data/annotations/words_symbols",  # updated labels path
            "dataset_yaml": "../data/dataset_yaml/word_symbol_detection.yaml",  # updated dataset yaml path
            "output_dir": "../data/outputs/words_symbols"
        },
        "params": {
            "model_name": "yolo11n.pt",
            "epochs": 50,
            "imgsz": 960,
            "device": "0",
            "save": True,
            "save_period": 1
        }
    },
    "classification_subparts": {
        "paths": {
            "images_dir": "../data/images/classification",  # updated images path
            "output_dir": "../outputs/classification_split"
        },
        "params": {}
    }
}


# Task functions
def run_split(cfg):
    split_images(cfg["paths"]["images_dir"], cfg["paths"]["output_dir"])


def run_split_annotation(cfg):
    split_annotations_in_dir(
        annotations_dir=cfg["paths"]["annotations_dir"],
        output_dir=cfg["paths"]["output_dir"],
        crop_images=False  # Set True if you want to crop images as well
    )


def run_crop(cfg):
    crop_from_json(
        annotations_dir=cfg["paths"]["annotations_dir"],
        images_dir=cfg["paths"]["images_dir"],
        output_dir=cfg["paths"]["output_dir"]
    )


def run_visualize(cfg):
    visualize_yolo_annotations(
        images_dir=cfg["paths"]["images_dir"],
        labels_dir=cfg["paths"]["labels_dir"],
        dataset_yaml=cfg["paths"]["dataset_yaml"],
        output_dir=cfg["paths"]["output_dir"]
    )


def run_rename_labels(cfg):
    rename_labels_in_annotations(cfg["paths"]["annotations_dir"])


# Dispatcher
task_map = {
    "split": run_split,
    "split_annotation": run_split_annotation,
    "crop": run_crop,
    "visualize": run_visualize,
    "rename_labels": run_rename_labels,  # NEW task
}

# Task and dataset selection
selected_dataset = "components"
task = "crop"

# Run selected task
if selected_dataset not in datasets:
    print(f"❌ Dataset '{selected_dataset}' is not defined.")
elif task not in task_map:
    print(f"❌ Task '{task}' is not supported.")
else:
    try:
        task_map[task](datasets[selected_dataset])
    except KeyError as e:
        print(f"❌ Missing key for task '{task}': {e}")
    except Exception as e:
        print(f"❌ Error during task '{task}': {e}")
