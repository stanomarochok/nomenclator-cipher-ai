import os
import json
from glob import glob


def rename_labels_in_annotations(annotations_dir):
    """
    Rename labels in JSON annotation files according to a predefined map.

    Args:
        annotations_dir (str): Directory containing JSON annotation files.
    """
    label_map = {
        "key-nomen-no-cat": "key-nomen-no-category",
        "key-nomen-unknown": "key-nomen-no-category",
        "key-nomen-vert-pairs": "key-nomen-vertical-pairs",
    }

    for json_file in glob(os.path.join(annotations_dir, "*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        updated = False
        for shape in data.get("shapes", []):
            old_label = shape.get("label")
            if old_label in label_map:
                shape["label"] = label_map[old_label]
                updated = True
                print(f"🔁 {os.path.basename(json_file)}: {old_label} → {shape['label']}")

        if updated:
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    print("✅ Label renaming complete.")


if __name__ == "__main__":
    # Default directory if run standalone
    annotations_dir = "../../data/annotations/components"
    rename_labels_in_annotations(annotations_dir)
