import os
import json
from collections import defaultdict


def print_all_labels(json_dir):
    # Dictionary to hold label counts
    label_counts = defaultdict(int)
    # Iterate over all JSON files in the directory
    for json_file in os.listdir(json_dir):
        if json_file.endswith(".json"):
            json_path = os.path.join(json_dir, json_file)
            with open(json_path, "r") as f:
                data = json.load(f)

            # Extract labels from the JSON file
            for shape in data.get("shapes", []):
                label = shape.get("label", "no-label")
                label_counts[label] += 1

    return label_counts


# Directory containing the JSON files
json_dir = "../../materials/dataset/annotations_symbol_word"
# Find labels and their counts
labels = print_all_labels(json_dir)

# Print results
print("Labels and their counts:")
for label, count in labels.items():
    print(f"{label}: {count}")
