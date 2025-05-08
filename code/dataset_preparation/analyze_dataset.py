import os
from collections import Counter
import yaml

# Set paths
# dataset_dir = "../../materials/dataset/word_symbol_annotations_yolo/yolo_dataset_2"
dataset_dir = "../../materials/dataset/subparts_annotations_yolo/dataset"
yaml_file = os.path.join(dataset_dir, 'dataset.yaml')

# Load classes
with open(yaml_file, 'r') as f:
    data = yaml.safe_load(f)
class_names = data['names']
num_classes = len(class_names)

# Define subsets
subsets = ['train', 'val', 'test']
subset_stats = {}

# Process each subset
for subset in subsets:
    labels_dir = os.path.join(dataset_dir, 'labels', subset)
    if not os.path.exists(labels_dir):
        continue

    class_counter = Counter()
    image_counter = 0

    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            image_counter += 1
            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, 'r') as lf:
                lines = lf.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        cls_id = int(parts[0])
                        class_counter[cls_id] += 1

    total_objects = sum(class_counter.values())
    avg_objects_per_image = total_objects / image_counter if image_counter else 0

    subset_stats[subset] = {
        'num_images': image_counter,
        'total_objects': total_objects,
        'avg_objects_per_image': avg_objects_per_image,
        'class_counter': class_counter
    }

# Print dataset details
print("YOLO Dataset Report")
print("--------------------\n")

total_images = 0
total_objects = 0
global_class_counter = Counter()

for subset, stats in subset_stats.items():
    print(f"Subset: {subset.upper()}")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Class names: {class_names}")
    print(f"  - Number of images: {stats['num_images']}")
    print(f"  - Total number of labeled objects: {stats['total_objects']}")
    print(f"  - Average number of objects per image: {stats['avg_objects_per_image']:.2f}")
    print(f"  - Class distribution:")

    for cls_id, count in stats['class_counter'].items():
        percentage = (count / stats['total_objects']) * 100 if stats['total_objects'] else 0
        print(f"      * {class_names[cls_id]}: {count} objects ({percentage:.2f}%)")

    print()

    total_images += stats['num_images']
    total_objects += stats['total_objects']
    global_class_counter.update(stats['class_counter'])

# Global totals
print("TOTAL (ALL SUBSETS COMBINED)")
print(f"  - Number of images: {total_images}")
print(f"  - Total number of labeled objects: {total_objects}")
print(f"  - Average number of objects per image: {total_objects/total_images:.2f}")

print(f"  - Overall class distribution:")
for cls_id, count in global_class_counter.items():
    percentage = (count / total_objects) * 100 if total_objects else 0
    print(f"      * {class_names[cls_id]}: {count} objects ({percentage:.2f}%)")

# Imbalance warning
if global_class_counter:
    max_class = max(global_class_counter.values())
    min_class = min(global_class_counter.values())
    if max_class > 2 * min_class:
        print("\n⚠️  Class imbalance detected: the most frequent class appears at least twice as often as the rarest.")
