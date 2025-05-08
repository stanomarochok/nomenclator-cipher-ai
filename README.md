# Historical Document Processing Pipeline

This project provides a modular pipeline for processing annotated images of historical documents. It supports tasks like page segmentation, word/symbol detection, classification of document subparts, and visualization and training of YOLO-based models.

---

## 📁 Project Structure

project_root/

├── data/

│ ├── images/ # Shared source images

│ └── annotations/

│ ├── page_segmentation/ # JSON annotations for segmentation

│ ├── word_symbol_detection/ # JSON or YOLO annotations

│ └── classification_subparts/ # (Optional) classification labels

├── outputs/

│ ├── page_segmentation/

│ ├── word_symbol_detection/

│ └── classification_subparts/

├── src/

│ ├── preprocessing/

│ │ ├── split_images.py

│ │ ├── crop_from_json.py

│ ├── training/

│ │ └── train_yolo.py

│ ├── evaluation/

│ ├── inference/

│ ├── dataset_vizualization.py

│ └── main.py # Central script for all tasks

---

## 🧩 Supported Tasks

| Task        | Description |
|-------------|-------------|
| `split`     | Splits each double-page image into left and right halves. |
| `crop`      | Extracts annotated regions from images based on LabelMe JSON files. |
| `visualize` | Draws YOLO bounding boxes on images using YOLO `.txt` labels and `dataset.yaml`. |
| `train`     | Trains a YOLO model using Ultralytics on any compatible dataset. |

---

## ⚙️ Configuration

Edit `src/main.py` to define which task and dataset to run:

```python
selected_dataset = "words_symbols"   # Choose: "regions", "words_symbols", "classification_subparts"
task = "train"                       # Choose: "split", "crop", "visualize", "train"
