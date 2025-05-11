# Nomenclator Cipher Key Processor

## Overview
The **Nomenclator Cipher Key Processor** is a Python-based GUI application using Tkinter to process nomenclator cipher key images. It consolidates all functionality into a single `gui.py` file, offering preprocessing, detection, and classification tools.

## Features
- **Image Preprocessing**:
  - Binarization
  - Gaussian Blur
  - Contrast Adjustment
  - Sharpening
- **Processing Steps**:
  - Segment Page
  - Detect Words Symbols
  - All Objects Detection
  - Detect Table Structure
  - Map Plaintext to Ciphertext
  - HTR (Handwriting Recognition)
  - Classification
- **Interactive GUI**:
  - Zoom in/out and pan
  - Draw/remove bounding boxes
  - Navigate images (Previous/Next)
  - Grid view for image selection
- **Model Support**:
  - Pre-trained models (e.g., ResNet50, AlexNet)
  - Custom model option
- **Reset Functionality**:
  - Clear all states and return to grid view

## Prerequisites
- **Python**: 3.7+
- **Packages**:
  - `tkinter` (included with Python)
  - `Pillow`
  - `opencv-python`
  - `numpy`

## Installation
### Steps
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/nomenclator-cipher-key-processor.git
   cd nomenclator-cipher-key-processor
   ```
2. Set up a virtual environment (optional):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install Pillow opencv-python numpy
   ```
4. Ensure only `gui.py` exists (remove old multi-file remnants).

## Usage
### How to Run
```bash
python gui.py
```

### GUI Actions
- Click **"Select Images/Folder"** to load images.
- Check preprocessing options (e.g., Binarization).
- Use buttons like **"Segment Page"** for processing steps.
- Select a model and click **"Run Model"** to classify.
- Navigate with **"Previous"**, **"Next"**, or **"Back to Grid"**.
- Click **"Reset"** to clear and return to grid view.

## File Structure
- `gui.py`: Main script with all GUI and processing logic.

## Contributing
1. Fork the repo.
2. Create a branch:
   ```bash
   git checkout -b my-feature
   ```
3. Commit changes:
   ```bash
   git commit -m "Add my feature"
   ```
4. Push branch:
   ```bash
   git push origin my-feature
   ```
5. Open a pull request.

## License
- MIT License (see [LICENSE](LICENSE) file).

## Contact
- Report issues: [GitHub Issues](https://github.com/yourusername/nomenclator-cipher-key-processor/issues)

## Acknowledgments
- Thanks to Tkinter, OpenCV, Pillow, and NumPy communities.
- Inspired by nomenclator cipher key analysis needs.