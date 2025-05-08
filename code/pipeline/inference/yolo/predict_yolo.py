from ultralytics import YOLO

# Load the saved model
model = YOLO("runs/detect/train/weights/best.pt")

# source= expects a list of path/to/image.png
sources = ["../../materials/dataset/images/img132.png", "../../materials/dataset/images/img128.png"]

# Perform inference
results = model.predict(source=sources, save=True, line_width=5)