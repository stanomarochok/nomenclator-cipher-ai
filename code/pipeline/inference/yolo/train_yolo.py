from ultralytics import YOLO


def train_yolo():
  model = YOLO("yolo11n.pt") # pass any model type

  train_results = model.train(
    data="../../materials/dataset/word_symbol_annotations_yolo/yolo_dataset_2/dataset.yaml", # path to dataset YAML
    epochs=50, # number of training epochs
    imgsz=960, # training image size
    device="0", # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    save=True,
    save_period=1
  )


if __name__ == "__main__":
  train_yolo()
