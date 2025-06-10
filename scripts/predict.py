from ultralytics import YOLO
import sys

def predict(img_path, model_path="runs/detect/train/weights/best.pt"):
    model = YOLO(model_path)
    results = model(img_path, save=True, imgsz=640, conf=0.3)

if __name__ == "__main__":
    predict("test.jpg")
