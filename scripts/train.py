from ultralytics import YOLO

def train():
     model = YOLO("yolov8n.yaml")

     model.train(
        data="dataset-counter-products/data.yaml",
        epochs=180,               
        imgsz=640,                
        batch=4,                
        device=0,              
        workers=2,                
        cache='disk',
        val=True             
    )


if __name__ == "__main__":
    train()
