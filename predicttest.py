from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Define path to the image file
source = "http://192.168.0.7:4747/video"

# Run inference on the source
model.predict(source, imgsz=320, conf=0.2, show=True)
