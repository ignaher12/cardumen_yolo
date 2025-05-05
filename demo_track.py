from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(".")
VIDEOS_PATH = Path('videos/prueba5_1270x720_13fotogramas.mp4')

model = YOLO('yolo11s.pt')

model.track(source=VIDEOS_PATH, verbose=False, classes=[0], show=True, save=False, tracker="bytetrack.yaml")

print(f"Procesamiento del video {VIDEOS_PATH.name} finalizado.")