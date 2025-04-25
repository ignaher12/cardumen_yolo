import os
import time
import psutil
from ultralytics import YOLO
import cv2
from datetime import datetime

videos_dir = "./videos"
resultados_dir = "./resultados"

# Ensure output directory exists
os.makedirs(resultados_dir, exist_ok=True)

# Add timestamp to output file name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(resultados_dir, f"metrics_results_{timestamp}.txt")

model = YOLO('yolo11n.pt')

video_files = [f for f in os.listdir(videos_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

with open(output_file, "w") as out_f:
    for video_file in video_files:
        video_path = os.path.join(videos_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        confianza_promedio = 0
        cantidad_detecciones_personas = 0
        tiempo_inferencia_total = 0
        tiempo_procesamiento_total = 0

        # --- Empezar temporizador ---
        start_cpu_time = time.process_time()
        start_wall_time = time.time()

        current_process = psutil.Process(os.getpid())
        current_process.cpu_percent(interval=None)

        resultados = model(video_path, stream=True, conf=0.5, verbose=False, save=True, project='resultados', exist_ok=True)

        frame_idx = 0

        for frame in resultados:
            frame_idx += 1
            print(f"Procesando {video_file} frame {frame_idx}/{total_frames}", end='\r')

            speed = frame.speed
            tiempo_inferencia_total += speed['inference']
            tiempo_procesamiento_total += speed['preprocess'] + speed['inference'] + speed['postprocess']

            for box in frame.boxes:
                clase = int(box.cls[0])
                confianza = float(box.conf[0])
                if clase == 0:
                    confianza_promedio += confianza
                    cantidad_detecciones_personas += 1
        print()

        # --- Terminar temporizador ---
        end_cpu_time = time.process_time()
        end_wall_time = time.time()

        cpu_usage = current_process.cpu_percent(interval=None)
        cpu_time_used = end_cpu_time - start_cpu_time
        wall_time_elapsed = end_wall_time - start_wall_time

        if cantidad_detecciones_personas > 0:
            confianza_promedio /= cantidad_detecciones_personas

        # Write metrics to file
        out_f.write(f"Video: {video_file}\n")
        out_f.write(f"Uso de CPU (%): {cpu_usage/psutil.cpu_count():.4f}%\n")
        out_f.write(f"CPU Time Used (s): {cpu_time_used:.4f} seconds\n")
        out_f.write(f"Wall-Clock Time Elapsed (s): {wall_time_elapsed:.4f} seconds\n")
        out_f.write(f"Confianza promedio (solo de personas) (%): {confianza_promedio:.4f}\n")
        out_f.write(f"Tiempo medio de inferencia/frame (ms): {tiempo_inferencia_total/max(total_frames,1):.4f}\n")
        out_f.write(f"Tiempo medio de procesamiento/frame (ms) (preprocesamiento + inferencia + postprocesamiento): {tiempo_procesamiento_total/max(total_frames,1):.4f}\n")
        out_f.write("-" * 40 + "\n")

print(f"Todos los resultados guardados en {output_file}")