import os
import time
import psutil
from ultralytics import YOLO
import cv2
from datetime import datetime
from pathlib import Path

# Use uppercase for constants as a convention
BASE_DIR = Path(".") # Or specify an absolute base if needed: Path(__file__).parent
VIDEOS_DIR = BASE_DIR / "videos"
RESULTADOS_DIR = BASE_DIR / "resultados"
DEVICE = 'cpu'
# Asegurarse de que el directorio de salida exista
RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

modelos_yolo = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt']
resultado_dir = RESULTADOS_DIR / DEVICE
resultado_dir.mkdir(exist_ok=True, parents=True)

for modelo_seleccionado in modelos_yolo:
    # Añadir timestamp al nombre del archivo de salida
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_path = resultado_dir / f"metrics_results_{DEVICE}_{modelo_seleccionado}_{timestamp}.txt"

    model = YOLO(modelo_seleccionado)

    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'} # Use a set for efficient lookup
    video_files = []
    if VIDEOS_DIR.is_dir(): # Check if the videos directory exists
        for item in VIDEOS_DIR.iterdir():
            # Check if it's a file and its suffix (extension) is in our set
            if item.is_file() and item.suffix.lower() in video_extensions:
                video_files.append(item) # Add the Path object to the list
    else:
        print(f"Advertencia: El directorio de videos no existe en {VIDEOS_DIR}")

    with open(output_file_path, "w", encoding='utf-8') as out_f:
        out_f.write(f"Modelo YOLO utilizado: {modelo_seleccionado}\n")
        out_f.write(f"Dispositivo: {DEVICE}\n")
        out_f.write(f"Monitoreo de GPU activo: false\n")
        out_f.write("=" * 40 + "\n")

        for video_path in video_files:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Error al abrir el video: {video_path.name}")
                continue
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            cap.release()

            print(f"\nProcesando video: {video_path.name} ({total_frames} frames @ {fps:.2f} FPS)")

            confianza_promedio = 0
            cantidad_detecciones_personas = 0
            tiempo_inferencia_total = 0
            tiempo_procesamiento_total = 0

            # --- Empezar temporizador ---
            start_cpu_time = time.process_time()
            start_wall_time = time.time()

            current_process = psutil.Process(os.getpid())
            current_process.cpu_percent(interval=None)

            resultados = model(video_path, stream=True, conf=0.5, verbose=True, save=True, project= resultado_dir / modelo_seleccionado, exist_ok=True, device=DEVICE)

            frame_idx = 0

            for frame in resultados:
                frame_idx += 1
                print(f"Procesando {video_path.name} frame {frame_idx}/{total_frames}", end='\r')

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

            cpu_usage_percent = current_process.cpu_percent(interval=None)
            cpu_time_used = end_cpu_time - start_cpu_time
            wall_time_elapsed = end_wall_time - start_wall_time

            if cantidad_detecciones_personas > 0:
                confianza_promedio /= cantidad_detecciones_personas

            # Escribir métricas al archivo
            out_f.write(f"Video: {video_path.name}\n")
            out_f.write(f"Frames totales: {total_frames}\n")
            out_f.write(f"Duración estimada (s): {total_frames / fps if fps > 0 else 0:.2f}\n")
            out_f.write("-" * 10 + " Rendimiento " + "-" * 10 + "\n")
            out_f.write(f"Tiempo real transcurrido (Wall-Clock Time) (s): {wall_time_elapsed:.4f}\n")
            out_f.write(f"Tiempo de CPU del proceso (s): {cpu_time_used:.4f}\n")
            # psutil.cpu_percent() devuelve el % para el proceso desde la última llamada.
            # Dividir por cpu_count() da un % normalizado respecto a todo el sistema,
            # pero cpu_usage_percent ya es el % para *este* proceso respecto a un core.
            # Mostrar ambos puede ser útil. % total del sistema es más complejo de calcular correctamente.
            out_f.write(f"Uso de CPU del proceso (relativo a 1 core) (%): {cpu_usage_percent:.2f}%\n")
            out_f.write(f"Uso de CPU normalizado sistema (%): {cpu_usage_percent / psutil.cpu_count():.2f}%\n") # Opcional
            out_f.write("-" * 10 + " Métricas YOLO " + "-" * 10 + "\n")
            out_f.write(f"Confianza promedio (solo de personas): {confianza_promedio:.4f}\n")
            out_f.write(f"Cantidad detecciones (personas): {cantidad_detecciones_personas}\n")
            # Calcular tiempos promedio por frame procesado
            frames_procesados = max(frame_idx, 1) # Usar frame_idx por si acaso total_frames era incorrecto
            out_f.write(f"Tiempo medio de inferencia/frame (ms): {tiempo_inferencia_total / frames_procesados:.4f}\n")
            out_f.write(f"Tiempo medio de procesamiento/frame (ms) (pre+inf+post): {tiempo_procesamiento_total / frames_procesados:.4f}\n")
            out_f.write(f"Frames procesados por segundo (FPS) estimado: {frames_procesados / wall_time_elapsed if wall_time_elapsed > 0 else 0:.2f}\n")
            out_f.write("=" * 40 + "\n")

    print(f"Todos los resultados guardados en {output_file_path}")