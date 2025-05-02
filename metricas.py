import os
from pathlib import Path
import time
import psutil
from ultralytics import YOLO
import cv2
from datetime import datetime
import pynvml

# Use uppercase for constants as a convention
BASE_DIR = Path(".") # Or specify an absolute base if needed: Path(__file__).parent
VIDEOS_DIR = BASE_DIR / "videos"
RESULTADOS_DIR = BASE_DIR / "resultados"

# Asegurarse de que el directorio de salida exista
RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

# Añadir timestamp al nombre del archivo de salida
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file_path = RESULTADOS_DIR / f"metrics_results_{timestamp}.txt"

# --- Inicialización de NVML para monitoreo de GPU ---
gpu_monitoring_available = False
gpu_handle = None
try:
    pynvml.nvmlInit()
    # Asumimos que se usará la GPU 0 por defecto con 'cuda'
    # Puedes ajustar el índice si usas una GPU específica diferente
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_monitoring_available = True
    print("NVML inicializado correctamente. Monitoreo de GPU activado.")
except pynvml.NVMLError as error:
    print(f"Error al inicializar NVML: {error}")
    print("El monitoreo de métricas de GPU no estará disponible.")
# ----------------------------------------------------

modelos_yolo = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt']

# --- Carga del modelo y movimiento a GPU ---
modelo_seleccionado = 'yolo11n.pt'
print(f"Cargando modelo: {modelo_seleccionado}")
model = YOLO(modelo_seleccionado)
model.to('cuda') # Mover el modelo a la GPU
print(f"Modelo {modelo_seleccionado} movido a CUDA.")
# -----------------------------------------

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
    out_f.write(f"Monitoreo de GPU activo: {gpu_monitoring_available}\n")
    out_f.write("=" * 40 + "\n")

    for video_path in video_files:
        video_path_str = str(video_path)
        cap = cv2.VideoCapture(video_path_str)

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

        # Variables para métricas de GPU (por video)
        gpu_util_samples = []
        gpu_mem_used_samples = []

        # --- Empezar temporizadores y monitoreo CPU ---
        start_cpu_time = time.process_time()
        start_wall_time = time.time()
        current_process = psutil.Process(os.getpid())
        # Llamada inicial para establecer punto de referencia para cpu_percent
        current_process.cpu_percent(interval=None)
        # ---

        # Realizar inferencia
        # Nota: stream=True es más eficiente en memoria pero dificulta medir
        # el uso pico *exacto* de GPU antes/después. Mediremos durante el bucle.
        resultados = model(video_path_str, 
                           stream=True, 
                           conf=0.5, 
                           verbose=False, 
                           save=True, 
                           project=RESULTADOS_DIR, # Use Path object for project dir
                           exist_ok=True, 
                           device='cuda')
        frame_idx = 0
        last_gpu_check_time = time.time()

        for frame in resultados:
            frame_idx += 1
            print(f"Procesando {video_path.name} frame {frame_idx}/{total_frames}", end='\r')

            speed = frame.speed # {'preprocess': ..., 'inference': ..., 'postprocess': ...} en ms
            tiempo_inferencia_total += speed['inference']
            tiempo_procesamiento_total += speed['preprocess'] + speed['inference'] + speed['postprocess']

            for box in frame.boxes:
                clase = int(box.cls[0])
                confianza = float(box.conf[0])
                # Clase 0 suele ser 'person' en COCO dataset que usa YOLO por defecto
                if clase == 0:
                    confianza_promedio += confianza
                    cantidad_detecciones_personas += 1

            # --- Capturar métricas de GPU periódicamente ---
            current_time = time.time()
            if gpu_monitoring_available and (current_time - last_gpu_check_time >= 1.0):
                try:
                    util_rates = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                    gpu_util_samples.append(util_rates.gpu) # % de uso del core de GPU
                    gpu_mem_used_samples.append(mem_info.used) # Memoria usada en Bytes
                    last_gpu_check_time = current_time
                except pynvml.NVMLError as error:
                    # Podría fallar si la GPU se resetea o algo inusual ocurre
                    print(f"\nAdvertencia: Error al leer métricas de GPU: {error}")
                    gpu_monitoring_available = False # Desactivar monitoreo para este video
            # --------------------------------------------------------------------

        print() # Nueva línea después del progreso del video

        # --- Terminar temporizadores y calcular métricas CPU ---
        # Capturar uso de CPU DESPUÉS del trabajo principal
        cpu_usage_percent = current_process.cpu_percent(interval=None)
        end_cpu_time = time.process_time()
        end_wall_time = time.time()
        # ---

        cpu_time_used = end_cpu_time - start_cpu_time
        wall_time_elapsed = end_wall_time - start_wall_time

        # --- Calcular métricas promedio/max de GPU ---
        avg_gpu_util = 0
        max_gpu_util = 0
        avg_gpu_mem_used_mb = 0
        max_gpu_mem_used_mb = 0
        total_gpu_mem_mb = 0

        if gpu_monitoring_available and gpu_util_samples:
            avg_gpu_util = sum(gpu_util_samples) / len(gpu_util_samples)
            max_gpu_util = max(gpu_util_samples)
        if gpu_monitoring_available and gpu_mem_used_samples:
            avg_gpu_mem_used_bytes = sum(gpu_mem_used_samples) / len(gpu_mem_used_samples)
            max_gpu_mem_used_bytes = max(gpu_mem_used_samples)
            avg_gpu_mem_used_mb = avg_gpu_mem_used_bytes / (1024**2)
            max_gpu_mem_used_mb = max_gpu_mem_used_bytes / (1024**2)
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                total_gpu_mem_mb = mem_info.total / (1024**2)
            except pynvml.NVMLError as error:
                 print(f"\nAdvertencia: Error al leer memoria total de GPU: {error}")

        # -------------------------------------------

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
        if gpu_monitoring_available:
             out_f.write(f"Uso promedio de GPU (%): {avg_gpu_util:.2f}%\n")
             out_f.write(f"Uso máximo de GPU (%): {max_gpu_util:.2f}%\n")
             out_f.write(f"Uso promedio de Memoria GPU (MB): {avg_gpu_mem_used_mb:.2f} MB\n")
             out_f.write(f"Uso máximo de Memoria GPU (MB): {max_gpu_mem_used_mb:.2f} MB\n")
             if total_gpu_mem_mb > 0:
                 out_f.write(f"Memoria GPU Total (MB): {total_gpu_mem_mb:.2f} MB\n")
                 out_f.write(f"Uso máximo de Memoria GPU (% del total): { (max_gpu_mem_used_mb / total_gpu_mem_mb * 100) if total_gpu_mem_mb else 0 :.2f}%\n")
        else:
             out_f.write("Uso de GPU: No disponible\n")
             out_f.write("Uso de Memoria GPU: No disponible\n")
        out_f.write("-" * 10 + " Métricas YOLO " + "-" * 10 + "\n")
        out_f.write(f"Confianza promedio (solo de personas): {confianza_promedio:.4f}\n")
        out_f.write(f"Cantidad detecciones (personas): {cantidad_detecciones_personas}\n")
        # Calcular tiempos promedio por frame procesado
        frames_procesados = max(frame_idx, 1) # Usar frame_idx por si acaso total_frames era incorrecto
        out_f.write(f"Tiempo medio de inferencia/frame (ms): {tiempo_inferencia_total / frames_procesados:.4f}\n")
        out_f.write(f"Tiempo medio de procesamiento/frame (ms) (pre+inf+post): {tiempo_procesamiento_total / frames_procesados:.4f}\n")
        out_f.write(f"Frames procesados por segundo (FPS) estimado: {frames_procesados / wall_time_elapsed if wall_time_elapsed > 0 else 0:.2f}\n")
        out_f.write("=" * 40 + "\n")

print(f"\nTodos los resultados guardados en {output_file_path}")

# --- Limpieza de NVML ---
if gpu_monitoring_available:
    try:
        pynvml.nvmlShutdown()
        print("NVML apagado correctamente.")
    except pynvml.NVMLError as error:
        print(f"Error al apagar NVML: {error}")
# ------------------------