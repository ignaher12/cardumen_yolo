import sqlite3
from pathlib import Path
import re # Using re for slightly more robust parsing of values

def parse_value(value_str):
    """
    Cleans and parses a value string, removing units and converting to float/int.
    Returns None if parsing fails.
    """
    if value_str is None:
        return None
    
    original_value = value_str # Keep original for potential error messages
    
    # Handle potential 'N/A' or empty strings explicitly
    if value_str.strip().lower() in ['n/a', '']:
        return None
        
    # Remove known units and symbols, be careful with order if units contain others (e.g., 'MB')
    value_str = value_str.replace('MB', '').replace('ms', '').replace('s', '').replace('%', '').strip()
    
    try:
        # Try converting to float first, as it handles integers too
        return float(value_str)
    except ValueError:
        # If float fails, maybe it was intended as an int? (less likely here)
        # try:
        #     return int(value_str)
        # except ValueError:
             print(f"  Advertencia: No se pudo convertir el valor '{original_value}' a número.")
             return None # Return None if conversion fails

def create_database_and_table(db_path):
    """Creates the SQLite database and the results table if they don't exist."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT,
                    device TEXT,
                    gpu_monitoring_active INTEGER, -- 0 for False, 1 for True
                    video_filename TEXT,
                    total_frames INTEGER,
                    estimated_duration_s REAL,
                    wall_clock_time_s REAL,
                    process_cpu_time_s REAL,
                    cpu_usage_process_percent REAL,
                    cpu_usage_system_percent REAL,
                    avg_gpu_usage_percent REAL,
                    max_gpu_usage_percent REAL,
                    avg_gpu_mem_usage_mb REAL,
                    max_gpu_mem_usage_mb REAL,
                    total_gpu_mem_mb REAL,
                    max_gpu_mem_percent REAL,
                    avg_confidence_people REAL,
                    detections_count_people INTEGER,
                    avg_inference_time_ms REAL,
                    avg_processing_time_ms REAL,
                    estimated_fps REAL,
                    source_file TEXT -- Store the name of the txt file processed
                )
            ''')
            conn.commit()
            print(f"Base de datos '{db_path}' asegurada y tabla 'results' lista.")
    except sqlite3.Error as e:
        print(f"Error al crear/conectar a la base de datos: {e}")
        raise # Reraise the exception to stop the script

def parse_results_file(txt_file_path, db_path):
    """Reads the text file, parses data, and inserts into the SQLite database."""
    if not txt_file_path.is_file():
        print(f"Error: El archivo de entrada no se encontró en '{txt_file_path}'")
        return

    # --- Column mapping: Text file key -> Database column name ---
    # Handles slight variations and makes code cleaner
    key_to_db_column = {
        "Video": "video_filename",
        "Frames totales": "total_frames",
        "Duración estimada (s)": "estimated_duration_s",
        "Tiempo real transcurrido (Wall-Clock Time) (s)": "wall_clock_time_s",
        "Tiempo de CPU del proceso (s)": "process_cpu_time_s",
        "Uso de CPU del proceso (relativo a 1 core) (%)": "cpu_usage_process_percent",
        "Uso de CPU normalizado sistema (%)": "cpu_usage_system_percent",
        "Uso promedio de GPU (%)": "avg_gpu_usage_percent",
        "Uso máximo de GPU (%)": "max_gpu_usage_percent",
        "Uso promedio de Memoria GPU (MB)": "avg_gpu_mem_usage_mb",
        "Uso máximo de Memoria GPU (MB)": "max_gpu_mem_usage_mb",
        "Memoria GPU Total (MB)": "total_gpu_mem_mb",
        "Uso máximo de Memoria GPU (% del total)": "max_gpu_mem_percent",
        "Confianza promedio (solo de personas)": "avg_confidence_people",
        "Cantidad detecciones (personas)": "detections_count_people",
        "Tiempo medio de inferencia/frame (ms)": "avg_inference_time_ms",
        "Tiempo medio de procesamiento/frame (ms) (pre+inf+post)": "avg_processing_time_ms",
        "Frames procesados por segundo (FPS) estimado": "estimated_fps",
    }

    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # --- Parse Metadata ---
        metadata = {}
        if len(lines) >= 3:
            try:
                metadata['model_name'] = lines[0].split(':', 1)[1].strip()
                # Add device parsing if it exists in the file header
                # Assuming the second line *might* be device
                metadata['device'] = lines[1].split(':', 1)[1].strip()
                metadata['gpu_monitoring_active'] = 1 if lines[2].split(':', 1)[1].strip().lower() == 'true' else 0

                # --- Ensure all required metadata keys exist ---
                if 'device' not in metadata: metadata['device'] = None # Set default if parsing failed
                if 'gpu_monitoring_active' not in metadata: metadata['gpu_monitoring_active'] = None

            except IndexError:
                 print("Error: Formato de metadata inesperado en las primeras líneas.")
                 return
            
            print("Metadata extraída:")
            print(f"  Modelo: {metadata.get('model_name')}")
            print(f"  Dispositivo: {metadata.get('device')}")
            print(f"  Monitoreo GPU Activo: {metadata.get('gpu_monitoring_active')}")
            
            # Remove metadata lines for block processing
            # Adjust number based on actual header lines (2 or 3)
            num_header_lines = 3
            content_lines = lines[num_header_lines:]
            
        else:
            print("Error: El archivo no contiene suficientes líneas para la metadata.")
            return
            
        # --- Combine lines and split into blocks ---
        full_content = "".join(content_lines)
        # Split by the separator line (handle potential variations in '=' count)
        blocks = re.split(r'^=+\s*$', full_content, flags=re.MULTILINE)

        # --- Process each block ---
        records_to_insert = []
        print(f"\nProcesando {len(blocks)} bloques de resultados...")

        for block in blocks:
            block = block.strip() # Remove leading/trailing whitespace
            if not block: # Skip empty blocks (e.g., after the last separator)
                continue

            record_data = {}
            lines_in_block = block.splitlines()

            for line in lines_in_block:
                line = line.strip()
                if not line or line.startswith('---'): # Skip empty lines and separators
                    continue

                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value_str = parts[1].strip()

                    # Map key to DB column name
                    db_column = key_to_db_column.get(key)
                    if db_column:
                        if db_column == "video_filename":
                             record_data[db_column] = value_str # Store filename as text
                        else:
                             record_data[db_column] = parse_value(value_str)
                    else:
                        print(f"  Advertencia: Clave no reconocida en el bloque: '{key}'")
                else:
                     print(f"  Advertencia: Línea con formato inesperado en bloque: '{line}'")


            # --- Combine with metadata and add to list for insertion ---
            if "video_filename" in record_data: # Only add if we have the filename key
                 # Start with metadata, create a copy
                full_record = metadata.copy()
                # Update/add parsed data from the block
                full_record.update(record_data)
                # Add the source file name
                full_record['source_file'] = txt_file_path.name 
                records_to_insert.append(full_record)
            else:
                print("Advertencia: Bloque omitido por faltar 'Video:'")


        # --- Insert records into database ---
        if not records_to_insert:
            print("No se encontraron registros válidos para insertar.")
            return
            
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Prepare column names for the INSERT statement dynamically
                # Get columns from the first record (assuming all records have the same keys derived from parsing)
                # Ensure order matches the table definition if not specifying columns explicitly
                
                # It's safer to specify columns explicitly in INSERT
                db_columns = [
                    "model_name", "device", "gpu_monitoring_active", "video_filename", 
                    "total_frames", "estimated_duration_s", "wall_clock_time_s", 
                    "process_cpu_time_s", "cpu_usage_process_percent", "cpu_usage_system_percent",
                    "avg_gpu_usage_percent", "max_gpu_usage_percent", "avg_gpu_mem_usage_mb",
                    "max_gpu_mem_usage_mb", "total_gpu_mem_mb", "max_gpu_mem_percent",
                    "avg_confidence_people", "detections_count_people", "avg_inference_time_ms",
                    "avg_processing_time_ms", "estimated_fps", "source_file"
                ]
                
                placeholders = ', '.join(['?'] * len(db_columns))
                sql_insert = f"INSERT INTO results ({', '.join(db_columns)}) VALUES ({placeholders})"

                # Prepare data tuples, ensuring None for missing keys
                data_tuples = []
                for record in records_to_insert:
                    data_tuples.append(tuple(record.get(col) for col in db_columns))

                cursor.executemany(sql_insert, data_tuples)
                conn.commit()
                print(f"\n¡Éxito! {len(records_to_insert)} registros insertados en '{db_path}'.")

        except sqlite3.Error as e:
            print(f"Error al insertar datos en la base de datos: {e}")

    except FileNotFoundError:
        print(f"Error: El archivo de entrada no se encontró en '{txt_file_path}'")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

def parse_results_files(results_dir_paths:list[Path], db_path:Path) -> None:
    for dir in results_dir_paths:
        for file in dir.iterdir():
            if file.is_file():
                parse_results_file(file, db_path)

# --- Main execution ---
if __name__ == "__main__":
    # --- Configuration ---
    DEFAULT_RESULTS_PATH = Path("resultados")
    DEFAULT_DB_FILE = "metrics_database.db"
    DEVICES = ['cpu']
    
    

    results_paths = [DEFAULT_RESULTS_PATH / device for device in DEVICES]
    output_db_path = Path(DEFAULT_DB_FILE)
    # --------------------
    print(f"Base de datos de salida: {output_db_path}")

    create_database_and_table(output_db_path)
    parse_results_files(results_paths, output_db_path)