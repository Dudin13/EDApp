import csv
import sys
import os
from datetime import datetime

def extract_map50(results_path, model_tag):
    if not os.path.exists(results_path):
        print(f"[ERROR] No se encontró el archivo de resultados: {results_path}")
        return

    try:
        with open(results_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Limpiar los nombres de las columnas (a veces tienen espacios extras)
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
            
            rows = list(reader)
            if not rows:
                print(f"[ERROR] El archivo {results_path} está vacío.")
                return

            last_row = rows[-1]
            # YOLOv8 suele usar 'metrics/mAP50(B)' o similar
            map50 = last_row.get('metrics/mAP50(B)')
            
            if map50 is None:
                # Intento alternativo por si el nombre cambia ligeramente
                for key in last_row.keys():
                    if 'mAP50(B)' in key:
                        map50 = last_row[key]
                        break

            if map50:
                history_path = "training_history.txt"
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_entry = f"[{timestamp}] Modelo: {model_tag} | mAP50: {map50}\n"
                
                with open(history_path, mode='a', encoding='utf-8') as hf:
                    hf.write(log_entry)
                
                print(f"[SUCCESS] mAP50 ({map50}) guardado en {history_path}")
            else:
                print(f"[ERROR] No se encontró la columna mAP50(B) en {results_path}")
                print(f"Columnas disponibles: {list(last_row.keys())}")

    except Exception as e:
        print(f"[ERROR] Fallo al extraer métricas: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python extract_metrics.py <ruta_csv> <tag_modelo>")
    else:
        extract_map50(sys.argv[1], sys.argv[2])
