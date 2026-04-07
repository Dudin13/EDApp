import os
import shutil
import subprocess
import sys
from pathlib import Path

# Configuración de Rutas
BASE_DIR = Path("C:/apped")
SCRIPTS_DIR = BASE_DIR / "scripts"
DATA_DIR = BASE_DIR / "data/datasets/veo_frames_raw_final"
MODEL_PATH = BASE_DIR / "models/players.pt"
BACKUP_PATH = BASE_DIR / "models/players_backup.pt"
DATA_YAML = DATA_DIR / "data.yaml"

def run_preparation():
    print("[1/4] Paso 1: Ejecutando preparacion del dataset...")
    prep_script = SCRIPTS_DIR / "prepare_training_dataset.py"
    try:
        # Usamos sys.executable si estamos en el venv, o simplemente python
        result = subprocess.run([sys.executable, str(prep_script)], check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error al preparar el dataset:\n{e.stderr}")
        return False
    return True

def check_images():
    img_dir = DATA_DIR / "images"
    if not img_dir.exists():
        return 0
    return len(list(img_dir.glob("*")))

def create_data_yaml():
    yaml_content = f"""
path: {DATA_DIR}
train: images
val: images  # Usamos las mismas para validación rápida en incremental

nc: 4
names:
  0: player
  1: goalkeeper
  2: referee
  3: ball
"""
    DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
    DATA_YAML.write_text(yaml_content.strip(), encoding="utf-8")
    print(f"[INFO] Archivo data.yaml generado en {DATA_YAML}")

def train():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 'ultralytics' no esta instalado. Asegurate de usar el entorno virtual correcto.")
        return None

    print("\n[3/4] Paso 3: Iniciando entrenamiento YOLO (Fine-tuning)...")
    model = YOLO(str(MODEL_PATH))
    
    results = model.train(
        data=str(DATA_YAML),
        epochs=20,
        imgsz=1280,
        device="cuda",
        name="incremental_train_run",
        exist_ok=True
    )
    
    return results

def finalize(results):
    print("\n[4/4] Paso 4: Finalizando y guardando modelo...")
    
    # Localizar el mejor modelo
    # YOLO guarda en runs/detect/incremental_train_run/weights/best.pt por defecto con name=...
    best_pt = BASE_DIR / "runs/detect/incremental_train_run/weights/best.pt"
    
    if not best_pt.exists():
        print(f"[WARN] No se encontro best.pt en {best_pt}. Buscando en el ultimo directorio de runs...")
        # fallback si la estructura de runs es diferente
        runs_dir = BASE_DIR / "runs/detect"
        all_runs = sorted(runs_dir.glob("incremental_train_run*"), key=os.path.getmtime, reverse=True)
        if all_runs:
            best_pt = all_runs[0] / "weights/best.pt"

    if best_pt.exists():
        # Backup del anterior
        if MODEL_PATH.exists():
            print(f"[FILE] Creando backup: {MODEL_PATH.name} -> {BACKUP_PATH.name}")
            shutil.copy2(MODEL_PATH, BACKUP_PATH)
        
        # Actualizar el principal
        print(f"[FILE] Actualizando modelo principal: {best_pt} -> {MODEL_PATH}")
        shutil.copy2(best_pt, MODEL_PATH)
        
        # Mostrar métricas
        print("\n" + "="*40)
        print("METRICAS FINALES (mAP50)")
        print("="*40)
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            map50 = metrics.get('metrics/mAP50(B)', 'N/A')
            print(f"mAP50: {map50}")
        else:
            print("Consulte los resultados detallados en la carpeta 'runs/'.")
        print("="*40)
    else:
        print("[ERROR] No se pudo localizar el archivo weights/best.pt generado.")

def main():
    print("-" * 50)
    print("      EDApp - ENTRENAMIENTO INCREMENTAL")
    print("-" * 50 + "\n")

    if not run_preparation():
        return

    count = check_images()
    print(f"[INFO] Imagenes revisadas disponibles: {count}")

    if count < 10:
        print("\n[WARN] AVISO: Se necesitan al menos 10 imagenes revisadas para entrenar.")
        print("[INFO] Accion sugerida: Revisa mas imagenes en el Labeller App.")
        return

    # Si hay 10 o más
    create_data_yaml()
    results = train()
    
    if results:
        finalize(results)
        print("\n[OK] Proceso completado con exito.")
    else:
        print("\n[FAIL] El entrenamiento fallo o fue cancelado.")

if __name__ == "__main__":
    main()
