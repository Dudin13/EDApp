import os
import zipfile
import requests
from tqdm import tqdm
from pathlib import Path

# Configuración
ZENODO_URL = "https://zenodo.org/api/records/7808511/files/YOLO.zip/content"
BASE_DIR = Path("c:/apped/04_Datasets_Entrenamiento/soccernet_h250")
ZIP_PATH = BASE_DIR / "soccernet_yolo.zip"
EXTRACT_DIR = BASE_DIR / "extracted"

# Mapeo de Clases:
# SoccerNet: 0 -> ball, 1 -> person
# Nuestro data.yaml: 0: goalkeeper, 1: player, 2: ball, 3: referee
CLASS_MAP = {
    0: 2,  # ball -> ball
    1: 1   # person -> player
}

def download_file(url, target_path):
    print(f"Descargando dataset desde Zenodo (~2.5 GB)...")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(target_path, 'wb') as f, tqdm(
        desc=target_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def extract_zip(zip_path, extract_to):
    print(f"Extrayendo {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def remap_labels(labels_dir):
    print(f"Remapeando etiquetas en {labels_dir}...")
    label_files = list(labels_dir.glob("*.txt"))
    for label_file in tqdm(label_files, desc="Remapeando"):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.split()
            if not parts: continue
            cls_id = int(parts[0])
            if cls_id in CLASS_MAP:
                parts[0] = str(CLASS_MAP[cls_id])
                new_lines.append(" ".join(parts) + "\n")
        
        with open(label_file, 'w') as f:
            f.writelines(new_lines)

def main():
    # Eliminar si existe un archivo corrupto (pequeño/HTML)
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size < 1000000: # < 1MB
        print(f"Eliminando archivo corrupto previo ({ZIP_PATH.stat().st_size} bytes)...")
        ZIP_PATH.unlink()

    if not ZIP_PATH.exists():
        try:
            download_file(ZENODO_URL, ZIP_PATH)
        except Exception as e:
            print(f"Error descargando: {e}")
            return

    if not EXTRACT_DIR.exists():
        extract_zip(ZIP_PATH, EXTRACT_DIR)

    # El zip suele contener carpetas 'train' y 'test' o similar
    # Vamos a buscar todas las subcarpetas de etiquetas.
    for folder in EXTRACT_DIR.rglob("labels"):
        if folder.is_dir():
            remap_labels(folder)

    print("\n✅ Integración de SoccerNet completada.")
    print(f"Datos listos en: {EXTRACT_DIR}")
    print("Siguiente paso: Actualizar train_yolo/data.yaml para incluir estas rutas.")

if __name__ == "__main__":
    main()
