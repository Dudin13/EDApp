"""
add_ball_dataset.py
===================
Descarga un dataset de balon de Roboflow y lo fusiona con el hybrid_dataset.

La API key se lee de la variable de entorno ROBOFLOW_API_KEY.
Configurarla en PowerShell (solo una vez):
    [System.Environment]::SetEnvironmentVariable("ROBOFLOW_API_KEY", "tu_key", "User")

Uso:
    python ml/training/add_ball_dataset.py
    python ml/training/add_ball_dataset.py --dry-run
"""

import os
import shutil
import argparse
import random
from pathlib import Path

BASE         = Path(os.environ.get("APPED_ROOT", "C:/apped"))
DATASET_ROOT = BASE / "data" / "datasets" / "hybrid_dataset"
DOWNLOAD_DIR = BASE / "data" / "datasets" / "ball_dataset_raw"

# Clase ball en el nuevo esquema
BALL_CLASS_ID = 3

# Datasets de balon en Roboflow (en orden de preferencia)
# Se intentan en orden hasta que uno funcione
ROBOFLOW_SOURCES = [
    {"workspace": "roboflow-jvuqo", "project": "football-ball-detection-rejhg", "version": 2},
    {"workspace": "roboflow-jvuqo", "project": "ball-detection-i8ohe",          "version": 3},
    {"workspace": "sports-wtpzq",   "project": "soccer-ball-detection",         "version": 1},
]


def get_api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        print("\nERROR: Variable de entorno ROBOFLOW_API_KEY no configurada.")
        print("Configurala en PowerShell:")
        print('  [System.Environment]::SetEnvironmentVariable("ROBOFLOW_API_KEY", "tu_key", "User")')
        print("Luego cierra y vuelve a abrir PowerShell.")
        raise SystemExit(1)
    return key


def download_ball_dataset(api_key: str) -> Path:
    """Descarga el dataset de balon desde Roboflow. Devuelve la ruta al dataset descargado."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Instalando roboflow...")
        os.system("pip install roboflow --quiet")
        from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    last_error = None

    for source in ROBOFLOW_SOURCES:
        try:
            print(f"\n  Intentando: {source['workspace']}/{source['project']} v{source['version']}...")
            project = rf.workspace(source["workspace"]).project(source["project"])
            dataset = project.version(source["version"]).download(
                "yolov8",
                location=str(DOWNLOAD_DIR),
                overwrite=True
            )
            print(f"  Descargado en: {DOWNLOAD_DIR}")
            return DOWNLOAD_DIR
        except Exception as e:
            last_error = e
            print(f"  No disponible: {e}")
            continue

    print(f"\nNinguna fuente de Roboflow funciono. Ultimo error: {last_error}")
    print("Intentando fuente alternativa sin API key...")
    return download_ball_fallback()


def download_ball_fallback() -> Path:
    """Descarga dataset de balon desde GitHub (sin API key)."""
    import urllib.request
    import zipfile

    url  = "https://github.com/niconielsen32/ComputerVision/raw/master/datasets/ball_dataset.zip"
    dest = BASE / "data" / "datasets" / "ball_fallback.zip"

    print(f"  Descargando desde GitHub...")
    try:
        urllib.request.urlretrieve(url, dest)
        with zipfile.ZipFile(dest, "r") as z:
            z.extractall(DOWNLOAD_DIR)
        dest.unlink()
        print(f"  Descargado en: {DOWNLOAD_DIR}")
        return DOWNLOAD_DIR
    except Exception as e:
        print(f"  Error en fallback: {e}")
        print("\n  Descarga manual:")
        print("  1. Ve a https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg")
        print("  2. Descarga en formato YOLOv8")
        print(f"  3. Extrae en: {DOWNLOAD_DIR}")
        raise SystemExit(1)


def find_ball_images_and_labels(raw_dir: Path):
    """
    Busca imagenes y etiquetas de balon en el dataset descargado.
    Roboflow puede tener estructura train/valid/test o directamente images/.
    """
    results = []
    for split_name in ("train", "valid", "test", ""):
        img_dir = raw_dir / split_name / "images" if split_name else raw_dir / "images"
        lab_dir = raw_dir / split_name / "labels" if split_name else raw_dir / "labels"
        if not img_dir.exists():
            continue
        for img in img_dir.iterdir():
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lab = lab_dir / (img.stem + ".txt")
            if lab.exists():
                results.append((img, lab))
    return results


def remap_ball_label(label_path: Path, ball_class_id: int) -> list[str]:
    """
    Remapea las etiquetas del dataset de balon a la clase correcta.
    En datasets de Roboflow de 1 sola clase, la clase 0 = ball.
    """
    lines_out = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if not parts or len(parts) < 5:
            continue
        # Cualquier clase en el dataset de balon se mapea a ball_class_id
        parts[0] = str(ball_class_id)
        lines_out.append(" ".join(parts))
    return lines_out


def merge_ball_dataset(raw_dir: Path, dry_run: bool) -> dict:
    """Fusiona el dataset de balon con el hybrid_dataset."""
    pairs = find_ball_images_and_labels(raw_dir)

    if not pairs:
        print(f"  No se encontraron pares imagen+etiqueta en {raw_dir}")
        return {"total": 0, "train": 0, "val": 0}

    print(f"\n  Encontradas {len(pairs)} imagenes de balon")

    # Split 85% train / 15% val
    random.seed(42)
    random.shuffle(pairs)
    n_val   = max(1, int(len(pairs) * 0.15))
    val_pairs   = pairs[:n_val]
    train_pairs = pairs[n_val:]

    splits = {"train": train_pairs, "valid": val_pairs}
    counts = {"train": 0, "val": 0}

    for split_name, split_pairs in splits.items():
        dst_imgs = DATASET_ROOT / split_name / "images"
        dst_labs = DATASET_ROOT / split_name / "labels"

        if not dry_run:
            dst_imgs.mkdir(parents=True, exist_ok=True)
            dst_labs.mkdir(parents=True, exist_ok=True)

        for img_path, lab_path in split_pairs:
            # Nombre unico para no colisionar con imagenes existentes
            new_name = f"ball_{img_path.name}"
            dst_img  = dst_imgs / new_name
            dst_lab  = dst_labs / (img_path.stem + "_ball.txt")

            if not dry_run:
                shutil.copy2(img_path, dst_img)
                remapped = remap_ball_label(lab_path, BALL_CLASS_ID)
                dst_lab.write_text("\n".join(remapped) + "\n", encoding="utf-8")

            key = "train" if split_name == "train" else "val"
            counts[key] += 1

    counts["total"] = counts["train"] + counts["val"]
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anadir dataset de balon al hybrid_dataset")
    parser.add_argument("--dry-run", action="store_true", help="Previsualizar sin modificar")
    parser.add_argument("--skip-download", action="store_true",
                        help=f"Saltar descarga (usar datos ya en {DOWNLOAD_DIR})")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  EDApp - Añadir dataset de balon")
    print("="*55)
    print(f"  Dataset destino: {DATASET_ROOT}")
    print(f"  Modo: {'DRY RUN' if args.dry_run else 'APLICAR CAMBIOS'}\n")

    # Descargar
    if not args.skip_download and not args.dry_run:
        api_key = get_api_key()
        raw_dir = download_ball_dataset(api_key)
    elif DOWNLOAD_DIR.exists():
        raw_dir = DOWNLOAD_DIR
        print(f"  Usando dataset ya descargado: {raw_dir}")
    else:
        print("  ERROR: No hay dataset descargado. Ejecuta sin --skip-download primero.")
        raise SystemExit(1)

    # Fusionar
    print("\n  Fusionando con hybrid_dataset...")
    counts = merge_ball_dataset(raw_dir, args.dry_run)

    print(f"\n  Imagenes de balon añadidas:")
    print(f"    Train: {counts['train']}")
    print(f"    Val:   {counts['val']}")
    print(f"    Total: {counts['total']}")

    if args.dry_run:
        print("\n  Ejecuta sin --dry-run para aplicar los cambios.")
    else:
        print(f"\n  Siguiente paso:")
        print(f"    python ml/training/train_unified.py --target players")

    print("="*55 + "\n")
