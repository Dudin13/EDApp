"""
download_kaggle_models.py — Descarga los 3 modelos especializados del notebook Kaggle.

Uso:
    python scripts/download_kaggle_models.py

Requiere:
    pip install gdown

Modelos que descarga:
    detect_players.pt  — jugadores/portero/arbitro (384x640)
    detect_ball.pt     — balon a alta resolucion 1280px
    pose_field.pt      — keypoints del campo (homografia automatica)
"""

import os
import sys
from pathlib import Path

try:
    import gdown
except ImportError:
    print("Instalando gdown...")
    os.system(f"{sys.executable} -m pip install gdown -q")
    import gdown

# Ruta destino (misma carpeta que best_football_seg.pt)
APPED_ROOT = Path(os.getenv("APPED_ROOT", "c:/apped"))
MODELS_DIR = APPED_ROOT / "ml" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# IDs de Google Drive del notebook Kaggle (akrambelha / final-models)
# Fuente: /kaggle/input/final-models/
MODELS = {
    "detect_players.pt": "17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q",
    "detect_ball.pt":    "1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V",
    "pose_field.pt":     "1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf",
}


def download_model(name, drive_id, dest_dir):
    dest = dest_dir / name
    if dest.exists():
        print(f"  [OK] {name} ya existe — omitiendo")
        return True
    url = f"https://drive.google.com/uc?id={drive_id}"
    print(f"  Descargando {name}...")
    try:
        gdown.download(url, str(dest), quiet=False)
        if dest.exists():
            size_mb = dest.stat().st_size / 1024 / 1024
            print(f"  [OK] {name} descargado ({size_mb:.1f} MB)")
            return True
        else:
            print(f"  [ERROR] {name} no se descargo correctamente")
            return False
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        return False


def main():
    print(f"\nDescargando modelos Kaggle a: {MODELS_DIR}\n")
    ok = 0
    for name, drive_id in MODELS.items():
        if download_model(name, drive_id, MODELS_DIR):
            ok += 1

    print(f"\n{ok}/{len(MODELS)} modelos listos.")
    if ok == len(MODELS):
        print("\nTodos los modelos disponibles. EDApp usara la arquitectura de 3 modelos.")
        print("  detect_players.pt -> jugadores/portero/arbitro")
        print("  detect_ball.pt    -> balon (1280px, 60% mejor en VEO)")
        print("  pose_field.pt     -> keypoints campo (calibracion automatica)")
    else:
        print("\nAlgunos modelos no se descargaron. Comprueba tu conexion o descargalos manualmente.")
        print("URLs de descarga manual:")
        for name, drive_id in MODELS.items():
            p = MODELS_DIR / name
            if not p.exists():
                print(f"  {name}: https://drive.google.com/uc?id={drive_id}")

    print(f"\nSi ya tienes los modelos en otra carpeta, copialos a:\n  {MODELS_DIR}")


if __name__ == "__main__":
    main()
