import os
import shutil
from pathlib import Path

# Configuración
BASE_DIR = Path("C:/apped")
SRC_DIR = BASE_DIR / "data/datasets/veo_frames_raw"
DST_DIR = BASE_DIR / "data/datasets/veo_frames_raw_final"

SRC_IMAGES = SRC_DIR / "images"
SRC_LABELS = SRC_DIR / "labels"

DST_IMAGES = DST_DIR / "images"
DST_LABELS = DST_DIR / "labels"

FLAG = "# revisado_manual"

def main():
    print("[INFO] Preparando dataset final de entrenamiento (Solo revisado por humanos)...")
    
    # Crear estructura básica
    DST_IMAGES.mkdir(parents=True, exist_ok=True)
    DST_LABELS.mkdir(parents=True, exist_ok=True)
    
    if not SRC_LABELS.exists():
        print("[ERROR] Error: No se encontro la carpeta de labels origen.")
        return

    label_files = list(SRC_LABELS.glob("*.txt"))
    print(f"[INFO] Escaneando {len(label_files)} archivos de etiquetas...")
    
    copied_count = 0
    
    for lab_path in label_files:
        try:
            content = lab_path.read_text(encoding="utf-8")
            if FLAG in content:
                # 1. Identificar imagen correspondiente
                # Intentamos con .jpg primero
                img_path = SRC_IMAGES / (lab_path.stem + ".jpg")
                if not img_path.exists():
                    img_path = SRC_IMAGES / (lab_path.stem + ".png")
                
                if not img_path.exists():
                    print(f"   [WARN] Imagen no encontrada para {lab_path.name}, saltando.")
                    continue
                
                # 2. Limpiar etiqueta (quitar comentario)
                lines = [l.strip() for l in content.splitlines() if l.strip() and not l.strip().startswith("#")]
                clean_content = "\n".join(lines)
                
                # 3. Copiar
                shutil.copy2(img_path, DST_IMAGES / img_path.name)
                (DST_LABELS / lab_path.name).write_text(clean_content, encoding="utf-8")
                
                copied_count += 1
                if copied_count % 50 == 0:
                    print(f"   [OK] Copiados {copied_count} archivos...")
        except Exception as e:
            print(f"   [ERROR] Error procesando {lab_path.name}: {e}")

    print("\n" + "="*40)
    print("PROCESO COMPLETADO")
    print("="*40)
    print(f"[INFO] Dataset destino: {DST_DIR}")
    print(f"[INFO] Imagenes/Labels listos: {copied_count}")
    print("="*40)
    if copied_count > 0:
        print("Tip: Ya puedes usar esta carpeta en tu data.yaml para entrenar.")
    else:
        print("Tip: Aun no has revisado ninguna imagen manualmente en el Labeller.")

if __name__ == "__main__":
    main()
