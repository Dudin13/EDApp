"""
clean_seg_labels.py — Elimina etiquetas de segmentacion con poligonos vacios o corruptos
que causan IndexError durante el entrenamiento de YOLOv8-seg.
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATASET_DIR = ROOT / "combined_dataset"

def clean_labels(split="train"):
    labels_dir = DATASET_DIR / split / "labels"
    if not labels_dir.exists():
        print(f"[WARN] No existe: {labels_dir}")
        return 0

    removed = 0
    fixed = 0
    files = list(labels_dir.glob("*.txt"))
    print(f"[INFO] Revisando {len(files)} etiquetas en {labels_dir}...")

    for label_file in files:
        lines = label_file.read_text(encoding="utf-8", errors="replace").strip().split("\n")
        valid_lines = []
        has_issues = False

        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # Formato YOLO-seg: class_id cx cy + poligono (minimo 6 valores = 3 puntos)
            # Total minimo: 1 (clase) + 4 (bbox) = 5, o 1 + poligono >=6 puntos = 7 valores
            if len(parts) < 5:
                has_issues = True
                continue
            # Para segmentacion: clase + puntos del poligono (pares x,y)
            # La clase va en partes[0], luego deben haber pares de coordenadas
            try:
                cls = int(float(parts[0]))
            except ValueError:
                has_issues = True
                continue

            coords = parts[1:]
            # El poligono debe tener al menos 3 puntos (6 valores)
            if len(coords) < 6:
                has_issues = True
                continue
            if len(coords) % 2 != 0:
                # Numero impar de coordenadas, truncar
                coords = coords[:-1]
                has_issues = True

            # Verificar que todas las coordenadas son floats validos entre 0 y 1
            try:
                vals = [float(v) for v in coords]
                if any(v < 0 or v > 1 for v in vals):
                    # Normalizar si estan fuera de rango
                    vals = [max(0.0, min(1.0, v)) for v in vals]
                    has_issues = True
            except ValueError:
                has_issues = True
                continue

            # Recontsruir linea limpia
            clean_coords = " ".join(f"{v:.6f}" for v in vals)
            valid_lines.append(f"{cls} {clean_coords}")

        if not valid_lines:
            # Archivo completamente vacio o invalido -> eliminar
            label_file.unlink()
            # Eliminar imagen asociada si existe
            for ext in [".jpg", ".jpeg", ".png"]:
                img = DATASET_DIR / split / "images" / (label_file.stem + ext)
                if img.exists():
                    img.unlink()
                    print(f"  [DEL] {label_file.stem} (sin anotaciones validas)")
            removed += 1
        elif has_issues:
            # Hay problemas pero algo es aprovechable -> reescribir limpio
            label_file.write_text("\n".join(valid_lines) + "\n", encoding="utf-8")
            fixed += 1

    print(f"[OK] Eliminados: {removed} | Corregidos: {fixed} | OK: {len(files) - removed - fixed}")
    return removed + fixed


if __name__ == "__main__":
    print("=" * 50)
    print("  Limpieza de etiquetas de segmentacion")
    print("=" * 50)
    n = clean_labels("train")
    n += clean_labels("valid")
    print(f"\n[TOTAL] {n} archivos procesados.")
    print("[INFO] Ahora puedes relanzar el entrenamiento.")
