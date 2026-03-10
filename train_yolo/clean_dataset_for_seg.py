import os
from pathlib import Path

def clean_labels(dataset_path):
    dataset_path = Path(dataset_path)
    for split in ['train', 'valid']:
        labels_dir = dataset_path / split / "labels"
        images_dir = dataset_path / split / "images"
        
        if not labels_dir.exists(): continue
        
        files = list(labels_dir.glob("*.txt"))
        removed_count = 0
        total_count = len(files)
        
        for lb_file in files:
            with open(lb_file, 'r') as f:
                lines = f.readlines()
            
            # Un label de segmento tiene > 5 valores (class, x1, y1, x2, y2, x3, y3...)
            # Un label de caja tiene exactamente 5 valores (class, x, y, w, h)
            valid_lines = [l for l in lines if len(l.split()) > 5]
            
            if not valid_lines:
                # Si no hay segmentos, eliminamos imagen y label para esta sesion de entrenamiento
                img_file = images_dir / (lb_file.stem + ".jpg")
                if img_file.exists(): 
                    # En lugar de borrar, movemos a una carpeta temp para no perder datos
                    temp_dir = dataset_path / "temp_boxes_only"
                    temp_dir.mkdir(exist_ok=True)
                    os.rename(img_file, temp_dir / img_file.name)
                    os.rename(lb_file, temp_dir / lb_file.name)
                    removed_count += 1
            else:
                # Si hay mezcla, dejamos solo los segmentos
                with open(lb_file, 'w') as f:
                    f.writelines(valid_lines)

        print(f"[{split}] Limpieza completada. Eliminados {removed_count}/{total_count} archivos sin siluetas.")

if __name__ == "__main__":
    clean_labels("04_Datasets_Entrenamiento/hybrid_dataset")
