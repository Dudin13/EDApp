import os
import shutil
from pathlib import Path

def main():
    # Source path where the dataset is expected to be extracted
    src_dir = Path("scripts/experimental/SoccerSynth-Detection")
    
    # Target path for the remapped dataset
    dst_dir = Path("data/datasets/soccersynth_remapped")
    dst_images = dst_dir / "images"
    dst_labels = dst_dir / "labels"
    
    # Create target directories
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning for label files in {src_dir}...")
    
    # We will search recursively for any .txt files
    # Exclude .git and system files
    label_files = []
    for root, dirs, files in os.walk(src_dir):
        if ".git" in root:
            continue
        for file in files:
            if file.endswith(".txt") and file != "requirements.txt" and file != "CMakeLists.txt":
                label_files.append(Path(root) / file)
                
    total_found = len(label_files)
    print(f"Found {total_found} label files.")
    
    if total_found == 0:
        print("\n[INFO] No se encontraron archivos de anotaciones (.txt) en SoccerSynth-Detection.")
        print("El dataset completo de SoccerSynth-Detection contiene 54,673 imágenes (37,621 normales y 17,052 borrosas).")
        print("Por favor, descarga 'SoccerSynth_Detection.zip' desde el enlace de Google Drive en el README,")
        print(f"y extráelo en la carpeta: {src_dir.absolute()}")
        return
        
    processed_count = 0
    for label_path in label_files:
        # Read labels
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading {label_path}: {e}")
            continue
            
        remapped_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
                # Remap class:
                # 0 (ball) -> 3 (ball in EDApp)
                # 1, 2, 3 (players) -> 0 (player in EDApp)
                if class_id == 0:
                    new_class_id = 3
                elif class_id in (1, 2, 3):
                    new_class_id = 0
                else:
                    # Keep as is or skip if unknown
                    new_class_id = class_id
                
                parts[0] = str(new_class_id)
                remapped_lines.append(" ".join(parts))
            except ValueError:
                # Skip header or malformed line
                continue
                
        # Determine image path
        # Check standard image extensions
        img_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
        img_path = None
        for ext in img_extensions:
            candidate = label_path.with_suffix(ext)
            if candidate.exists():
                img_path = candidate
                break
                
        # Save remapped label
        dst_label_path = dst_labels / label_path.name
        try:
            with open(dst_label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(remapped_lines) + "\n")
        except Exception as e:
            print(f"Error writing label to {dst_label_path}: {e}")
            continue
            
        # Copy image if found
        if img_path:
            dst_img_path = dst_images / img_path.name
            try:
                if not dst_img_path.exists():
                    os.link(img_path, dst_img_path) # Fast copy via hardlink
            except:
                try:
                    shutil.copy(img_path, dst_img_path)
                except Exception as e:
                    print(f"Error copying image {img_path} to {dst_img_path}: {e}")
                    
        processed_count += 1
        
    print(f"\nRemapeo completado exitosamente.")
    print(f"Archivos procesados: {processed_count} / {total_found}")
    print(f"Imágenes y etiquetas guardadas en: {dst_dir.absolute()}")

if __name__ == "__main__":
    main()
