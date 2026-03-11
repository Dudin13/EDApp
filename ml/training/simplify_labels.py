import os
from pathlib import Path

def simplify_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    label_dirs = [dataset_path / "train" / "labels", dataset_path / "valid" / "labels"]
    
    # Mapping: 0=player, 1=goalkeeper -> 0=person
    # 2=referee -> 1=referee
    # 3=ball -> 2=ball
    
    mapping = {
        '0': '0', # player -> person
        '1': '0', # goalkeeper -> person
        '2': '1', # referee -> referee
        '3': '2'  # ball -> ball
    }
    
    print(f"🔄 Simplificando etiquetas en {dataset_path}...")
    
    count = 0
    for label_dir in label_dirs:
        if not label_dir.exists():
            continue
            
        for label_file in label_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                
                old_cls = parts[0]
                if old_cls in mapping:
                    parts[0] = mapping[old_cls]
                    new_lines.append(" ".join(parts) + "\n")
            
            with open(label_file, 'w') as f:
                f.writelines(new_lines)
            count += 1
            
    print(f"✅ Se han simplificado {count} archivos de etiquetas.")

if __name__ == "__main__":
    simplify_dataset("04_Datasets_Entrenamiento/hybrid_dataset")
