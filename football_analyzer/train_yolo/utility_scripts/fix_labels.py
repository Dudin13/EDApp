
import os
from pathlib import Path

def fix_labels(directory):
    path = Path(directory)
    label_files = list(path.rglob("*.txt"))
    
    # Mapping: COCO Person (0) -> Player (1), COCO Ball (32) -> Ball (2)
    # Mapping for referee/gk might need manual work, but for now we prioritize Player and Ball.
    mapping = {0: 1, 32: 2} 
    
    print(f"Fixing labels in {directory}...")
    fixed_count = 0
    for lbfile in label_files:
        with open(lbfile, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            
            old_cls = int(parts[0])
            if old_cls in mapping:
                parts[0] = str(mapping[old_cls])
                new_lines.append(" ".join(parts) + "\n")
            elif old_cls == 1: # Already 1? keep it
                 new_lines.append(" ".join(parts) + "\n")
            elif old_cls == 2: # Already 2? keep it
                 new_lines.append(" ".join(parts) + "\n")
            # Discard others (referees, suitcases, etc) for now to keep it clean
            
        with open(lbfile, 'w') as f:
            f.writelines(new_lines)
        fixed_count += 1
    
    print(f"Fixed {fixed_count} files.")

if __name__ == "__main__":
    fix_labels("c:/apped/football_analyzer/dataset_pro")
