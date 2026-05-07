import os
import shutil

def has_manual_review(label_path):
    if not os.path.exists(label_path):
        return False
    try:
        with open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            return '# revisado_manual' in content
    except:
        return False

def find_label_path(image_path):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    dir_path = os.path.dirname(image_path)
    # Same folder
    p1 = os.path.join(dir_path, base_name + '.txt')
    if os.path.exists(p1): return p1
    # Sibling labels folder
    if os.path.basename(dir_path) == 'images':
        p2 = os.path.join(os.path.dirname(dir_path), 'labels', base_name + '.txt')
        if os.path.exists(p2): return p2
    return None

def main():
    root_data = r'c:\apped\data'
    datasets_path = os.path.join(root_data, 'datasets')
    output_path = os.path.join(root_data, 'para_etiquetar', 'mediacoach_pendientes')
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    total_revisados = 0
    total_pendientes = 0
    mediacoach_pendientes_count = 0
    
    image_extensions = ('.jpg', '.png', '.jpeg')
    
    print("Scanning datasets for summary and Mediacoach pending...")
    
    for root, dirs, files in os.walk(datasets_path):
        for f in files:
            if f.lower().endswith(image_extensions):
                image_path = os.path.join(root, f)
                label_path = find_label_path(image_path)
                
                is_revisado = False
                if label_path and has_manual_review(label_path):
                    is_revisado = True
                    total_revisados += 1
                else:
                    total_pendientes += 1
                
                # Check if it is Mediacoach and pending
                if 'mediacoach' in f.lower() and not is_revisado:
                    mediacoach_pendientes_count += 1
                    # Copy to para_etiquetar
                    shutil.copy2(image_path, os.path.join(output_path, f))
                    if label_path:
                        shutil.copy2(label_path, os.path.join(output_path, os.path.basename(label_path)))
    
    print(f"\nTask 2 Results:")
    print(f"Mediacoach pending images copied: {mediacoach_pendientes_count}")
    
    print(f"\nPreliminary Summary Data:")
    print(f"Total images with # revisado_manual: {total_revisados}")
    print(f"Total images pending: {total_pendientes}")

if __name__ == "__main__":
    main()
