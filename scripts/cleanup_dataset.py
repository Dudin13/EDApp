import os
import shutil

def get_file_size(path):
    try:
        return os.path.getsize(path)
    except:
        return 0

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
    # Try common YOLO structures
    # 1. same folder
    # 2. ../labels/ (if in images/)
    # 3. ../../labels/ (if in train/images/)
    
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
    root_data = r'c:\apped\data\datasets'
    
    # 1. Specific deletions
    files_to_delete = [
        os.path.join(root_data, 'dataset', 'test_segmentation_video.mp4'),
        os.path.join(root_data, 'soccernet_h250', 'soccernet_yolo.zip')
    ]
    
    freed_space = 0
    for f in files_to_delete:
        if os.path.exists(f):
            size = get_file_size(f)
            print(f"Deleting {f} ({size/1024/1024:.2f} MB)")
            os.remove(f)
            freed_space += size
        else:
            print(f"File not found: {f}")

    # 2. Deduplication
    print("Scanning for duplicates...")
    image_extensions = ('.jpg', '.png', '.jpeg')
    images_by_name = {}
    
    for root, dirs, files in os.walk(root_data):
        for f in files:
            if f.lower().endswith(image_extensions):
                full_path = os.path.join(root, f)
                if f not in images_by_name:
                    images_by_name[f] = []
                images_by_name[f].append(full_path)
    
    duplicate_count = 0
    deleted_images = 0
    
    for name, paths in images_by_name.items():
        if len(paths) > 1:
            duplicate_count += 1
            # Decide which one to keep
            # Priority: has '# revisado_manual' in label
            
            best_path = None
            for p in paths:
                label_p = find_label_path(p)
                if label_p and has_manual_review(label_p):
                    best_path = p
                    break
            
            if not best_path:
                best_path = paths[0] # Just keep the first one
            
            # Delete others
            for p in paths:
                if p != best_path:
                    size = get_file_size(p)
                    # Also try to delete its label if it exists and is NOT the best label
                    label_p = find_label_path(p)
                    best_label_p = find_label_path(best_path)
                    
                    try:
                        os.remove(p)
                        freed_space += size
                        deleted_images += 1
                        if label_p and label_p != best_label_p:
                            label_size = get_file_size(label_p)
                            os.remove(label_p)
                            freed_space += label_size
                    except Exception as e:
                        print(f"Error deleting {p}: {e}")

    print(f"\nDeduplication Summary:")
    print(f"Sets of duplicates found: {duplicate_count}")
    print(f"Duplicate images deleted: {deleted_images}")
    print(f"Total space freed: {freed_space/1024/1024:.2f} MB ({freed_space/1024/1024/1024:.2f} GB)")

if __name__ == "__main__":
    main()
