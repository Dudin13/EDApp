import os

path = 'data/datasets/veo_frames_raw/labels/'
files = [f for f in os.listdir(path) if f.startswith('J10_DH_2526_CadeteA_SevillaFC')]
count_manual = 0
missing_files = []

for f in files:
    full_path = os.path.join(path, f)
    if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
        with open(full_path, 'r', encoding='utf-8') as file:
            line = file.readline()
            if '# revisado_manual' in line:
                count_manual += 1
            else:
                missing_files.append(f)

print(f"Total files for J10: {len(files)}")
print(f"Files WITH manual flag: {count_manual}")
print(f"Files MISSING manual flag: {len(missing_files)}")
if missing_files:
    print(f"Sample missing: {missing_files[:5]}")
