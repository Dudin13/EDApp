import os

path = 'data/datasets/veo_frames_raw/labels/'
if not os.path.exists(path):
    print(f"Directory not found: {path}")
    exit(1)

files = os.listdir(path)
prefixes = {}
for f in files:
    if not f.endswith('.txt'):
        continue
    # Extract prefix
    prefix = f
    if '_f' in f:
        prefix = f.split('_f')[0]
    if '_min' in f:
        prefix = prefix.split('_min')[0]
    
    if prefix not in prefixes:
        prefixes[prefix] = []
    prefixes[prefix].append(f)

print(f"Total prefixes: {len(prefixes)}")
print("-" * 50)
for p, fs in sorted(prefixes.items()):
    sample_file = fs[0]
    full_path = os.path.join(path, sample_file)
    first_line = ""
    if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
        with open(full_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
    
    print(f"{p}: First line is '{first_line}' (Sample: {sample_file})")
