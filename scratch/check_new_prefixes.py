import os
from datetime import datetime

path = 'data/datasets/veo_frames_raw/images/'
if not os.path.exists(path):
    print(f"Directory not found: {path}")
    exit(1)

files = os.listdir(path)
stats = {}
for f in files:
    if not (f.endswith('.jpg') or f.endswith('.png')):
        continue
    # Extract prefix: split by _f or _min
    prefix = f
    if '_f' in f:
        prefix = f.split('_f')[0]
    if '_min' in f:
        prefix = prefix.split('_min')[0]
    
    mtime = os.path.getmtime(os.path.join(path, f))
    if prefix not in stats:
        stats[prefix] = {'count': 0, 'latest': 0}
    stats[prefix]['count'] += 1
    stats[prefix]['latest'] = max(stats[prefix]['latest'], mtime)

print(f"Total images: {len(files)}")
print("-" * 50)
for p, s in sorted(stats.items(), key=lambda x: x[1]['latest'], reverse=True):
    dt = datetime.fromtimestamp(s['latest']).strftime('%Y-%m-%d %H:%M:%S')
    print(f"{p}: {s['count']} images, last modified: {dt}")
