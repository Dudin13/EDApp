import os
from datetime import datetime

path = 'data/datasets/veo_frames_raw/labels/'
files = os.listdir(path)
stats = {}
for f in files:
    if not f.endswith('.txt'): continue
    prefix = f.split('_f')[0].split('_min')[0]
    mtime = os.path.getmtime(os.path.join(path, f))
    if prefix not in stats:
        stats[prefix] = {'min': mtime, 'max': mtime, 'count': 0}
    stats[prefix]['min'] = min(stats[prefix]['min'], mtime)
    stats[prefix]['max'] = max(stats[prefix]['max'], mtime)
    stats[prefix]['count'] += 1

print(f"{'Prefix':<50} | {'Count':<5} | {'Min Date':<20} | {'Max Date':<20}")
print("-" * 105)
for p, s in sorted(stats.items(), key=lambda x: x[1]['max'], reverse=True):
    min_d = datetime.fromtimestamp(s['min']).strftime('%Y-%m-%d %H:%M:%S')
    max_d = datetime.fromtimestamp(s['max']).strftime('%Y-%m-%d %H:%M:%S')
    print(f"{p:<50} | {s['count']:<5} | {min_d:<20} | {max_d:<20}")
