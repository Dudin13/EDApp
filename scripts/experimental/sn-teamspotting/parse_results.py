import json
import os

path = r'c:\apped\sn-teamspotting\outputs\SoccerNetBall_test_infer\pred-challenge\test_5min\results_spotting.json'
with open(path) as f:
    data = json.load(f)

events = data['predictions']
print(f'Total events: {len(events)}')

stats = {}
for e in events:
    label = e['label']
    team = e['team']
    if label not in stats:
        stats[label] = {'count': 0, 'conf': 0, 'teams': {}}
    stats[label]['count'] += 1
    stats[label]['conf'] += float(e['confidence'])
    stats[label]['teams'][team] = stats[label]['teams'].get(team, 0) + 1

print('\nClass Breakdown:')
sorted_stats = sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True)
for label, s in sorted_stats:
    avg_conf = s['conf'] / s['count']
    teams_str = ', '.join([f'{t}: {c}' for t, c in s['teams'].items()])
    print(f'- {label}: {s["count"]} events, Avg Conf: {avg_conf:.4f}, Teams: {teams_str}')
