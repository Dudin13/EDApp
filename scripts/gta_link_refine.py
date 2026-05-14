import os
import pickle
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict

def main():
    # Paths
    mot_path = "output/tracks_raw.txt"
    reid_pkl = "output/reid_features.pkl"
    output_refined = "output/tracks_refined.txt"
    
    if not os.path.exists(mot_path) or not os.path.exists(reid_pkl):
        print(f"Error: Required files not found.")
        return

    # Load raw tracks
    raw_data = np.loadtxt(mot_path, delimiter=',')
    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(1, -1)

    # Group by track id
    raw_tracks = defaultdict(list)
    for line in raw_data:
        tid = int(line[1])
        raw_tracks[tid].append(line)
        
    # Load ReID features
    with open(reid_pkl, "rb") as f:
        reid_features = pickle.load(f)
        
    # Phase 3.1: Tracklet Splitting (DBSCAN)
    # Detect if a track contains more than one person based on ReID inconsistency
    next_id = int(np.max(raw_data[:, 1])) + 1
    
    processed_tracks = []
    
    print("Fase 3.1: Dividiendo tracklets inconsistentes...")
    for tid, lines in raw_tracks.items():
        feats = []
        valid_indices = []
        for i, line in enumerate(lines):
            f_num = int(line[0])
            if (tid, f_num) in reid_features:
                feats.append(reid_features[(tid, f_num)])
                valid_indices.append(i)
        
        if len(feats) < 8: # Too short to split reliably
            processed_tracks.append(lines)
            continue
            
        feats = np.array(feats)
        # Cosine distance = 1 - dot product (for L2 normalized)
        sim_matrix = np.dot(feats, feats.T)
        dist_matrix = np.clip(1 - sim_matrix, 0, 1)
        
        # DBSCAN clustering
        # Un eps de 0.6-0.7 es tipico para OSNet en futbol
        clustering = DBSCAN(eps=0.75, min_samples=2, metric='precomputed').fit(dist_matrix)
        labels = clustering.labels_
        
        unique_labels = set(labels)
        if len(unique_labels) <= 1:
            processed_tracks.append(lines)
        else:
            # Split track
            label_to_new_id = {}
            for i, idx in enumerate(valid_indices):
                label = labels[i]
                if label == -1: continue # Noise, keep old or discard? Let's assign to largest cluster
                if label not in label_to_new_id:
                    if len(label_to_new_id) == 0:
                        label_to_new_id[label] = tid
                    else:
                        label_to_new_id[label] = next_id
                        next_id += 1
                
                lines[idx][1] = label_to_new_id[label]
            
            # Re-group after splitting
            new_groups = defaultdict(list)
            for line in lines:
                new_groups[int(line[1])].append(line)
            for g in new_groups.values():
                processed_tracks.append(g)

    # Phase 3.2: Tracklet Connection (Cosine Similarity)
    # Merge tracklets that belong to the same person
    processed_tracks = [t for t in processed_tracks if len(t) > 1]
    print(f"Fase 3.2: Conectando tracklets (Total filtrados > 1 frame: {len(processed_tracks)})")
    
    # Calculate mean feature for each tracklet
    tracklet_data = []
    for lines in processed_tracks:
        tid = int(lines[0][1])
        feats = []
        for line in lines:
            f_num = int(line[0])
            if (tid, f_num) in reid_features:
                feats.append(reid_features[(tid, f_num)])
        
        if not feats: continue
        
        mean_feat = np.mean(feats, axis=0)
        mean_feat /= np.linalg.norm(mean_feat)
        
        tracklet_data.append({
            'id': tid,
            'lines': [list(l) for l in lines],
            'feat': mean_feat,
            'start': lines[0][0],
            'end': lines[-1][0]
        })
        
    # Simple Greedy Connection
    # We want to merge tracklets i and j if they are visually similar and don't overlap in time
    merged_indices = [False] * len(tracklet_data)
    final_groups = []
    
    # Sort by start time
    tracklet_data.sort(key=lambda x: x['start'])
    
    for i in range(len(tracklet_data)):
        if merged_indices[i]: continue
        
        current_id = tracklet_data[i]['id']
        current_lines = tracklet_data[i]['lines']
        current_feat = tracklet_data[i]['feat']
        current_end = tracklet_data[i]['end']
        merged_indices[i] = True
        
        while True:
            best_j = -1
            max_sim = 0.75 # Threshold de similitud para union global
            
            for j in range(len(tracklet_data)):
                if merged_indices[j]: continue
                # Temporal gap constraint: no overlap
                if tracklet_data[j]['start'] <= current_end: continue 
                
                # Visual similarity
                sim = np.dot(current_feat, tracklet_data[j]['feat'])
                if sim > max_sim:
                    max_sim = sim
                    best_j = j
            
            if best_j != -1:
                # Merge best_j into current
                merged_indices[best_j] = True
                # Update ID of lines and add to current
                for line in tracklet_data[best_j]['lines']:
                    line[1] = current_id
                    current_lines.append(line)
                
                # Update cumulative feature (weighted by length if possible, here simple avg)
                current_feat = (current_feat + tracklet_data[best_j]['feat']) / 2
                current_feat /= np.linalg.norm(current_feat)
                current_end = tracklet_data[best_j]['end']
            else:
                break
        
        final_groups.append(current_lines)

    # Save refined tracks
    from collections import Counter
    refined_all_lines = []
    
    for g in final_groups:
        # Calcular la moda de la clase y el equipo para todo el track fusionado
        # asumiendo que los índices 10 (clase) y 11 (equipo) existen. Si no, default a 0 y -1
        clases = [int(line[10]) if len(line) > 10 else 0 for line in g]
        equipos = [int(line[11]) if len(line) > 11 else -1 for line in g]
        
        mode_clase = Counter(clases).most_common(1)[0][0] if clases else 0
        mode_equipo = Counter(equipos).most_common(1)[0][0] if equipos else -1
        
        for line in g:
            # Añadimos los campos a la línea si no existen para estandarizar
            line_out = list(line)
            while len(line_out) < 12:
                line_out.append(0)
            line_out[10] = mode_clase
            line_out[11] = mode_equipo
            refined_all_lines.append(line_out)
    
    # Sort by frame, then id
    refined_all_lines.sort(key=lambda x: (x[0], x[1]))
    
    with open(output_refined, "w") as f:
        for line in refined_all_lines:
            f.write(f"{int(line[0])},{int(line[1])},{line[2]:.1f},{line[3]:.1f},{line[4]:.1f},{line[5]:.1f},{line[6]:.1f},-1,-1,-1,{int(line[10])},{int(line[11])}\n")
            
    print(f"Refined tracks saved to {output_refined}.")

    # Phase 4: Validation
    raw_ids = len(raw_tracks)
    refined_ids = len(final_groups)
    
    raw_lengths = [len(lines) for lines in raw_tracks.values()]
    refined_lengths = [len(lines) for lines in final_groups]
    
    print("\n" + "="*50)
    print("GTA-LINK VALIDATION RESULTS")
    print("="*50)
    print(f"Unique IDs (Before): {raw_ids}")
    print(f"Unique IDs (After):  {refined_ids}")
    reduction = 100*(1 - refined_ids/raw_ids) if raw_ids > 0 else 0
    print(f"Reduction:           {reduction:.1f}%")
    print("-" * 30)
    print(f"Mean Track Length (Before): {np.mean(raw_lengths):.2f} frames")
    print(f"Mean Track Length (After):  {np.mean(refined_lengths):.2f} frames")
    print("="*50)

if __name__ == "__main__":
    main()
