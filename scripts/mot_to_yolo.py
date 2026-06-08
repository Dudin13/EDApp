"""
mot_to_yolo.py
===============
Convierte anotaciones MOT Challenge (gt.txt) al formato YOLO
para entrenamiento con Ultralytics YOLOv8/v11.

Mapeado de clases:
    SoccerTrack MOT class → EDApp YOLO class
    1 (player)            → 0 (player)
    2 (goalkeeper)        → 1 (goalkeeper)
    3 (referee)           → 2 (referee)
    (balón no incluido en MOT de SoccerTrack v2)

Uso:
    python scripts/mot_to_yolo.py \\
        --gt  c:/apped/data/datasets/soccertrack_v2/MATCH_ID/mot/gt/gt.txt \\
        --seq c:/apped/data/datasets/soccertrack_v2/MATCH_ID/mot/seqinfo.ini \\
        --out c:/apped/data/datasets/soccertrack_v2/yolo_labels/

Formato YOLO output por frame (labels/000001.txt):
    class_id cx cy w h
    (normalizados 0-1 respecto al tamaño del frame)
"""

import argparse
import configparser
from pathlib import Path
from collections import defaultdict


# ── Configuración de clases ────────────────────────────────────────────────

MOT_TO_YOLO = {
    -1: 0,  # default / player → 0
    1: 0,   # player     → 0
    2: 1,   # goalkeeper → 1
    3: 2,   # referee    → 2
    # Balón (class 4) no existe en SoccerTrack v2 MOT → ignorado
}

CLASS_NAMES = {0: "player", 1: "goalkeeper", 2: "referee"}


# ── Parser de seqinfo.ini ──────────────────────────────────────────────────

def parse_seqinfo(seqinfo_path: Path) -> dict:
    """Lee los metadatos de la secuencia MOT."""
    config = configparser.ConfigParser()
    config.read(str(seqinfo_path))

    seq = config["Sequence"]
    info = {
        "name":       seq.get("name", "unknown"),
        "frame_rate": int(seq.get("framerate", 25)),
        "seq_length": int(seq.get("seqlength", 0)),
        "im_width":   int(seq.get("imwidth", 3840)),
        "im_height":  int(seq.get("imheight", 2160)),
        "im_ext":     seq.get("imext", ".jpg"),
    }
    print(f"[seqinfo] Nombre:     {info['name']}")
    print(f"[seqinfo] Frames:     {info['seq_length']}")
    print(f"[seqinfo] FPS:        {info['frame_rate']}")
    print(f"[seqinfo] Resolución: {info['im_width']}×{info['im_height']}")
    return info


# ── Parser y conversor de gt.txt ───────────────────────────────────────────

def parse_gt_txt(gt_path: Path) -> dict:
    """
    Lee gt.txt y agrupa anotaciones por frame.
    
    Returns:
        dict {frame_id: [(track_id, bb_left, bb_top, bb_width, bb_height, class_id), ...]}
    """
    frames = defaultdict(list)
    skipped_classes = defaultdict(int)
    total = 0

    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            parts = line.split(",")
            if len(parts) < 8:
                continue
            
            frame_id = int(parts[0])
            track_id = int(parts[1])
            bb_left   = float(parts[2])
            bb_top    = float(parts[3])
            bb_width  = float(parts[4])
            bb_height = float(parts[5])
            conf      = float(parts[6])
            class_id  = int(parts[7])
            # visibility = float(parts[8]) if len(parts) > 8 else 1.0

            total += 1

            if class_id not in MOT_TO_YOLO:
                skipped_classes[class_id] += 1
                continue

            frames[frame_id].append({
                "track_id": track_id,
                "class_id": class_id,
                "bb_left":  bb_left,
                "bb_top":   bb_top,
                "bb_width": bb_width,
                "bb_height": bb_height,
                "conf":     conf,
            })

    print(f"[gt.txt] Total anotaciones: {total}")
    print(f"[gt.txt] Frames únicos:     {len(frames)}")
    if skipped_classes:
        for cls, count in skipped_classes.items():
            print(f"[gt.txt] Clase {cls} omitida: {count} anotaciones (no en MOT_TO_YOLO)")
    
    return frames


def convert_bbox_to_yolo(bb_left, bb_top, bb_width, bb_height, img_w, img_h):
    """Convierte bbox MOT (absoluta) a formato YOLO (normalizada)."""
    cx = (bb_left + bb_width  / 2) / img_w
    cy = (bb_top  + bb_height / 2) / img_h
    nw = bb_width  / img_w
    nh = bb_height / img_h

    # Clamp [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))

    return cx, cy, nw, nh


# ── Main ───────────────────────────────────────────────────────────────────

def convert(gt_path: Path, seqinfo_path: Path, output_dir: Path):
    """Conversión completa MOT → YOLO labels."""

    print(f"\n{'='*60}")
    print(f"  MOT -> YOLO Conversor")
    print(f"  SoccerTrack v2 -> EDApp YOLO format")
    print(f"{'='*60}\n")

    # 1. Leer metadatos
    info = parse_seqinfo(seqinfo_path)
    img_w = info["im_width"]
    img_h = info["im_height"]
    match_name = info["name"]
    print()

    # 2. Leer gt.txt
    frames = parse_gt_txt(gt_path)
    print()

    # 3. Crear directorio de salida
    out_dir = output_dir / match_name / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] Directorio: {out_dir}")

    # 4. Convertir cada frame
    stats = defaultdict(int)
    frames_with_players = set()
    frames_per_class = defaultdict(set)

    for frame_id, detections in sorted(frames.items()):
        label_filename = f"{frame_id:06d}.txt"
        label_path = out_dir / label_filename

        lines = []
        for det in detections:
            yolo_cls = MOT_TO_YOLO[det["class_id"]]
            cx, cy, nw, nh = convert_bbox_to_yolo(
                det["bb_left"], det["bb_top"],
                det["bb_width"], det["bb_height"],
                img_w, img_h
            )
            lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            stats[yolo_cls] += 1
            frames_per_class[yolo_cls].add(frame_id)
            if yolo_cls in (0, 1):  # player o goalkeeper
                frames_with_players.add(frame_id)

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    # 5. Estadísticas finales
    print(f"\n{'='*60}")
    print(f"  ESTADISTICAS DE CONVERSION")
    print(f"{'='*60}")
    print(f"  Partido:              {match_name}")
    print(f"  Duracion total:       {info['seq_length']} frames @ {info['frame_rate']} FPS")
    print(f"  Duracion (segundos):  {info['seq_length'] / info['frame_rate']:.1f} s")
    print(f"  Frames anotados:      {len(frames)}")
    print(f"  Archivos .txt escritos: {len(frames)}")
    print()
    print(f"  Anotaciones por clase:")
    for yolo_id, name in CLASS_NAMES.items():
        count = stats[yolo_id]
        f_count = len(frames_per_class[yolo_id])
        print(f"    [{yolo_id}] {name:<12}: {count:>6} bboxes en {f_count:>5} frames")
    print()
    print(f"  * Frames con jugadores (player+goalkeeper): {len(frames_with_players)}")
    print(f"  * Ratio util: {len(frames_with_players)/info['seq_length']*100:.1f}% del video total")
    print()
    print(f"  Labels guardados en: {out_dir}")
    print(f"{'='*60}\n")

    # 6. Guardar resumen como txt
    summary_path = output_dir / match_name / "conversion_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Partido: {match_name}\n")
        f.write(f"Frames totales: {info['seq_length']}\n")
        f.write(f"Frames anotados: {len(frames)}\n")
        f.write(f"Frames con jugadores: {len(frames_with_players)}\n")
        f.write(f"Resolución original: {img_w}x{img_h}\n")
        f.write(f"FPS: {info['frame_rate']}\n")
        for yolo_id, name in CLASS_NAMES.items():
            f.write(f"Clase {yolo_id} ({name}): {stats[yolo_id]} bboxes\n")

    return {
        "match": match_name,
        "total_frames": info["seq_length"],
        "annotated_frames": len(frames),
        "frames_with_players": len(frames_with_players),
        "stats": dict(stats),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convierte MOT gt.txt → YOLO labels")
    parser.add_argument("--gt",  required=True, help="Ruta al gt.txt")
    parser.add_argument("--seq", required=True, help="Ruta al seqinfo.ini")
    parser.add_argument("--out", default="c:/apped/data/datasets/soccertrack_v2/yolo_labels",
                        help="Directorio de salida para los labels YOLO")
    args = parser.parse_args()

    result = convert(
        gt_path=Path(args.gt),
        seqinfo_path=Path(args.seq),
        output_dir=Path(args.out),
    )
    print(f"\nResumen final: {result}")
