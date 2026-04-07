"""
run_demo_analysis.py
====================
Script de prueba end-to-end del pipeline EDApp.
Procesa test_clip_2m30s.mp4 y genera un video anotado con:
  - Detección de jugadores (players.pt V4 Clean)
  - Clasificación de equipos (TeamClassifier)
  - Tracking con ID persistente (ByteTrack)
  - Calibración automática del campo (AutoCalibrator)
  - Minimap táctico en tiempo real
  - Eventos detectados (EventSpotter)
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# ── Rutas del proyecto ─────────────────────────────────────────────────────
ROOT       = Path("C:/apped")
VIDEO_PATH = ROOT / "data/samples/test_5min.mp4" # Cambiado de test_clip_2m30s.mp4 por el usuario
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "demo_result.mp4"

sys.path.insert(0, str(ROOT / "app"))

# ── Imports del pipeline (REORDENADOS PARA EVITAR HANG) ────────────────────
from modules.calibration_auto import AutoCalibrator

# El resto se cargará después

# ── Configuración ──────────────────────────────────────────────────────────
MODEL_PLAYERS = str(ROOT / "models/players.pt")
MODEL_PITCH   = str(ROOT / "models/pitch.pt")
SAMPLE_RATE   = 0.5   # procesar 1 frame cada 0.5s
IMGSZ         = 1280  # crítico para VEO panorámico
CONF          = 0.20

MINIMAP_W, MINIMAP_H = 320, 213
MINIMAP_X, MINIMAP_Y = 20, 20   # posición en el frame

print("=" * 60)
print("EDApp — Test End-to-End")
print("=" * 60)
print(f"Video:   {VIDEO_PATH}")
print(f"Output:  {OUTPUT_PATH}")
print(f"imgsz:   {IMGSZ}")

# ── Inicializar módulos ────────────────────────────────────────────────────
print("\n[1/6] Cargando AutoCalibrator...")
calibrator = AutoCalibrator(MODEL_PITCH)
print(f"      {calibrator.get_summary()}")

print("[2/6] Cargando detector...")
from modules.detector import detect_frame

print("[3/6] Inicializando TeamClassifier...")
from modules.team_classifier import TeamClassifier, Team
team_clf = TeamClassifier()

print("[4/6] Inicializando Tracker...")
from modules.tracker import ProfessionalTracker
tracker = ProfessionalTracker(sample_rate=SAMPLE_RATE)

print("[5/6] Inicializando EventSpotter...")
from modules.event_spotter_tdeed import EventSpotterTDEED
spotter = EventSpotterTDEED()

print("[6/6] Inicializando CameraMotion...")
from modules.camera_motion import CameraMotionEstimator
cam_motion = CameraMotionEstimator()

# ── Abrir video ────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    print(f"[ERROR] No se puede abrir el video: {VIDEO_PATH}")
    sys.exit(1)

fps        = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration   = total_frames / fps

print(f"\nVideo: {w}x{h} · {fps:.1f}fps · {total_frames} frames · {duration:.1f}s")

# ── Writer de output ───────────────────────────────────────────────────────
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (w, h))

# ── Variables de estado ────────────────────────────────────────────────────
frame_interval   = max(1, int(fps * SAMPLE_RATE))
team_clf_fitted  = False
calib_done       = False
frame_idx        = 0
processed        = 0
total_events     = 0

colors = {
    Team.A.value:       (255, 100,  50),
    Team.B.value:       ( 50, 200, 255),
    Team.REFEREE.value: ( 50, 255,  50),
    Team.UNKNOWN.value: (180, 180, 180),
}

print("\n[PROCESANDO]\n")

# ── Loop principal ─────────────────────────────────────────────────────────
MAX_FRAMES_LIMIT = 1000
while frame_idx < MAX_FRAMES_LIMIT:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # Procesar solo 1 de cada N frames
    if frame_idx % frame_interval != 0:
        out.write(frame)
        continue

    processed  += 1
    second      = frame_idx / fps
    minute      = second / 60.0

    # ── 1. Movimiento de cámara ────────────────────────────────────────────
    dx, dy = cam_motion.compute_offset(frame)

    # ── 2. Detección YOLO ─────────────────────────────────────────────────
    detections = detect_frame(frame, confidence=CONF, imgsz=IMGSZ)
    print(f"  [DEBUG] YOLO raw: {len(detections)} detecciones · imgsz={IMGSZ}", flush=True)

    players  = [d for d in detections if d.get("clase") in ("player", "goalkeeper")]
    referees = [d for d in detections if d.get("clase") == "referee"]
    balls    = [d for d in detections if d.get("clase") == "ball"]

    # ── 3. Calibración automática (una sola vez) ───────────────────────────
    if not calib_done and len(players) >= 3:
        calib_done = calibrator.calibrate(frame)
        if calib_done:
            print(f"  [CALIB] Campo calibrado en frame {frame_idx} "
                  f"({calibrator.get_summary()['keypoints_detected']} keypoints)")

    # ── 4. TeamClassifier fit (una sola vez con suficientes jugadores) ─────
    if not team_clf_fitted and len(players) >= 8:
        bboxes = [d["bbox"] for d in players]
        team_clf_fitted = team_clf.fit(frame, bboxes)
        if team_clf_fitted:
            print(f"  [TEAM] Equipos aprendidos en frame {frame_idx}: "
                  f"{team_clf.get_summary()}")

    # ── 5. Clasificar equipos ──────────────────────────────────────────────
    equipo_map = []
    all_dets   = players + referees

    for d in players:
        team = team_clf.predict(frame, d["bbox"]) if team_clf_fitted else Team.UNKNOWN
        equipo_map.append(0 if team == Team.A else 1)

    for d in referees:
        equipo_map.append(2)

    # ── 6. Tracking ───────────────────────────────────────────────────────
    tracks = tracker.update(
        detecciones   = all_dets,
        equipo_map    = equipo_map,
        minute        = minute,
        camera_offset = (dx, dy)
    )

    # ── 7. Balón ──────────────────────────────────────────────────────────
    ball_pos   = None
    ball_conf  = 0.0
    pitch_pos  = (0.0, 0.0)

    if balls:
        b         = balls[0]
        ball_pos  = (int(b["x"]), int(b["y"]))
        ball_conf = b.get("conf", 0.0)
        if calib_done:
            pitch_pos = calibrator.pixel_to_pitch(b["x"], b["y"])

    # ── 8. Eventos ────────────────────────────────────────────────────────
    events = spotter.update(
        frame_second = second,
        minute       = minute,
        tracks       = tracks,
        ball_pos     = ball_pos,
        pitch_pos    = pitch_pos,
        ball_conf    = ball_conf
    )
    total_events += len(events)
    for ev in events:
        print(f"  [EVENT] {ev.minute:.1f}' {ev.action} (track #{ev.track_id})")

    # ── 9. Dibujar anotaciones ─────────────────────────────────────────────
    vis = frame.copy()

    # Jugadores y árbitros
    for tid, track in tracks.items():
        cx, cy, tw, th = track.last_box
        x1 = int(cx - tw/2); y1 = int(cy - th/2)
        x2 = int(cx + tw/2); y2 = int(cy + th/2)

        team_val = Team.A.value if track.equipo == 0 else \
                   Team.B.value if track.equipo == 1 else Team.REFEREE.value
        color = colors.get(team_val, (180, 180, 180))

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"#{tid} {track.clase[:3].upper()}"
        cv2.putText(vis, label, (x1, max(y1-6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Balón
    if ball_pos:
        cv2.circle(vis, ball_pos, 8, (0, 255, 255), -1)
        cv2.circle(vis, ball_pos, 8, (0, 0, 0), 1)

    # Minimap
    if calib_done:
        det_list = []
        for tid, track in tracks.items():
            px, py = calibrator.pixel_to_pitch(track.last_box[0], track.last_box[1])
            det_list.append({"pitch_pos": (px, py), "team": track.equipo})

        minimap = calibrator.draw_pitch_minimap(det_list, MINIMAP_W, MINIMAP_H)
        vis[MINIMAP_Y:MINIMAP_Y+MINIMAP_H, MINIMAP_X:MINIMAP_X+MINIMAP_W] = minimap

    # Keypoints de calibración
    if calib_done:
        vis = calibrator.draw_keypoints(vis)

    # HUD
    status_calib = "CALIB OK" if calib_done else "CALIB..."
    status_team  = "TEAMS OK" if team_clf_fitted else "TEAMS..."
    cv2.putText(vis, f"{status_calib} | {status_team} | "
                     f"Tracks:{len(tracks)} | Events:{total_events} | "
                     f"{minute:.1f}'",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)

    out.write(vis)

    if processed % 20 == 0:
        pct = frame_idx / total_frames * 100
        print(f"  [{pct:5.1f}%] frame {frame_idx}/{total_frames} · "
              f"tracks={len(tracks)} · calib={'✓' if calib_done else '✗'} · "
              f"teams={'✓' if team_clf_fitted else '✗'}", flush=True)

# ── Finalizar ──────────────────────────────────────────────────────────────
cap.release()
out.release()

print("\n" + "=" * 60)
print("RESULTADO FINAL")
print("=" * 60)
print(f"Frames procesados: {processed}")
print(f"Tracks totales:    {len(tracker.get_all_tracks())}")
print(f"Eventos:           {total_events}")
print(f"Calibración:       {'OK' if calib_done else 'FALLIDA'}")
print(f"TeamClassifier:    {'OK' if team_clf_fitted else 'FALLIDO'}")
print(f"Output:            {OUTPUT_PATH}")
print("\nAbre output/demo_result.mp4 para ver el resultado visual.")
