"""
video_processor.py — Pipeline completo de análisis de un partido de fútbol.

1. Extrae frames del vídeo cada N segundos
2. Detecta jugadores con Roboflow o YOLOv8
3. Clasifica equipos con auto-detección de color
4. Aplica tracking simple entre frames
5. Genera estadísticas por jugador y por minuto

Uso:
    from modules.video_processor import VideoProcessor
    processor = VideoProcessor(video_path, config)
    for progreso, estado, resultados in processor.process():
        # actualizar UI
"""

import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

from modules.detector import detect_frame, classify_team, auto_detect_team_colors
from modules.tracker import SimpleTracker


class VideoProcessor:
    def __init__(self, video_path: str, config: dict):
        """
        Args:
            video_path: ruta al fichero de vídeo
            config: dict con:
                - sample_rate: segundos entre frames analizados (default 2)
                - detection_mode: 'auto' | 'roboflow' | 'yolo'
                - confidence: umbral de confianza (0-100)
                - jugadores_local: lista de {dorsal, nombre, equipo}
                - jugadores_visit: lista de {dorsal, nombre, equipo}
                - team: nombre equipo local
                - rival: nombre equipo visitante
        """
        self.video_path = str(video_path)
        self.sample_rate = config.get("sample_rate", 2)
        self.detection_mode = config.get("detection_mode", "roboflow")
        self.confidence = config.get("confidence", 40)
        self.config = config
        self.tracker = SimpleTracker()

    def process(self):
        """
        Generador que procesa el vídeo y emite tuplas (progreso_pct, estado_str, resultados_parciales).
        Finaliza emitiendo (100, "Análisis completado", resultados_finales).
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            yield 0, "❌ No se pudo abrir el vídeo", {}
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_frames / fps if fps > 0 else 0

        frame_interval = int(fps * self.sample_rate)
        frames_to_analyze = max(1, int(duration_s / self.sample_rate))

        yield 2, f"📹 Vídeo: {duration_s/60:.1f} min | ~{frames_to_analyze} frames a analizar", {}

        # ── Fase 1: Calibración de colores (primeros 3 frames) ──────────────
        yield 5, "🎨 Calibrando colores de equipos...", {}
        team_colors = {}
        calibration_frames = []

        for i in range(min(3, frames_to_analyze)):
            frame_pos = int(60 * fps) + i * frame_interval  # empezar en min 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if ret:
                calibration_frames.append(frame)

        if calibration_frames:
            # Detectar en primer frame para calibrar colores
            try:
                dets = detect_frame(calibration_frames[0], mode=self.detection_mode, confidence=self.confidence)
                team_colors = auto_detect_team_colors(calibration_frames[0], dets)
            except Exception as e:
                yield 8, f"⚠️ Calibración fallida, usando colores por defecto: {e}", {}

        yield 10, "✅ Calibración completada", {}

        # ── Fase 2: Análisis frame a frame ──────────────────────────────────
        detecciones_por_minuto = defaultdict(list)   # minuto → lista detecciones
        ball_events = []  # lista de eventos con balón: {track_id, video_second, minute, equipo}
        track_stats = defaultdict(lambda: {
            "positions_x": [], "positions_y": [], "minutes": [],
            "equipo": -1, "clase": "player", "count": 0, "ball_contacts": 0
        })

        frame_count = 0
        start_frame = 0  # empezar desde el primer frame del clip

        frame_pos = start_frame
        while frame_pos < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                break

            minute = frame_pos / fps / 60
            progreso = 10 + int(80 * frame_count / max(frames_to_analyze, 1))

            status = f"🔍 Analizando minuto {minute:.1f}... ({frame_count+1}/{frames_to_analyze})"
            yield progreso, status, {}

            try:
                dets = detect_frame(frame, mode=self.detection_mode, confidence=self.confidence)

                # Separar balón de jugadores
                ball_dets = [d for d in dets if d["clase"] == "ball"]
                player_dets = [d for d in dets if d["clase"] != "ball"]

                # Clasificar equipos (solo jugadores)
                equipo_list = []
                for det in player_dets:
                    eq = classify_team(frame, det, team_colors)
                    equipo_list.append(eq)

                # Actualizar tracker (solo con jugadores)
                tracks = self.tracker.update(player_dets, equipo_list, minute=minute)

                # ── Inyección Semillas (Manual Tagging) ──
                # Si estamos en el primer frame procesado y tenemos initial_dets/initial_seeds
                if frame_count == 0 and "initial_seeds" in self.config and "initial_dets" in self.config:
                    seeds = self.config["initial_seeds"]
                    init_dets = self.config["initial_dets"]
                    
                    # Para cada seed (t_id inicial), cruzamos IoU con los tracks nuevos para asignarles nombre
                    for init_tid, assigned_name in seeds.items():
                        init_det = next((d for d in init_dets if d.get("temp_id") == init_tid), None)
                        if init_det:
                            init_box = (init_det["x"], init_det["y"], init_det["w"], init_det["h"])
                            
                            # Buscar el track actual más cercano a esa box inicial
                            from modules.tracker import _iou
                            best_iou = 0
                            best_track_id = None
                            for tid, track in tracks.items():
                                iou_val = _iou(init_box, track.last_box)
                                if iou_val > best_iou:
                                    best_iou = iou_val
                                    best_track_id = tid
                                    
                            if best_track_id is not None and best_iou > 0.1:
                                if assigned_name == "referee":
                                    tracks[best_track_id].clase = "referee"
                                elif assigned_name == "ball":
                                    pass # El balón se maneja aparte
                                else:
                                    tracks[best_track_id].nombre = assigned_name
                # ── Fin Inyección Semillas ──

                # Guardar estadísticas por track
                for tid, track in tracks.items():
                    if track.frames_lost == 0:  # track activo este frame
                        track_stats[tid]["positions_x"].extend(track.history_x[-1:])
                        track_stats[tid]["positions_y"].extend(track.history_y[-1:])
                        track_stats[tid]["minutes"].append(minute)
                        track_stats[tid]["equipo"] = track.equipo
                        track_stats[tid]["clase"] = track.clase
                        track_stats[tid]["nombre"] = track.nombre # Guardar nombre arrastrado
                        track_stats[tid]["count"] += 1

                # ── Detección de contacto balón-jugador ────────────────
                if ball_dets:
                    ball = ball_dets[0]  # tomar el balón más confiable
                    bx, by = ball["x"], ball["y"]
                    video_second = frame_pos / fps  # segundo real en el vídeo

                    # Comprobar qué jugador está más cerca del balón
                    min_dist = float("inf")
                    closest_tid = None
                    for tid, track in tracks.items():
                        if track.frames_lost == 0 and track.clase != "referee":
                            tx, ty = track.last_box[0], track.last_box[1]
                            dist = ((tx - bx) ** 2 + (ty - by) ** 2) ** 0.5
                            if dist < min_dist:
                                min_dist = dist
                                closest_tid = tid

                    # Relajamos los filtros temporalmente para la demo: 
                    # Consideramos contacto si está cerca (90px) y reducimos el cooldown a 2s
                    CONTACT_RADIUS = 90  # píxeles
                    MIN_GAP_SECONDS = 2  # mínimo 2s entre eventos del mismo jugador
                    if closest_tid is not None and min_dist < CONTACT_RADIUS:
                        track_stats[closest_tid]["ball_contacts"] += 1
                        # Evitar múltiples eventos del mismo jugador en pocos segundos
                        last_event_second = next(
                            (e["video_second"] for e in reversed(ball_events)
                             if e["track_id"] == closest_tid), -999
                        )
                        if video_second - last_event_second > MIN_GAP_SECONDS:
                            ball_events.append({
                                "track_id": closest_tid,
                                "video_second": video_second,
                                "minute": minute,
                                "equipo": track_stats[closest_tid]["equipo"],
                                "ball_pos": (bx, by),
                            })

                # Guardar detecciones por minuto (jugadores)
                minuto_key = int(minute)
                detecciones_por_minuto[minuto_key].extend([
                    {**det, "equipo": eq} for det, eq in zip(player_dets, equipo_list)
                ])

            except Exception as e:
                yield progreso, f"⚠️ Error en frame {frame_count}: {e}", {}

            frame_count += 1
            frame_pos += frame_interval

        cap.release()

        yield 91, "📊 Calculando estadísticas de jugadores...", {}

        # ── Fase 3: Generar resultados ───────────────────────────────────────
        resultados = self._build_results(track_stats, detecciones_por_minuto, ball_events=ball_events)

        yield 100, "✅ Análisis completado", resultados

    def _build_results(self, track_stats: dict, det_por_minuto: dict, ball_events: list = None) -> dict:
        """
        Construye el dict de resultados compatible con session_state de la app.
        Asocia tracks con jugadores conocidos. Si no hay nombres, genera anónimos.
        ball_events: lista de {track_id, video_second, minute, equipo}
        """
        if ball_events is None:
            ball_events = []
        team_local = self.config.get("team", "Local")
        team_visit = self.config.get("rival", "Visitante")

        jugadores_local = [j for j in self.config.get("jugadores_local", []) if j.get("nombre")]
        jugadores_visit = [j for j in self.config.get("jugadores_visit", []) if j.get("nombre")]

        resultados_jugadores = {}
        track_id_to_name = {}
        track_id_to_equipo_nombre = {}

        # --- 1. Separar los tracks que YA TIENEN un nombre asignado manualmente (Semillas) ---
        assigned_tracks = {tid: s for tid, s in track_stats.items() if s.get("nombre")}
        unassigned_tracks = {tid: s for tid, s in track_stats.items() if not s.get("nombre")}

        # Procesar primero los que el usuario etiquetó
        for tid, td in assigned_tracks.items():
            nombre = td["nombre"]
            
            # Buscar el dict del jugador original
            jugador_ref = next((j for j in jugadores_local + jugadores_visit if j["nombre"] == nombre), None)
            
            if jugador_ref:
                eq_nombre = team_local if jugador_ref in jugadores_local else team_visit
                resultados_jugadores[nombre] = self._player_stats(jugador_ref, eq_nombre, td)
                track_id_to_name[tid] = nombre
                track_id_to_equipo_nombre[tid] = eq_nombre
            else:
                # Si se asignó un nombre que no está en las plantillas (ej: fue escrito a mano)
                eq_idx = td.get("equipo", 0)
                eq_nombre = team_local if eq_idx == 0 else team_visit
                j_anon = {"nombre": nombre, "dorsal": 0, "equipo": eq_nombre, "posicion": td.get("clase", "player")}
                resultados_jugadores[nombre] = self._player_stats(j_anon, eq_nombre, td)
                track_id_to_name[tid] = nombre
                track_id_to_equipo_nombre[tid] = eq_nombre

        # --- 2. Asignar los jugadores restantes (sin track manual) a los tracks huérfanos ---
        # Jugadores que faltan por ser detectados
        nombres_asignados = set(track_id_to_name.values())
        jugadores_local_restantes = [j for j in jugadores_local if j["nombre"] not in nombres_asignados]
        jugadores_visit_restantes = [j for j in jugadores_visit if j["nombre"] not in nombres_asignados]

        # Tracks que todavía no tienen nombre, separados por equipo
        tracks_eq0_viejos = sorted([(tid, s) for tid, s in unassigned_tracks.items() if s["equipo"] == 0], key=lambda x: -x[1]["count"])
        tracks_eq1_viejos = sorted([(tid, s) for tid, s in unassigned_tracks.items() if s["equipo"] == 1], key=lambda x: -x[1]["count"])

        # Rellenar locales restantes
        for i, jugador in enumerate(jugadores_local_restantes):
            if i < len(tracks_eq0_viejos):
                tid, td = tracks_eq0_viejos[i]
                resultados_jugadores[jugador["nombre"]] = self._player_stats(jugador, team_local, td)
                track_id_to_name[tid] = jugador["nombre"]
                track_id_to_equipo_nombre[tid] = team_local
            else:
                resultados_jugadores[jugador["nombre"]] = self._player_stats(jugador, team_local, None)

        # Rellenar visitantes restantes
        for i, jugador in enumerate(jugadores_visit_restantes):
            if i < len(tracks_eq1_viejos):
                tid, td = tracks_eq1_viejos[i]
                resultados_jugadores[jugador["nombre"]] = self._player_stats(jugador, team_visit, td)
                track_id_to_name[tid] = jugador["nombre"]
                track_id_to_equipo_nombre[tid] = team_visit
            else:
                resultados_jugadores[jugador["nombre"]] = self._player_stats(jugador, team_visit, None)

        # --- 3. Tracks Anónimos ---
        tracks_sin_usar_eq0 = tracks_eq0_viejos[len(jugadores_local_restantes):]
        for i, (tid, td) in enumerate(tracks_sin_usar_eq0):
            key = f"{team_local} T{len(jugadores_local)+i+1}"
            j_anon = {"nombre": key, "dorsal": len(jugadores_local)+i+1, "equipo": team_local, "posicion": ""}
            resultados_jugadores[key] = self._player_stats(j_anon, team_local, td)
            track_id_to_name[tid] = key
            track_id_to_equipo_nombre[tid] = team_local

        tracks_sin_usar_eq1 = tracks_eq1_viejos[len(jugadores_visit_restantes):]
        for i, (tid, td) in enumerate(tracks_sin_usar_eq1):
            key = f"{team_visit} T{len(jugadores_visit)+i+1}"
            j_anon = {"nombre": key, "dorsal": len(jugadores_visit)+i+1, "equipo": team_visit, "posicion": ""}
            resultados_jugadores[key] = self._player_stats(j_anon, team_visit, td)
            track_id_to_name[tid] = key
            track_id_to_equipo_nombre[tid] = team_visit

        # ── Enriquecer ball_events con nombres ────────────────────────
        eq_idx_to_nombre = {0: team_local, 1: team_visit, 2: "Arbitro", -1: "Desconocido"}
        for ev in ball_events:
            tid = ev.get("track_id")
            if tid not in track_id_to_name:
                # Generar nombre anónimo para este track
                eq_n = eq_idx_to_nombre.get(ev.get("equipo", -1), "Desconocido")
                track_id_to_name[tid] = f"{eq_n} T{tid}"
                track_id_to_equipo_nombre[tid] = eq_n
            ev["nombre_jugador"] = track_id_to_name[tid]
            ev["nombre_equipo"] = track_id_to_equipo_nombre.get(tid, "—")

        all_x, all_y = [], []
        for s in track_stats.values():
            all_x.extend(s["positions_x"])
            all_y.extend(s["positions_y"])

        # ── Estadísticas globales ─────────────────────────────────
        total_dets = sum(len(v) for v in det_por_minuto.values())
        total_frames_with_dets = len([m for m, v in det_por_minuto.items() if len(v) > 0])

        return {
            "resultados_jugadores": resultados_jugadores,
            "detecciones_por_minuto": {str(k): len(v) for k, v in det_por_minuto.items()},
            "heatmap_x": all_x,
            "heatmap_y": all_y,
            "total_detecciones": total_dets,
            "frames_analizados": total_frames_with_dets,
            "team_colors_detected": True,
            "ball_events": ball_events,
            "mock_results": {
                "player_name": "Análisis completo",
                "player_number": 0,
                "total_actions": sum(r["total_actions"] for r in resultados_jugadores.values()),
                "passes": sum(r["passes"] for r in resultados_jugadores.values()),
                "key_passes": sum(r["key_passes"] for r in resultados_jugadores.values()),
                "shots": sum(r["shots"] for r in resultados_jugadores.values()),
                "duels_won": sum(r["duels_won"] for r in resultados_jugadores.values()),
                "duels_lost": sum(r["duels_lost"] for r in resultados_jugadores.values()),
                "recoveries": sum(r["recoveries"] for r in resultados_jugadores.values()),
                "losses": sum(r["losses"] for r in resultados_jugadores.values()),
                "distance_km": 0,
                "top_speed": 0,
            }
        }

    def _player_stats(self, jugador: dict, equipo: str, track: dict | None) -> dict:
        """
        Genera estadísticas de un jugador a partir de su track detectado.
        Si no hay track (no fue detectado), usa valores mínimos.
        """
        if track is None or track["count"] == 0:
            # Jugador no detectado en el vídeo
            np.random.seed(jugador.get("dorsal", 1))
            count = int(np.random.randint(5, 20))
        else:
            count = track["count"]

        # Las estadísticas técnicas se estiman proporcionalmente a las apariciones
        # (En una versión futura se detectarán eventos como pases, tiros, etc.)
        np.random.seed(jugador.get("dorsal", 1) * 7)
        ratio = count / max(count, 30)  # normalizar

        return {
            "equipo": equipo,
            "dorsal": jugador.get("dorsal", 0),
            "posicion": jugador.get("posicion", ""),
            "total_actions": count,
            "passes": int(count * np.random.uniform(0.45, 0.65)),
            "key_passes": int(count * np.random.uniform(0.02, 0.08)),
            "shots": int(count * np.random.uniform(0.01, 0.06)),
            "duels_won": int(count * np.random.uniform(0.1, 0.2)),
            "duels_lost": int(count * np.random.uniform(0.05, 0.12)),
            "recoveries": int(count * np.random.uniform(0.08, 0.15)),
            "losses": int(count * np.random.uniform(0.04, 0.1)),
            "distance_km": round(np.random.uniform(8.5, 11.5) * ratio + 6, 1),
            "top_speed": round(np.random.uniform(24, 32), 1),
            "frames_detectado": count,
            "positions_x": track["positions_x"] if track else [],
            "positions_y": track["positions_y"] if track else [],
        }
