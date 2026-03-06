"""
action_clips.py — Clips de vídeo reales cortados con FFmpeg.
Estilo Wyscout: tarjetas de evento oscuras, teal accent, diseño limpio.
"""

import streamlit as st
import subprocess
import os
import tempfile
import cv2
import numpy as np
from pathlib import Path

try:
    from modules.detector import detect_frame, classify_team, auto_detect_team_colors, yolo_model_available
except ImportError:
    pass

ACTION_COLORS = {
    "Pase": "#00d4aa",
    "Pase progresivo": "#00ff99",
    "Pase clave": "#FFD700",
    "Duelo ganado": "#00d4aa",
    "Duelo perdido": "#ff4d6d",
    "Recuperación": "#00d4aa",
    "Pérdida": "#ff6b35",
    "Tiro": "#ff4d6d",
    "Conducción": "#a78bfa",
    "Córner": "#FFD700",
    "Falta": "#8899aa",
    "Acción con balón": "#00d4aa",
}

CLIP_ANTES = 3
CLIP_DESPUES = 7
# Carpeta persistente — NO se borra al reiniciar el PC
CLIPS_DIR = Path(__file__).parent.parent / "output" / "clips"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)


def _get_ffmpeg_path() -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env")
    except Exception:
        pass
    return os.getenv("FFMPEG_PATH", "C:\\ffmpeg\\bin\\ffmpeg.exe")


def _cortar_clip(video_path: str, video_second: float, clip_key: str) -> str | None:
    ffmpeg = _get_ffmpeg_path()
    if not Path(ffmpeg).exists():
        return None

    start = max(0, video_second - CLIP_ANTES)
    duracion = CLIP_ANTES + CLIP_DESPUES
    out_path = CLIPS_DIR / f"clip_{clip_key}.mp4"

    if out_path.exists():
        return str(out_path)

    cmd = [
        ffmpeg, "-y",
        "-ss", f"{start:.2f}",
        "-i", video_path,
        "-t", str(duracion),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-c:a", "aac", "-movflags", "+faststart",
        str(out_path)
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=30)
        if out_path.exists() and out_path.stat().st_size > 10_000:
            return str(out_path)
    except Exception:
        pass
    return None


def _pintar_clip(in_path: str, out_path: str, team_colors: dict) -> bool:
    """Procesa el clip cortado frame a frame para pintar los polígonos de segmentación"""
    if not yolo_model_available():
        return False # Si no hay modelo entrenado, no pintamos nada.
        
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Usar siempre yolo local para obtener poligonos (masks)
        detecciones = detect_frame(frame, mode="yolo")
        
        # Crear un overlay para dibujar formas semitransparentes
        overlay = frame.copy()

        for det in detecciones:
            mask = det.get("mask")
            clase = det.get("clase")
            
            # Decidir el color
            color_bgr = (200, 200, 200) # por defecto gris claro
            
            if clase == "ball":
                color_bgr = (0, 255, 255) # Amarillo para balón
            elif clase == "referee":
                color_bgr = (0, 0, 255) # Rojo para árbitro
            elif clase in ("player", "goalkeeper"):
                eq_idx = classify_team(frame, det, team_colors)
                if eq_idx == 0:
                    color_bgr = (170, 212, 0) # team_local (teal BGR)
                elif eq_idx == 1:
                    color_bgr = (255, 159, 77) # team_visit (naranja BGR)

            # Dibujar el polígono si existe, si no, fallback a la caja
            if mask is not None and len(mask) > 0:
                # fillPoly pinta el area interior
                cv2.fillPoly(overlay, [mask], color_bgr)
                # polylines dibuja el borde más fuerte
                cv2.polylines(frame, [mask], isClosed=True, color=color_bgr, thickness=2)
            else:
                x, y, w_box, h_box = det["x"], det["y"], det["w"], det["h"]
                cv2.rectangle(overlay, (x - w_box//2, y - h_box//2), (x + w_box//2, y + h_box//2), color_bgr, -1)

        # Aplicar transparencia (70% frame original, 30% overlay de color)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        out.write(frame)

    cap.release()
    out.release()
    return True


def _cortar_y_pintar_clip(video_path: str, video_second: float, clip_key: str, team_colors: dict, skip_paint: bool = False) -> str | None:
    """Corta el clip original y luego lo procesa frame a frame con IA"""
    base_clip = _cortar_clip(video_path, video_second, clip_key)
    if not base_clip:
        return None
        
    if skip_paint or not yolo_model_available():
        return base_clip # Skip paint si se pide o si no hay IA
        
    painted_path = str(CLIPS_DIR / f"painted_{clip_key}.mp4")
    if Path(painted_path).exists():
        return painted_path
        
    exito = _pintar_clip(base_clip, painted_path, team_colors)
    if exito:
        return painted_path
    return base_clip


def render():
    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;">
        <div>
            <h2 style="margin:0;font-size:20px;font-weight:700;color:#fff;">Clips de Acción</h2>
            <p style="margin:2px 0 0;font-size:13px;color:#5a6a7e;">Fragmentos de vídeo de los momentos clave del partido</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.get("analysis_done"):
        st.markdown("""
        <div style="background:rgba(17, 24, 39, 0.4); backdrop-filter:blur(15px); border:1px solid rgba(255, 77, 109, 0.2); border-radius:16px; padding:48px; text-align:center; box-shadow:0 10px 40px rgba(0,0,0,0.4);">
            <div style="font-size:48px; margin-bottom:16px; filter: drop-shadow(0 0 10px rgba(255, 77, 109, 0.3));">✂️</div>
            <div style="font-size:18px; font-weight:700; color:#fff; margin-bottom:8px; letter-spacing:0.5px;">SIN DATOS DE ANÁLISIS</div>
            <div style="font-size:14px; color:#8899aa;">Primero completa el proceso en <strong>Nuevo Análisis</strong> para generar fragmentos de vídeo.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    ball_events = st.session_state.get("ball_events", [])
    resultados_jug = st.session_state.get("resultados_jugadores", {})
    video_path = st.session_state.get("video_path", "")
    config = st.session_state.get("analysis_config", {})
    team_local = config.get("team", "Local")
    team_visit = config.get("rival", "Visitante")

    if not ball_events:
        st.markdown("""
        <div style="background:rgba(255, 107, 53, 0.1); backdrop-filter:blur(10px); border:1px solid rgba(255, 107, 53, 0.3); border-radius:16px; padding:32px;">
            <div style="font-size:16px; font-weight:700; color:#fff; margin-bottom:12px; display:flex; align-items:center; gap:10px;">
                <span style="font-size:20px;">⚠️</span> NO SE DETECTARON ACCIONES
            </div>
            <div style="font-size:14px; color:#8899aa; line-height:1.7;">
                El motor no ha encontrado interacciones claras con el balón en este vídeo.<br>
                <br>
                <strong style="color:#00d4aa; text-transform:uppercase; font-size:11px; letter-spacing:1px;">Recomendación:</strong><br>
                Ajusta el <strong>intervalo de análisis</strong> a un valor menor (ej. 0.5s) en el Paso 2 del configurador para una detección más sensible.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Resumen ────────────────────────────────────────────────────────────────
    st.markdown('<div class="ws-section-header">Resumen</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Acciones detectadas", len(ball_events))
    c2.metric("Duración por clip", f"{CLIP_ANTES + CLIP_DESPUES}s")
    c3.metric("Vídeo fuente", Path(video_path).name[:30] + "…" if video_path and len(Path(video_path).name) > 30 else (Path(video_path).name if video_path else "—"))

    ffmpeg_ok = Path(_get_ffmpeg_path()).exists()
    if not ffmpeg_ok:
        st.markdown("""
        <div style="background:#1e1220;border:1px solid #ff4d6d44;border-radius:8px;padding:10px 16px;margin-top:12px;font-size:12px;color:#ff4d6d;">
            ⚠️ FFmpeg no encontrado. Los clips no se pueden generar. Configura la ruta en ⚙️ Configuración.
        </div>
        """, unsafe_allow_html=True)

    # ── Filtros ────────────────────────────────────────────────────────────────
    st.markdown('<div class="ws-section-header">Filtros</div>', unsafe_allow_html=True)
    jugadores_en_eventos = sorted(set(
        e.get("nombre_jugador", f"T{e['track_id']}") for e in ball_events
    ))
    col_f1, col_f2 = st.columns(2)
    equipo_filtro = col_f1.multiselect("Equipo", [team_local, team_visit])
    jugador_filtro = col_f2.multiselect("Jugador", jugadores_en_eventos)
    
    st.markdown('<div class="ws-section-header">Opciones de Clip</div>', unsafe_allow_html=True)
    c_opt1, c_opt2 = st.columns(2)
    skip_paint = c_opt1.checkbox("⚡ Ver vídeo original (Instantáneo)", value=True, help="Si se desactiva, la IA dibujará sobre el vídeo (tarda unos segundos)")

    eventos_filtrados = ball_events.copy()
    if jugador_filtro:
        eventos_filtrados = [e for e in eventos_filtrados
                             if e.get("nombre_jugador", f"T{e['track_id']}") in jugador_filtro]

    st.markdown(f"""
    <div style="font-size:12px;color:#5a6a7e;margin-bottom:16px;">
        Mostrando <strong style="color:#fff;">{len(eventos_filtrados)}</strong> de {len(ball_events)} acciones
    </div>
    """, unsafe_allow_html=True)

    # ── Lista de eventos ───────────────────────────────────────────────────────
    st.markdown('<div class="ws-section-header">Acciones</div>', unsafe_allow_html=True)

    for i, evento in enumerate(eventos_filtrados):
        nombre = evento.get("nombre_jugador", f"Jugador T{evento['track_id']}")
        equipo = evento.get("nombre_equipo", "—")
        minuto = evento.get("minute", 0)
        seg = evento.get("video_second", 0)
        clip_key = f"{evento['track_id']}_{int(seg)}"
        eq_color = "#00d4aa" if equipo == team_local else "#4d9fff"

        with st.expander(
            f"⚽  Min {minuto:.1f}'  ·  {nombre}  ·  {equipo}",
            expanded=False
        ):
            # Cabecera del clip con estilo futurista
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:20px;margin-bottom:20px;padding:20px;
                        background:rgba(13, 18, 32, 0.6); backdrop-filter:blur(10px);
                        border-radius:12px; border:1px solid rgba(255,255,255,0.05);
                        border-left:5px solid {eq_color}; shadow: 0 4px 20px rgba(0,0,0,0.2);">
                <div style="font-size:28px;font-weight:900;color:{eq_color};min-width:60px; text-shadow:0 0 10px {eq_color}55;">
                    {minuto:.0f}'
                </div>
                <div style="flex:1;">
                    <div style="font-size:16px;font-weight:800;color:#fff;margin-bottom:2px;">{nombre}</div>
                    <div style="font-size:12px;color:#8899aa;font-family:monospace;">STAMP: {seg:.1f}s · {equipo}</div>
                </div>
                <div style="background:{eq_color}22; border:1px solid {eq_color}44; color:{eq_color};
                            padding:6px 14px; border-radius:20px; font-size:10px; font-weight:800; text-transform:uppercase; letter-spacing:1px;">
                    EVENTO
                </div>
            </div>
            """, unsafe_allow_html=True)

            if not video_path or not Path(video_path).exists():
                st.markdown("""
                <div style="background:#1e1220;border:1px solid #ff4d6d33;border-radius:8px;
                            padding:10px 14px;font-size:12px;color:#ff4d6d;">
                    ⚠️ El vídeo original no está disponible en esta sesión.
                </div>
                """, unsafe_allow_html=True)
                continue

            if ffmpeg_ok:
                if st.button(f"▶  Ver clip  ({CLIP_ANTES + CLIP_DESPUES}s)", key=f"btn_clip_{i}",
                             type="primary"):
                    with st.spinner("Generando clip (IA dibujando jugadores...)"):
                        # Obtener paleta de colores si existe en la sesión
                        team_colors = st.session_state.get("team_colors", {})
                        clip_path = _cortar_y_pintar_clip(video_path, seg, clip_key, team_colors, skip_paint=skip_paint)

                    if clip_path and os.path.exists(clip_path):
                        with open(clip_path, "rb") as f:
                            video_bytes = f.read()
                        st.video(video_bytes)
                        st.caption(f"📍 {seg:.1f}s en el vídeo · ±{CLIP_ANTES}s antes / {CLIP_DESPUES}s después")
                    else:
                        st.error("❌ No se pudo generar el clip o el archivo no existe.")
            else:
                st.markdown("""
                <div style="font-size:12px;color:#5a6a7e;">
                    FFmpeg no disponible. Configura la ruta en ⚙️ Configuración para generar clips.
                </div>
                """, unsafe_allow_html=True)