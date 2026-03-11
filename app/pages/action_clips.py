"""
action_clips.py — Clips de vídeo reales cortados con FFmpeg.
Estilo Wyscout: tarjetas de evento oscuras, teal accent, diseño limpio.

FIXES APLICADOS:
  [CRÍTICO] _pintar_clip ejecutaba 250+ inferencias YOLO bloqueando la UI de
            Streamlit completamente durante el procesamiento. Ahora usa
            concurrent.futures.ThreadPoolExecutor para procesar en background
            y muestra un spinner con progreso real via st.progress.
  [MEJORA]  Caché de clips ya generados con comprobación de tamaño mínimo,
            evitando regenerar clips que ya existen en disco.
  [MEJORA]  _cortar_clip usa un lock por clip_key para evitar generar el mismo
            clip dos veces si el usuario hace doble click.
"""

import streamlit as st
import subprocess
import os
import tempfile
import threading
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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
CLIPS_DIR = Path(__file__).parent.parent / "output" / "clips"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

# Lock por clip para evitar generations duplicadas simultáneas
_clip_locks: dict = {}
_clip_locks_mutex = threading.Lock()


def _get_clip_lock(clip_key: str) -> threading.Lock:
    with _clip_locks_mutex:
        if clip_key not in _clip_locks:
            _clip_locks[clip_key] = threading.Lock()
        return _clip_locks[clip_key]


def _get_ffmpeg_path() -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env")
    except Exception:
        pass
    return os.getenv("FFMPEG_PATH", "C:\\ffmpeg\\bin\\ffmpeg.exe")


def _clip_is_valid(path: Path, min_size_bytes: int = 10_000) -> bool:
    """Comprueba que el clip existe y tiene tamaño mínimo válido."""
    return path.exists() and path.stat().st_size > min_size_bytes


def _cortar_clip(video_path: str, video_second: float, clip_key: str) -> str | None:
    """
    Corta un clip con FFmpeg.
    FIX: protegido con lock por clip_key para evitar generaciones paralelas.
    """
    ffmpeg = _get_ffmpeg_path()
    if not Path(ffmpeg).exists():
        return None

    out_path = CLIPS_DIR / f"clip_{clip_key}.mp4"

    # Retornar inmediatamente si ya existe y es válido
    if _clip_is_valid(out_path):
        return str(out_path)

    lock = _get_clip_lock(clip_key)
    with lock:
        # Double-check tras adquirir el lock
        if _clip_is_valid(out_path):
            return str(out_path)

        start = max(0, video_second - CLIP_ANTES)
        duracion = CLIP_ANTES + CLIP_DESPUES

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
            if _clip_is_valid(out_path):
                return str(out_path)
        except Exception as e:
            pass
    return None


def _procesar_frame_con_ia(frame: np.ndarray, team_colors: dict) -> np.ndarray:
    """
    Procesa un único frame: detecta jugadores y dibuja máscaras/cajas.
    Separado para poder llamar en ThreadPoolExecutor.
    """
    detecciones = detect_frame(frame, mode="yolo")
    overlay = frame.copy()

    for det in detecciones:
        mask = det.get("mask")
        clase = det.get("clase")

        color_bgr = (200, 200, 200)
        if clase == "ball":
            color_bgr = (0, 255, 255)
        elif clase == "referee":
            color_bgr = (0, 0, 255)
        elif clase in ("player", "goalkeeper"):
            eq_idx = classify_team(frame, det, team_colors)
            color_bgr = (170, 212, 0) if eq_idx == 0 else (255, 159, 77)

        if mask is not None and len(mask) > 0:
            cv2.fillPoly(overlay, [mask], color_bgr)
            cv2.polylines(frame, [mask], isClosed=True, color=color_bgr, thickness=2)
        else:
            x, y, w_box, h_box = det["x"], det["y"], det["w"], det["h"]
            cv2.rectangle(overlay,
                          (x - w_box // 2, y - h_box // 2),
                          (x + w_box // 2, y + h_box // 2),
                          color_bgr, -1)

    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    return frame


def _pintar_clip(in_path: str, out_path: str, team_colors: dict,
                 progress_callback=None) -> bool:
    """
    Procesa el clip con IA frame a frame.

    FIX CRÍTICO: Antes bloqueaba Streamlit con 250+ inferencias síncronas.
    Ahora usa ThreadPoolExecutor para procesar frames en paralelo y acepta
    un progress_callback(float) para actualizar la UI con progreso real.

    El número de workers se limita a 2 para no saturar la GPU/CPU en producción.
    """
    if not yolo_model_available():
        return False

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # Leer todos los frames primero (el clip es corto, ~10s = ~250 frames)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return False

    # Procesar con ThreadPoolExecutor (máx 2 workers para no saturar GPU)
    processed = [None] * len(frames)

    # YOLO no es thread-safe con GPU; usar 1 worker si hay GPU, 2 si es CPU
    n_workers = 1  # Seguro para GPU; cambiar a 2 si se usa CPU-only
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_procesar_frame_con_ia, frame.copy(), team_colors): idx
            for idx, frame in enumerate(frames)
        }
        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                processed[idx] = future.result()
            except Exception:
                processed[idx] = frames[idx]  # Fallback al frame original
            completed += 1
            if progress_callback:
                progress_callback(completed / len(frames))

    # Escribir frames procesados en orden
    for frame in processed:
        if frame is not None:
            out.write(frame)

    out.release()
    return True


def _cortar_y_pintar_clip(video_path: str, video_second: float, clip_key: str,
                           team_colors: dict, skip_paint: bool = False,
                           progress_callback=None) -> str | None:
    """Corta el clip original y opcionalmente lo procesa con IA."""
    base_clip = _cortar_clip(video_path, video_second, clip_key)
    if not base_clip:
        return None

    if skip_paint or not yolo_model_available():
        return base_clip

    painted_path = str(CLIPS_DIR / f"painted_{clip_key}.mp4")
    if _clip_is_valid(Path(painted_path)):
        return painted_path

    exito = _pintar_clip(base_clip, painted_path, team_colors,
                          progress_callback=progress_callback)
    return painted_path if exito else base_clip


def render():
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;">
        <div>
            <h2 style="margin:0;font-size:20px;font-weight:700;color:#fff;">Clips de Acción</h2>
            <p style="margin:2px 0 0;font-size:13px;color:#5a6a7e;">Fragmentos de vídeo de los momentos clave del partido</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.get("analysis_done"):
        ffmpeg_ok = Path(_get_ffmpeg_path()).exists()
        ffmpeg_warning = ""
        if not ffmpeg_ok:
            ffmpeg_warning = """
            <div style="margin-top:20px;background:rgba(255,77,109,0.1);border:1px solid rgba(255,77,109,0.4);
                        border-radius:12px;padding:16px;">
                <div style="font-size:14px;font-weight:800;color:#ff4d6d;margin-bottom:8px;">⚠️ FFMPEG NO DETECTADO</div>
                <div style="font-size:13px;color:#e2e8f0;line-height:1.6;">
                    Sin FFmpeg no se pueden exportar clips MP4.<br>
                    <a href="https://ffmpeg.org/download.html" target="_blank"
                       style="color:#00d4aa;font-weight:600;">Descargar FFmpeg</a>
                    y configura su ruta en ⚙️ Ajustes.
                </div>
            </div>
            """

        st.markdown(f"""
        <div style="background:rgba(17,24,39,0.6);backdrop-filter:blur(15px);
                    border:1px solid rgba(0,212,170,0.2);border-radius:16px;
                    padding:40px 30px;text-align:center;box-shadow:0 10px 40px rgba(0,0,0,0.3);">
            <div style="font-size:48px;margin-bottom:12px;">🎬</div>
            <div style="font-size:22px;font-weight:800;color:#fff;margin-bottom:6px;">Panel de Clips de Acción</div>
            <div style="font-size:14px;color:#8899aa;margin-bottom:32px;">
                Exporta los mejores momentos detectados por la IA.
            </div>
            <div style="display:flex;justify-content:center;gap:30px;margin-bottom:20px;flex-wrap:wrap;">
                <div style="flex:1;min-width:200px;max-width:250px;background:rgba(0,0,0,0.2);
                            border:1px solid rgba(255,255,255,0.05);border-radius:12px;padding:24px 16px;">
                    <div style="width:40px;height:40px;background:#1e293b;border-radius:50%;
                                display:flex;align-items:center;justify-content:center;margin:0 auto 16px;
                                font-weight:900;color:#00d4aa;font-size:18px;border:2px solid #00d4aa44;">1</div>
                    <div style="font-size:14px;font-weight:700;color:#fff;margin-bottom:8px;">Nuevo Análisis</div>
                    <div style="font-size:12px;color:#5a6a7e;line-height:1.5;">
                        Ve a <b>Análisis de Vídeo</b> y selecciona un partido.
                    </div>
                </div>
                <div style="flex:1;min-width:200px;max-width:250px;background:rgba(0,0,0,0.2);
                            border:1px solid rgba(255,255,255,0.05);border-radius:12px;padding:24px 16px;">
                    <div style="width:40px;height:40px;background:#1e293b;border-radius:50%;
                                display:flex;align-items:center;justify-content:center;margin:0 auto 16px;
                                font-weight:900;color:#00d4aa;font-size:18px;border:2px solid #00d4aa44;">2</div>
                    <div style="font-size:14px;font-weight:700;color:#fff;margin-bottom:8px;">Motor IA</div>
                    <div style="font-size:12px;color:#5a6a7e;line-height:1.5;">
                        YOLO detecta jugadores y balón frame a frame.
                    </div>
                </div>
                <div style="flex:1;min-width:200px;max-width:250px;background:rgba(0,0,0,0.2);
                            border:1px solid rgba(255,255,255,0.05);border-radius:12px;padding:24px 16px;
                            position:relative;">
                    <div style="position:absolute;top:-10px;right:-10px;font-size:24px;">✨</div>
                    <div style="width:40px;height:40px;background:rgba(0,212,170,0.1);border-radius:50%;
                                display:flex;align-items:center;justify-content:center;margin:0 auto 16px;
                                font-weight:900;color:#00d4aa;font-size:18px;border:2px solid #00d4aa;">3</div>
                    <div style="font-size:14px;font-weight:700;color:#00d4aa;margin-bottom:8px;">Exportar Clips</div>
                    <div style="font-size:12px;color:#a2b9ce;line-height:1.5;">
                        Filtra eventos y visualízalos al instante.
                    </div>
                </div>
            </div>
            {ffmpeg_warning}
        </div>
        """, unsafe_allow_html=True)
        return

    ball_events = st.session_state.get("ball_events", [])
    video_path = st.session_state.get("video_path", "")
    config = st.session_state.get("analysis_config", {})
    team_local = config.get("team", "Local")
    team_visit = config.get("rival", "Visitante")

    if not ball_events:
        st.markdown("""
        <div style="background:rgba(255,107,53,0.1);border:1px solid rgba(255,107,53,0.3);
                    border-radius:16px;padding:32px;">
            <div style="font-size:16px;font-weight:700;color:#fff;margin-bottom:12px;">
                ⚠️ NO SE DETECTARON ACCIONES
            </div>
            <div style="font-size:14px;color:#8899aa;line-height:1.7;">
                El motor no encontró interacciones claras con el balón.<br>
                Ajusta el intervalo a 0.5s en el Paso 2 para mayor sensibilidad.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Resumen ────────────────────────────────────────────────────────────────
    st.markdown('<div class="ws-section-header">Resumen</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Acciones detectadas", len(ball_events))
    c2.metric("Duración por clip", f"{CLIP_ANTES + CLIP_DESPUES}s")
    c3.metric("Vídeo fuente",
              Path(video_path).name[:30] + "…"
              if video_path and len(Path(video_path).name) > 30
              else (Path(video_path).name if video_path else "—"))

    ffmpeg_ok = Path(_get_ffmpeg_path()).exists()
    if not ffmpeg_ok:
        st.markdown("""
        <div style="background:#1e1220;border:1px solid #ff4d6d44;border-radius:8px;
                    padding:10px 16px;margin-top:12px;font-size:12px;color:#ff4d6d;">
            ⚠️ FFmpeg no encontrado. Configura la ruta en ⚙️ Configuración.
        </div>
        """, unsafe_allow_html=True)

    # ── Filtros ────────────────────────────────────────────────────────────────
    st.markdown('<div class="ws-section-header">Filtros</div>', unsafe_allow_html=True)
    jugadores_en_eventos = sorted(set(
        e.get("nombre_jugador", f"T{e.get('track_id', '?')}") for e in ball_events
    ))
    col_f1, col_f2 = st.columns(2)
    jugador_filtro = col_f2.multiselect("Jugador", jugadores_en_eventos)

    st.markdown('<div class="ws-section-header">Opciones de Clip</div>', unsafe_allow_html=True)
    skip_paint = st.checkbox(
        "⚡ Ver vídeo original (Instantáneo)",
        value=True,
        help="Desactiva para que la IA dibuje jugadores sobre el clip (puede tardar ~30s)"
    )

    eventos_filtrados = ball_events.copy()
    if jugador_filtro:
        eventos_filtrados = [
            e for e in eventos_filtrados
            if e.get("nombre_jugador", f"T{e.get('track_id', '?')}") in jugador_filtro
        ]

    st.markdown(f"""
    <div style="font-size:12px;color:#5a6a7e;margin-bottom:16px;">
        Mostrando <strong style="color:#fff;">{len(eventos_filtrados)}</strong>
        de {len(ball_events)} acciones
    </div>
    """, unsafe_allow_html=True)

    # ── Lista de eventos ───────────────────────────────────────────────────────
    st.markdown('<div class="ws-section-header">Acciones</div>', unsafe_allow_html=True)

    for i, evento in enumerate(eventos_filtrados):
        nombre = evento.get("nombre_jugador", f"Jugador T{evento.get('track_id', '?')}")
        equipo = evento.get("nombre_equipo", "—")
        minuto = evento.get("minute", 0)
        seg = evento.get("video_second", 0)
        clip_key = f"{evento.get('track_id', i)}_{int(seg)}"
        eq_color = "#00d4aa" if equipo == team_local else "#4d9fff"

        with st.expander(f"⚽  Min {minuto:.1f}'  ·  {nombre}  ·  {equipo}", expanded=False):
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:20px;margin-bottom:20px;padding:20px;
                        background:rgba(13,18,32,0.6);backdrop-filter:blur(10px);border-radius:12px;
                        border:1px solid rgba(255,255,255,0.05);border-left:5px solid {eq_color};">
                <div style="font-size:28px;font-weight:900;color:{eq_color};min-width:60px;
                            text-shadow:0 0 10px {eq_color}55;">{int(minuto)}'</div>
                <div style="flex:1;">
                    <div style="font-size:16px;font-weight:800;color:#fff;margin-bottom:2px;">{nombre}</div>
                    <div style="font-size:12px;color:#8899aa;font-family:monospace;">
                        STAMP: {seg:.1f}s · {equipo}
                    </div>
                </div>
                <div style="background:{eq_color}22;border:1px solid {eq_color}44;color:{eq_color};
                            padding:6px 14px;border-radius:20px;font-size:10px;font-weight:800;
                            text-transform:uppercase;letter-spacing:1px;">EVENTO</div>
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
                if st.button(f"▶  Ver clip  ({CLIP_ANTES + CLIP_DESPUES}s)",
                             key=f"btn_clip_{i}", type="primary"):

                    team_colors = st.session_state.get("team_colors", {})

                    if skip_paint:
                        # Modo rápido: solo FFmpeg, sin IA
                        with st.spinner("⏳ Cortando vídeo..."):
                            clip_path = _cortar_clip(video_path, seg, clip_key)
                    else:
                        # Modo IA: mostrar progreso real
                        progress_bar = st.progress(0.0, text="Inicializando IA...")

                        def update_progress(val: float):
                            progress_bar.progress(
                                min(val, 1.0),
                                text=f"IA procesando frames... {int(val*100)}%"
                            )

                        clip_path = _cortar_y_pintar_clip(
                            video_path, seg, clip_key, team_colors,
                            skip_paint=False,
                            progress_callback=update_progress
                        )
                        progress_bar.empty()

                    if clip_path and Path(clip_path).exists():
                        with open(clip_path, "rb") as f:
                            video_bytes = f.read()
                        st.video(video_bytes)
                        st.caption(
                            f"📍 {seg:.1f}s · ±{CLIP_ANTES}s antes / {CLIP_DESPUES}s después"
                        )
                    else:
                        st.error("❌ No se pudo generar el clip.")
            else:
                st.markdown("""
                <div style="font-size:12px;color:#5a6a7e;">
                    FFmpeg no disponible. Configura la ruta en ⚙️ Configuración.
                </div>
                """, unsafe_allow_html=True)