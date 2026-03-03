"""
action_clips.py — Clips de vídeo reales cortados con FFmpeg.
Estilo Wyscout: tarjetas de evento oscuras, teal accent, diseño limpio.
"""

import streamlit as st
import subprocess
import os
import tempfile
from pathlib import Path

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
CLIPS_DIR = Path(tempfile.gettempdir()) / "ed_analytics_clips"
CLIPS_DIR.mkdir(exist_ok=True)


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
        <div style="background:#111827;border:1px solid #1e2a3a;border-radius:12px;padding:32px;text-align:center;">
            <div style="font-size:36px;margin-bottom:12px;">✂️</div>
            <div style="font-size:16px;font-weight:600;color:#fff;margin-bottom:8px;">Sin datos de análisis</div>
            <div style="font-size:13px;color:#5a6a7e;">Primero realiza un análisis en "Análisis de Vídeo"</div>
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
        <div style="background:#111827;border:1px solid #ff6b3544;border-radius:12px;padding:28px 32px;">
            <div style="font-size:15px;font-weight:600;color:#fff;margin-bottom:10px;">⚠️ No se detectaron acciones con balón</div>
            <div style="font-size:13px;color:#5a6a7e;line-height:1.7;">
                Posibles causas:<br>
                • El modelo no detectó el balón en el vídeo<br>
                • El intervalo entre frames es demasiado grande<br>
                <br>
                <strong style="color:#00d4aa;">Consejo:</strong> Repite el análisis con intervalo de <strong>1–2 segundos</strong>.
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
            # Cabecera del clip
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;padding:12px 16px;
                        background:#0d1220;border-radius:8px;border-left:3px solid {eq_color};">
                <div style="font-size:22px;font-weight:800;color:{eq_color};min-width:50px;">
                    {minuto:.0f}'
                </div>
                <div style="flex:1;">
                    <div style="font-size:14px;font-weight:600;color:#fff;">{nombre}</div>
                    <div style="font-size:12px;color:#5a6a7e;">{equipo} · Segundo {seg:.1f}s del vídeo</div>
                </div>
                <div style="background:{eq_color}22;border:1px solid {eq_color}44;color:{eq_color};
                            padding:4px 12px;border-radius:16px;font-size:11px;font-weight:600;">
                    {equipo.upper()[:12]}
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
                    with st.spinner("Generando clip…"):
                        clip_path = _cortar_clip(video_path, seg, clip_key)

                    if clip_path:
                        st.video(clip_path)
                        st.caption(f"📍 {seg:.1f}s en el vídeo · ±{CLIP_ANTES}s antes / {CLIP_DESPUES}s después")
                    else:
                        st.error("❌ No se pudo generar el clip.")
            else:
                st.markdown("""
                <div style="font-size:12px;color:#5a6a7e;">
                    FFmpeg no disponible. Configura la ruta en ⚙️ Configuración para generar clips.
                </div>
                """, unsafe_allow_html=True)