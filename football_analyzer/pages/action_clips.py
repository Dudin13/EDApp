"""
action_clips.py — Clips de vídeo reales cortados con FFmpeg.

Para cada evento con balón detectado (video_second, track_id),
FFmpeg corta un fragmento de ±5 segundos del vídeo original.
El clip se reproduce directamente en la app con st.video().
"""

import streamlit as st
import subprocess
import os
import tempfile
from pathlib import Path

ACTION_COLORS = {
    "Pase": "#3498db",
    "Pase progresivo": "#2ecc71",
    "Pase clave": "#f39c12",
    "Duelo ganado": "#27ae60",
    "Duelo perdido": "#e74c3c",
    "Recuperación": "#1abc9c",
    "Pérdida": "#e67e22",
    "Tiro": "#e74c3c",
    "Conducción": "#9b59b6",
    "Córner": "#f1c40f",
    "Falta": "#95a5a6",
    "Acción con balón": "#FFD700",
}

CLIP_ANTES = 3    # segundos antes de la acción
CLIP_DESPUES = 7  # segundos después de la acción
CLIPS_DIR = Path(tempfile.gettempdir()) / "ed_analytics_clips"
CLIPS_DIR.mkdir(exist_ok=True)


def _get_ffmpeg_path() -> str:
    """Obtiene la ruta de FFmpeg desde .env o por defecto."""
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env")
    except Exception:
        pass
    return os.getenv("FFMPEG_PATH", "C:\\ffmpeg\\bin\\ffmpeg.exe")


def _cortar_clip(video_path: str, video_second: float, clip_key: str) -> str | None:
    """
    Corta un clip de 10 segundos del vídeo usando FFmpeg.
    Retorna la ruta del clip o None si falla.
    """
    ffmpeg = _get_ffmpeg_path()
    if not Path(ffmpeg).exists():
        return None

    start = max(0, video_second - CLIP_ANTES)
    duracion = CLIP_ANTES + CLIP_DESPUES
    out_path = CLIPS_DIR / f"clip_{clip_key}.mp4"

    if out_path.exists():
        return str(out_path)  # ya existe, reutilizar

    cmd = [
        ffmpeg, "-y",
        "-ss", f"{start:.2f}",
        "-i", video_path,
        "-t", str(duracion),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(out_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if out_path.exists() and out_path.stat().st_size > 10_000:
            return str(out_path)
    except Exception:
        pass
    return None


def _nombre_jugador_por_track(track_id, resultados_jug: dict, track_map: dict) -> str:
    """Intenta mapear un track_id a un nombre de jugador."""
    return track_map.get(track_id, f"Jugador T{track_id}")


def render():
    st.header("✂️ Clips por Acción")

    if not st.session_state.get("analysis_done"):
        st.info("⚠️ Primero realiza un análisis en 'Subir y Analizar Vídeo'")
        return

    ball_events = st.session_state.get("ball_events", [])
    resultados_jug = st.session_state.get("resultados_jugadores", {})
    video_path = st.session_state.get("video_path", "")
    config = st.session_state.get("analysis_config", {})
    team_local = config.get("team", "Local")
    team_visit = config.get("rival", "Visitante")

    # ── Sin eventos reales: mostrar mensaje claro ─────────────────
    if not ball_events:
        st.warning("""
        ⚠️ **No se detectaron eventos con balón** en este análisis.

        Esto puede deberse a que:
        - El modelo no detectó el balón en el vídeo (fundos oscuros, balón pequeño)
        - El intervalo entre frames es demasiado grande para capturar contactos

        **Consejo:** Repite el análisis con **intervalo de 1-2 segundos** para capturar más momentos con balón.
        """)
        return

    # ── Resumen ────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Acciones con balón", len(ball_events))
    col2.metric("Duración clip", f"{CLIP_ANTES+CLIP_DESPUES}s por acción")
    col3.metric("Vídeo fuente", Path(video_path).name if video_path else "—")

    st.markdown("---")

    # ── Filtros ────────────────────────────────────────────────────
    jugadores_en_eventos = sorted(set(
        e.get("nombre_jugador", f"T{e['track_id']}") for e in ball_events
    ))

    col_f1, col_f2 = st.columns(2)
    equipo_filtro = col_f1.multiselect("Filtrar por equipo", [team_local, team_visit])
    jugador_filtro = col_f2.multiselect("Filtrar por jugador", jugadores_en_eventos)

    eventos_filtrados = ball_events.copy()
    if jugador_filtro:
        eventos_filtrados = [
            e for e in eventos_filtrados
            if e.get("nombre_jugador", f"T{e['track_id']}") in jugador_filtro
        ]

    st.markdown(f"**{len(eventos_filtrados)} acciones** encontradas")
    st.markdown("---")

    # ── Lista de eventos ───────────────────────────────────────────
    ffmpeg_ok = Path(_get_ffmpeg_path()).exists()

    for i, evento in enumerate(eventos_filtrados):
        nombre = evento.get("nombre_jugador", f"Jugador T{evento['track_id']}")
        equipo = evento.get("nombre_equipo", "—")
        minuto = evento.get("minute", 0)
        seg = evento.get("video_second", 0)
        clip_key = f"{evento['track_id']}_{int(seg)}"

        color = "#FFD700" if equipo == team_local else "#3498db"

        with st.expander(
            f"⚽ **{nombre}** — Min {minuto:.1f}' | {equipo}",
            expanded=False
        ):
            k1, k2, k3 = st.columns(3)
            k1.markdown(f"**Jugador:** {nombre}")
            k2.markdown(f"**Equipo:** {equipo}")
            k3.markdown(f"**Segundo:** {seg:.1f}s en el vídeo")

            if not video_path or not Path(video_path).exists():
                st.warning("⚠️ El vídeo original ya no está disponible en la sesión.")
                continue

            if st.button(f"▶ Ver clip ({CLIP_ANTES+CLIP_DESPUES}s)", key=f"btn_clip_{i}"):
                with st.spinner("Cortando clip con FFmpeg..."):
                    clip_path = _cortar_clip(video_path, seg, clip_key)

                if clip_path:
                    st.video(clip_path)
                    st.caption(f"📍 Segundo {seg:.1f}s del vídeo | ±{CLIP_ANTES}s antes / {CLIP_DESPUES}s después")
                else:
                    st.error("❌ No se pudo generar el clip. Comprueba FFmpeg en Configuración.")