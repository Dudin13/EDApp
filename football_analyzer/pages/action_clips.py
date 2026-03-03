import streamlit as st
import numpy as np
import pandas as pd

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
    "Centro": "#3498db",
    "Córner": "#f1c40f",
    "Falta recibida": "#95a5a6",
}

def clip_card(clip, key):
    color = ACTION_COLORS.get(clip["accion"], "#888")
    resultado_color = "#2ecc71" if clip["resultado"] == "Exitosa" else "#e74c3c"
    st.markdown(f"""
    <div style="
        border: 1px solid {color};
        border-left: 5px solid {color};
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        background: #1a1a2e;
        cursor: pointer;
    ">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="color:{color}; font-weight:bold; font-size:0.95em;">
                {clip['accion']}
            </span>
            <span style="color:#FFD700; font-size:0.85em;">
                ⏱️ {clip['minuto']}'
            </span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-top:5px;">
            <span style="color:#cccccc; font-size:0.82em;">
                👤 {clip['jugador']} · #{clip['dorsal']}
            </span>
            <span style="color:{resultado_color}; font-size:0.82em;">
                {'✅' if clip['resultado'] == 'Exitosa' else '❌'} {clip['resultado']}
            </span>
        </div>
        <div style="color:#888; font-size:0.78em; margin-top:3px;">
            📍 {clip['zona']} · {clip['equipo']}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.button("▶ Ver clip", key=key, use_container_width=True)


def generar_clips_simulados():
    config = st.session_state.get("analysis_config", {})
    resultados = st.session_state.get("resultados_jugadores", {})

    if not resultados:
        return []

    np.random.seed(42)
    clips = []
    tipos = list(ACTION_COLORS.keys())
    zonas = ["Zona defensiva", "Zona media", "Zona ofensiva"]

    for nombre, datos in resultados.items():
        n = datos["total_actions"]
        for _ in range(n):
            clips.append({
                "jugador": nombre,
                "dorsal": datos["dorsal"],
                "equipo": datos["equipo"],
                "posicion": datos.get("posicion", ""),
                "accion": np.random.choice(tipos),
                "minuto": int(np.random.randint(1, 90)),
                "zona": np.random.choice(zonas),
                "resultado": np.random.choice(["Exitosa", "Fallida"], p=[0.7, 0.3]),
                "duracion": int(np.random.randint(8, 15)),
            })

    return sorted(clips, key=lambda x: x["minuto"])


def render():
    st.header("✂️ Clips por Acción")

    if not st.session_state.get("analysis_done"):
        st.info("⚠️ Primero realiza un análisis en 'Subir y Analizar Vídeo'")
        return

    clips = generar_clips_simulados()
    config = st.session_state.get("analysis_config", {})
    resultados = st.session_state.get("resultados_jugadores", {})

    team_local = config.get("team", "Local")
    team_visit = config.get("rival", "Visitante")

    # ── SELECTOR DE VISTA ─────────────────────────────────────────
    vista = st.radio("Vista", ["Por tipo de acción", "Por jugador"],
                     horizontal=True)

    st.markdown("---")

    # ── FILTROS COMUNES ───────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    equipo_filtro = col1.multiselect("Equipo", [team_local, team_visit])
    resultado_filtro = col2.radio("Resultado", ["Todos", "Exitosas", "Fallidas"],
                                   horizontal=True)
    zona_filtro = col3.multiselect("Zona", ["Zona defensiva", "Zona media", "Zona ofensiva"])

    # Aplicar filtros
    clips_filtrados = clips.copy()
    if equipo_filtro:
        clips_filtrados = [c for c in clips_filtrados if c["equipo"] in equipo_filtro]
    if resultado_filtro == "Exitosas":
        clips_filtrados = [c for c in clips_filtrados if c["resultado"] == "Exitosa"]
    elif resultado_filtro == "Fallidas":
        clips_filtrados = [c for c in clips_filtrados if c["resultado"] == "Fallida"]
    if zona_filtro:
        clips_filtrados = [c for c in clips_filtrados if c["zona"] in zona_filtro]

    st.markdown(f"**{len(clips_filtrados)} clips** encontrados")
    st.markdown("---")

    # ── VISTA POR TIPO DE ACCIÓN ──────────────────────────────────
    if vista == "Por tipo de acción":
        tipos_presentes = sorted(set(c["accion"] for c in clips_filtrados))

        for tipo in tipos_presentes:
            clips_tipo = [c for c in clips_filtrados if c["accion"] == tipo]
            color = ACTION_COLORS.get(tipo, "#888")

            with st.expander(
                f"**{tipo}** — {len(clips_tipo)} clips",
                expanded=False
            ):
                # Stats rápidas del tipo
                exitosas = sum(1 for c in clips_tipo if c["resultado"] == "Exitosa")
                pct = round(exitosas / len(clips_tipo) * 100) if clips_tipo else 0

                m1, m2, m3 = st.columns(3)
                m1.metric("Total", len(clips_tipo))
                m2.metric("Exitosas", exitosas)
                m3.metric("% éxito", f"{pct}%")

                st.markdown("---")

                # Grid de clips
                cols = st.columns(2)
                for i, clip in enumerate(clips_tipo):
                    with cols[i % 2]:
                        clip_card(clip, key=f"tipo_{tipo}_{i}")

    # ── VISTA POR JUGADOR ─────────────────────────────────────────
    elif vista == "Por jugador":
        jugadores_presentes = sorted(set(c["jugador"] for c in clips_filtrados))

        if not jugadores_presentes:
            st.info("No hay clips con los filtros seleccionados")
            return

        # Selector de jugador estilo Wyscout
        col_lista, col_clips = st.columns([1, 3])

        with col_lista:
            st.markdown("**Jugadores**")
            jugador_sel = None
            for jugador in jugadores_presentes:
                datos = resultados.get(jugador, {})
                equipo = datos.get("equipo", "")
                dorsal = datos.get("dorsal", "")
                n_clips = sum(1 for c in clips_filtrados if c["jugador"] == jugador)
                color_eq = "#FFD700" if equipo == team_local else "#3498db"

                if st.button(
                    f"#{dorsal} {jugador}\n{n_clips} clips",
                    key=f"jugador_btn_{jugador}",
                    use_container_width=True
                ):
                    st.session_state["jugador_seleccionado"] = jugador

            jugador_sel = st.session_state.get("jugador_seleccionado", jugadores_presentes[0] if jugadores_presentes else None)

        with col_clips:
            if jugador_sel:
                datos_j = resultados.get(jugador_sel, {})
                clips_jugador = [c for c in clips_filtrados if c["jugador"] == jugador_sel]

                # Header jugador
                st.markdown(f"""
                <div style="background:#1a1a2e; border-left:5px solid #FFD700;
                            padding:12px 16px; border-radius:8px; margin-bottom:15px;">
                    <h3 style="color:#FFD700; margin:0;">
                        #{datos_j.get('dorsal','')} {jugador_sel}
                    </h3>
                    <p style="color:#ccc; margin:3px 0 0 0;">
                        {datos_j.get('equipo','')} · {datos_j.get('posicion','')}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # KPIs del jugador
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Acciones", datos_j.get("total_actions", 0))
                k2.metric("Pases", datos_j.get("passes", 0))
                k3.metric("Duelos G.", datos_j.get("duels_won", 0))
                k4.metric("Recuper.", datos_j.get("recoveries", 0))

                st.markdown("---")

                # Filtro por tipo dentro del jugador
                tipos_jugador = sorted(set(c["accion"] for c in clips_jugador))
                tipo_sel = st.multiselect("Filtrar por acción", tipos_jugador, key="tipo_jugador")

                clips_mostrar = clips_jugador
                if tipo_sel:
                    clips_mostrar = [c for c in clips_jugador if c["accion"] in tipo_sel]

                # Clips del jugador
                cols = st.columns(2)
                for i, clip in enumerate(clips_mostrar):
                    with cols[i % 2]:
                        clip_card(clip, key=f"jugador_clip_{jugador_sel}_{i}")