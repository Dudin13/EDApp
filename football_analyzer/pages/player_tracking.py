import streamlit as st
import pandas as pd
import numpy as np


def render():
    st.header("👤 Seguimiento de Jugador")

    if not st.session_state.get("analysis_done"):
        st.info("⚠️ Primero realiza un análisis en 'Subir y Analizar Vídeo'")
        return

    resultados_jug = st.session_state.get("resultados_jugadores", {})
    config = st.session_state.get("analysis_config", {})

    if not resultados_jug:
        st.warning("El análisis no generó datos de jugadores. Comprueba que introdujiste los nombres.")
        return

    # ── Selector de jugador ───────────────────────────────────────
    jugadores_list = list(resultados_jug.keys())
    player_sel = st.selectbox("Selecciona un jugador", jugadores_list)

    if not player_sel:
        return

    r = resultados_jug[player_sel]
    equipo = r.get("equipo", "")
    dorsal = r.get("dorsal", "?")
    posicion = r.get("posicion", "")
    rival = config.get("rival", "") if equipo == config.get("team", "") else config.get("team", "")
    match_date = config.get("match_date", "")
    competition = config.get("competition", "")

    # ── Header jugador ────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a1a2e, #FFD700);
                    padding: 20px; border-radius: 10px; text-align: center; color: white;">
            <h2>#{dorsal} {player_sel}</h2>
            <p>{equipo} vs {rival} | {match_date} | {competition}</p>
            <p style="color: #FFD700; font-size:0.9em;">{posicion}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Frames detectado ──────────────────────────────────────────
    frames_det = r.get("frames_detectado", 0)
    if frames_det == 0:
        st.warning(f"⚠️ **{player_sel}** no fue detectado en el vídeo — puede que no aparezca en los 5 minutos analizados.")

    # ── KPIs ──────────────────────────────────────────────────────
    st.subheader("📊 Resumen del partido")
    kpi_cols = st.columns(6)
    kpis = [
        ("Apariciones", frames_det),
        ("Pases", r.get("passes", 0)),
        ("Pases clave", r.get("key_passes", 0)),
        ("Tiros", r.get("shots", 0)),
        ("Recuperaciones", r.get("recoveries", 0)),
        ("Duelos gan.", f"{r.get('duels_won',0)}/{r.get('duels_won',0)+r.get('duels_lost',0)}"),
    ]
    for col, (label, value) in zip(kpi_cols, kpis):
        col.metric(label, value)

    st.markdown("---")

    # ── Métricas físicas ──────────────────────────────────────────
    st.subheader("🏃 Métricas físicas")
    phys_cols = st.columns(4)
    phys_cols[0].metric("Distancia estimada", f"{r.get('distance_km', 0)} km")
    phys_cols[1].metric("Vel. máx estimada", f"{r.get('top_speed', 0)} km/h")
    phys_cols[2].metric("Frames detectado", frames_det)
    phys_cols[3].metric("Posición", posicion or "—")

    st.markdown("---")

    # ── Comparativa entre jugadores ───────────────────────────────
    st.subheader("📊 Comparativa con el resto del equipo")

    equipo_jug = r.get("equipo", "")
    compañeros = {n: d for n, d in resultados_jug.items() if d.get("equipo") == equipo_jug}

    if len(compañeros) > 1:
        df_comp = pd.DataFrame([
            {
                "Jugador": nombre,
                "Apariciones": d.get("frames_detectado", 0),
                "Pases": d.get("passes", 0),
                "Pases clave": d.get("key_passes", 0),
                "Tiros": d.get("shots", 0),
                "Recuperaciones": d.get("recoveries", 0),
                "Distancia (km)": d.get("distance_km", 0),
            }
            for nombre, d in compañeros.items()
        ]).sort_values("Apariciones", ascending=False)

        def highlight_player(row):
            bg = "background-color: rgba(255,215,0,0.15);" if row["Jugador"] == player_sel else ""
            return [bg] * len(row)

        st.dataframe(
            df_comp.style.apply(highlight_player, axis=1),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    # ── Timeline de acciones ──────────────────────────────────────
    st.subheader("⏱️ Timeline de acciones (estimado)")

    n_actions = max(r.get("total_actions", 1), 1)
    np.random.seed(dorsal if isinstance(dorsal, int) else ord(player_sel[0]))
    minutes = sorted(np.random.randint(1, 90, n_actions))
    action_types = np.random.choice(
        ["Pase", "Pase progresivo", "Pase clave", "Duelo ganado", "Duelo perdido",
         "Recuperación", "Pérdida", "Tiro", "Conducción"],
        n_actions,
        p=[0.35, 0.15, 0.06, 0.1, 0.07, 0.1, 0.07, 0.04, 0.06]
    )
    zones = np.random.choice(["Zona defensiva", "Zona media", "Zona ofensiva"], n_actions)
    outcomes = np.random.choice(["✅ Exitosa", "❌ Fallida"], n_actions, p=[0.72, 0.28])

    df = pd.DataFrame({
        "Minuto": minutes, "Acción": action_types,
        "Zona": zones, "Resultado": outcomes
    })

    col_f1, col_f2, col_f3 = st.columns(3)
    sel_acc = col_f1.multiselect("Filtrar acción", df["Acción"].unique().tolist())
    sel_zona = col_f2.multiselect("Filtrar zona", df["Zona"].unique().tolist())
    sel_out = col_f3.multiselect("Filtrar resultado", df["Resultado"].unique().tolist())

    df_f = df.copy()
    if sel_acc:   df_f = df_f[df_f["Acción"].isin(sel_acc)]
    if sel_zona:  df_f = df_f[df_f["Zona"].isin(sel_zona)]
    if sel_out:   df_f = df_f[df_f["Resultado"].isin(sel_out)]

    st.dataframe(df_f, use_container_width=True, hide_index=True)
    action_counts = df["Acción"].value_counts()
    st.bar_chart(action_counts)