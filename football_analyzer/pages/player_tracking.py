import streamlit as st
import pandas as pd
import numpy as np

def render():
    st.header("👤 Seguimiento de Jugador")

    if not st.session_state.get("analysis_done"):
        st.info("⚠️ Primero realiza un análisis en la sección 'Subir y Analizar Vídeo'")
        return

    results = st.session_state.get("mock_results", {})
    config = st.session_state.get("analysis_config", {})
    player = results.get("player_name", "Jugador")
    number = results.get("player_number", "?")
    team = config.get("team", "")
    rival = config.get("rival", "")
    date = config.get("match_date", "")
    comp = config.get("competition", "")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a1a2e, #FFD700); 
                    padding: 20px; border-radius: 10px; text-align: center; color: white;">
            <h2>#{number} {player}</h2>
            <p>{team} vs {rival} | {date} | {comp}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Resumen del partido")
    kpi_cols = st.columns(6)
    kpis = [
        ("Acciones totales", results["total_actions"]),
        ("Pases", results["passes"]),
        ("Pases clave", results["key_passes"]),
        ("Tiros", results["shots"]),
        ("Recuperaciones", results["recoveries"]),
        ("Duelos ganados", f"{results['duels_won']}/{results['duels_won']+results['duels_lost']}"),
    ]
    for col, (label, value) in zip(kpi_cols, kpis):
        col.metric(label, value)

    st.markdown("---")
    st.subheader("🏃 Métricas físicas")
    phys_cols = st.columns(4)
    phys_cols[0].metric("Distancia total", f"{results['distance_km']} km")
    phys_cols[1].metric("Velocidad máxima", f"{results['top_speed']} km/h")
    phys_cols[2].metric("Sprints (+25 km/h)", "12")
    phys_cols[3].metric("Tiempo en posesión", "4:32 min")

    st.markdown("---")
    st.subheader("⏱️ Timeline de acciones")

    np.random.seed(42)
    n_actions = results["total_actions"]
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
        "Minuto": minutes,
        "Acción": action_types,
        "Zona": zones,
        "Resultado": outcomes
    })

    col_f1, col_f2, col_f3 = st.columns(3)
    selected_actions = col_f1.multiselect("Filtrar por acción", df["Acción"].unique().tolist())
    selected_zones = col_f2.multiselect("Filtrar por zona", df["Zona"].unique().tolist())
    selected_outcome = col_f3.multiselect("Filtrar por resultado", df["Resultado"].unique().tolist())

    df_filtered = df.copy()
    if selected_actions:
        df_filtered = df_filtered[df_filtered["Acción"].isin(selected_actions)]
    if selected_zones:
        df_filtered = df_filtered[df_filtered["Zona"].isin(selected_zones)]
    if selected_outcome:
        df_filtered = df_filtered[df_filtered["Resultado"].isin(selected_outcome)]

    st.dataframe(df_filtered, use_container_width=True, hide_index=True)

    st.subheader("📈 Distribución de acciones")
    action_counts = df["Acción"].value_counts()
    st.bar_chart(action_counts)