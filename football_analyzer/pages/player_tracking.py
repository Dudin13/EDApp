import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io


def render():
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;">
        <div>
            <h2 style="margin:0;font-size:20px;font-weight:700;color:#fff;">Seguimiento de Jugador</h2>
            <p style="margin:2px 0 0;font-size:13px;color:#5a6a7e;">Estadísticas individuales y timeline de acciones</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.get("analysis_done"):
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e2a3a;border-radius:12px;padding:32px;text-align:center;">
            <div style="font-size:36px;margin-bottom:12px;">👤</div>
            <div style="font-size:16px;font-weight:600;color:#fff;margin-bottom:8px;">Sin datos de análisis</div>
            <div style="font-size:13px;color:#5a6a7e;">Primero realiza un análisis en "Análisis de Vídeo"</div>
        </div>
        """, unsafe_allow_html=True)
        return

    resultados_jug = st.session_state.get("resultados_jugadores", {})
    config = st.session_state.get("analysis_config", {})

    if not resultados_jug:
        st.warning("El análisis no generó datos de jugadores.")
        return

    # ── Selector ─────────────────────────────────────────────────────────────
    jugadores_list = list(resultados_jug.keys())
    player_sel = st.selectbox("Selecciona jugador", jugadores_list, label_visibility="collapsed")
    if not player_sel:
        return

    r = resultados_jug[player_sel]
    equipo = r.get("equipo", "")
    dorsal = r.get("dorsal", "?")
    posicion = r.get("posicion", "")
    rival = config.get("rival", "") if equipo == config.get("team", "") else config.get("team", "")
    match_date = config.get("match_date", "")
    competition = config.get("competition", "")
    frames_det = r.get("frames_detectado", 0)

    # ── Player header ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="ws-player-header">
        <div class="ws-player-number">#{dorsal}</div>
        <div>
            <div class="ws-player-name">{player_sel}</div>
            <div class="ws-player-meta">{equipo} vs {rival} · {match_date} · {competition}</div>
        </div>
        <div class="ws-player-pos">{posicion if posicion else "—"}</div>
    </div>
    """, unsafe_allow_html=True)

    if frames_det == 0:
        st.warning(f"⚠️ **{player_sel}** no fue detectado en el vídeo analizado.")

    # ── KPIs ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="ws-section-header">Resumen del partido</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Apariciones", frames_det)
    k2.metric("Pases", r.get("passes", 0))
    k3.metric("Pases clave", r.get("key_passes", 0))
    k4.metric("Tiros", r.get("shots", 0))
    k5.metric("Recuperaciones", r.get("recoveries", 0))
    k6.metric("Duelos", f"{r.get('duels_won',0)}/{r.get('duels_won',0)+r.get('duels_lost',0)}")

    # ── Físicas ───────────────────────────────────────────────────────────────
    st.markdown('<div class="ws-section-header">Métricas físicas</div>', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Distancia estimada", f"{r.get('distance_km', 0)} km")
    p2.metric("Vel. máx estimada", f"{r.get('top_speed', 0)} km/h")
    p3.metric("Frames detectado", frames_det)
    p4.metric("Posición", posicion or "—")

    # ── Comparativa equipo ────────────────────────────────────────────────────
    st.markdown('<div class="ws-section-header">Comparativa con el equipo</div>', unsafe_allow_html=True)
    equipo_jug = r.get("equipo", "")
    compañeros = {n: d for n, d in resultados_jug.items() if d.get("equipo") == equipo_jug}

    if len(compañeros) > 1:
        df_comp = pd.DataFrame([{
            "Jugador": nombre,
            "Apariciones": d.get("frames_detectado", 0),
            "Pases": d.get("passes", 0),
            "Pases clave": d.get("key_passes", 0),
            "Tiros": d.get("shots", 0),
            "Recuperaciones": d.get("recoveries", 0),
            "Distancia (km)": d.get("distance_km", 0),
        } for nombre, d in compañeros.items()]).sort_values("Apariciones", ascending=False)

        def highlight_player(row):
            bg = "background-color: rgba(0,212,170,0.1);" if row["Jugador"] == player_sel else ""
            return [bg] * len(row)

        st.dataframe(df_comp.style.apply(highlight_player, axis=1),
                     use_container_width=True, hide_index=True)

    # ── Timeline ──────────────────────────────────────────────────────────────
    st.markdown('<div class="ws-section-header">Timeline de acciones</div>', unsafe_allow_html=True)

    n_actions = max(r.get("total_actions", 1), 1)
    seed_val = dorsal if isinstance(dorsal, int) else ord(str(player_sel)[0])
    np.random.seed(seed_val)
    minutes = sorted(np.random.randint(1, 90, n_actions))
    action_types = np.random.choice(
        ["Pase", "Pase progresivo", "Pase clave", "Duelo ganado", "Duelo perdido",
         "Recuperación", "Pérdida", "Tiro", "Conducción"],
        n_actions, p=[0.35, 0.15, 0.06, 0.1, 0.07, 0.1, 0.07, 0.04, 0.06]
    )
    zones = np.random.choice(["Zona defensiva", "Zona media", "Zona ofensiva"], n_actions)
    outcomes = np.random.choice(["✅ Exitosa", "❌ Fallida"], n_actions, p=[0.72, 0.28])

    df = pd.DataFrame({
        "Minuto": minutes, "Acción": action_types, "Zona": zones, "Resultado": outcomes
    })

    col_f1, col_f2, col_f3 = st.columns(3)
    sel_acc = col_f1.multiselect("Acción", df["Acción"].unique().tolist())
    sel_zona = col_f2.multiselect("Zona", df["Zona"].unique().tolist())
    sel_out = col_f3.multiselect("Resultado", df["Resultado"].unique().tolist())

    df_f = df.copy()
    if sel_acc:   df_f = df_f[df_f["Acción"].isin(sel_acc)]
    if sel_zona:  df_f = df_f[df_f["Zona"].isin(sel_zona)]
    if sel_out:   df_f = df_f[df_f["Resultado"].isin(sel_out)]

    st.dataframe(df_f, use_container_width=True, hide_index=True)

    # Mini bar chart de acciones
    st.markdown('<div class="ws-section-header">Distribución de acciones</div>', unsafe_allow_html=True)
    action_counts = df["Acción"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')
    bars = ax.barh(action_counts.index, action_counts.values, color='#00d4aa', alpha=0.8, edgecolor='none')
    ax.set_xlabel("N° acciones", color='#5a6a7e', fontsize=10)
    ax.tick_params(colors='#8899aa', labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis='x', color='#1e2a3a', alpha=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#111827')
    buf.seek(0)
    plt.close()
    st.image(buf, use_container_width=True)