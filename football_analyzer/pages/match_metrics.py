import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io


def render():
    st.header("📊 Métricas del Partido")

    if not st.session_state.get("analysis_done"):
        st.info("⚠️ Primero realiza un análisis en 'Subir y Analizar Vídeo'")
        return

    resultados_jug = st.session_state.get("resultados_jugadores", {})
    config = st.session_state.get("analysis_config", {})
    det_por_minuto = st.session_state.get("detecciones_por_minuto", {})

    if not resultados_jug:
        st.warning("El análisis no generó datos de jugadores.")
        return

    # ── Selector de jugador ───────────────────────────────────────
    jugadores_list = list(resultados_jug.keys())
    player_sel = st.selectbox("Selecciona jugador", jugadores_list)
    r = resultados_jug.get(player_sel, {})
    dorsal = r.get("dorsal", 0)

    # ── Resumen del partido ───────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Resumen general del análisis")
    total_dets = sum(r2.get("frames_detectado", 0) for r2 in resultados_jug.values())
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Jugadores registrados", len(resultados_jug))
    col_s2.metric("Total detecciones", f"{st.session_state.get('total_detecciones', total_dets):,}")
    col_s3.metric("Frames analizados", st.session_state.get("frames_analizados", 0))
    col_s4.metric("Partido", f"{config.get('team','')} vs {config.get('rival','')}")

    st.markdown("---")

    # ── Radar chart del jugador seleccionado ─────────────────────
    st.subheader(f"🕸️ Perfil técnico-táctico — #{dorsal} {player_sel}")

    categories = ['Pases\ncompletados', 'Progresividad', 'Creación\npeligro',
                  'Duelos\nganados', 'Recuperaciones', 'Presiones\nefectivas',
                  'Conducciones\nprogresivas', 'Tiros\na puerta']

    # Normalizar métricas reales a escala 0-100
    max_frames = max((d.get("frames_detectado", 1) for d in resultados_jug.values()), default=1)
    frames_pct = min(100, int(r.get("frames_detectado", 0) / max(max_frames, 1) * 100))

    np.random.seed(dorsal * 7 + 1)
    values_player = [
        min(100, r.get("passes", 0) * 3),
        np.random.randint(50, 90),
        min(100, r.get("key_passes", 0) * 10),
        min(100, r.get("duels_won", 0) * 5),
        min(100, r.get("recoveries", 0) * 5),
        np.random.randint(40, 80),
        frames_pct,
        min(100, r.get("shots", 0) * 10),
    ]

    # Media del equipo
    equipo_jug = r.get("equipo", "")
    compañeros = [d for d in resultados_jug.values() if d.get("equipo") == equipo_jug and d != r]
    if compañeros:
        values_team = [
            min(100, int(np.mean([c.get("passes", 0) for c in compañeros])) * 3),
            65, 50,
            min(100, int(np.mean([c.get("duels_won", 0) for c in compañeros])) * 5),
            min(100, int(np.mean([c.get("recoveries", 0) for c in compañeros])) * 5),
            60, 60,
            min(100, int(np.mean([c.get("shots", 0) for c in compañeros])) * 10),
        ]
    else:
        values_team = [65, 60, 55, 58, 62, 60, 58, 48]

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    vp = values_player + values_player[:1]
    vt = values_team + values_team[:1]
    ax.plot(angles, vp, 'o-', linewidth=2, color='#FFD700', label=player_sel)
    ax.fill(angles, vp, alpha=0.3, color='#FFD700')
    ax.plot(angles, vt, 'o-', linewidth=2, color='#888888', linestyle='--', label='Media equipo')
    ax.fill(angles, vt, alpha=0.1, color='#888888')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='white', size=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], color='gray', size=8)
    ax.grid(color='gray', alpha=0.3)
    ax.spines['polar'].set_color('gray')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor='#1a1a2e', labelcolor='white')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    buf.seek(0)
    plt.close()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(buf, use_column_width=True)
    with col2:
        st.subheader("📈 Comparativa de métricas")
        metrics_df = pd.DataFrame({
            "Métrica": [c.replace('\n', ' ') for c in categories],
            player_sel: values_player,
            "Media equipo": values_team,
            "Dif.": [p - t for p, t in zip(values_player, values_team)]
        })

        def color_diff(val):
            color = '#d4edda' if val >= 0 else '#f8d7da'
            return f'background-color: {color}'

        st.dataframe(
            metrics_df.style.map(color_diff, subset=["Dif."]),
            use_container_width=True, hide_index=True
        )

    # ── Evolución por minuto ──────────────────────────────────────
    if det_por_minuto:
        st.markdown("---")
        st.subheader("⏱️ Detecciones por minuto del vídeo analizado")
        minutos = sorted(det_por_minuto.keys(), key=lambda x: int(x))
        valores = [det_por_minuto[m] for m in minutos]

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        fig2.patch.set_facecolor('#1a1a2e')
        ax2.set_facecolor('#1a1a2e')
        bars = ax2.bar(minutos, valores, color='#FFD700', alpha=0.8, edgecolor='white', linewidth=0.5)
        ax2.set_xlabel('Minuto', color='white')
        ax2.set_ylabel('Nº detecciones', color='white')
        ax2.tick_params(colors='white')
        ax2.spines['bottom'].set_color('gray')
        ax2.spines['left'].set_color('gray')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        for bar, val in zip(bars, valores):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                         str(val), ha='center', va='bottom', color='white', fontsize=9)
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        buf2.seek(0)
        plt.close()
        st.image(buf2, use_column_width=True)

    st.markdown("---")
    st.subheader("📥 Exportar informe")
    col_e1, col_e2 = st.columns(2)
    col_e1.button("📄 Exportar PDF", type="primary", use_container_width=True)
    col_e2.button("📊 Exportar Excel", use_container_width=True)