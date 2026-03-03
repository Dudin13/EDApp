import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

def render():
    st.header("📊 Métricas del Partido")

    if not st.session_state.get("analysis_done"):
        st.info("⚠️ Primero realiza un análisis en la sección 'Subir y Analizar Vídeo'")
        return

    results = st.session_state.get("mock_results", {})
    player = results.get("player_name", "Jugador")

    st.subheader(f"🕸️ Perfil técnico-táctico — {player}")

    categories = ['Pases\ncompletados', 'Progresividad', 'Creación\npeligro',
                  'Duelos\nganados', 'Recuperaciones', 'Presiones\nefectivas',
                  'Conducciones\nprogresivas', 'Tiros\na puerta']

    values_player = [72, 85, 68, 60, 78, 55, 80, 50]
    values_team = [65, 60, 55, 58, 62, 60, 58, 48]

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    values_p = values_player + values_player[:1]
    values_t = values_team + values_team[:1]

    ax.plot(angles, values_p, 'o-', linewidth=2, color='#FFD700', label=player)
    ax.fill(angles, values_p, alpha=0.3, color='#FFD700')
    ax.plot(angles, values_t, 'o-', linewidth=2, color='#888888', linestyle='--', label='Media equipo')
    ax.fill(angles, values_t, alpha=0.1, color='#888888')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='white', size=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], color='gray', size=8)
    ax.grid(color='gray', alpha=0.3)
    ax.spines['polar'].set_color('gray')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
              facecolor='#1a1a2e', labelcolor='white')

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
            player: values_player,
            "Media equipo": values_team,
            "Diferencia": [p - t for p, t in zip(values_player, values_team)]
        })

        def color_diff(val):
            color = '#d4edda' if val >= 0 else '#f8d7da'
            return f'background-color: {color}'

        st.dataframe(
            metrics_df.style.applymap(color_diff, subset=["Diferencia"]),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")
    st.subheader("⏱️ Evolución de acciones por período (15 min)")
    periods = ['1-15', '16-30', '31-45', '46-60', '61-75', '76-90']
    np.random.seed(7)
    actions_by_period = np.random.randint(5, 15, 6)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    fig2.patch.set_facecolor('#1a1a2e')
    ax2.set_facecolor('#1a1a2e')
    bars = ax2.bar(periods, actions_by_period, color='#FFD700', alpha=0.8, edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Período (minutos)', color='white')
    ax2.set_ylabel('Nº acciones', color='white')
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('gray')
    ax2.spines['left'].set_color('gray')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    for bar, val in zip(bars, actions_by_period):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                str(val), ha='center', va='bottom', color='white', fontsize=11)

    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    buf2.seek(0)
    plt.close()
    st.image(buf2, use_column_width=True)

    st.markdown("---")
    st.subheader("📥 Exportar informe")
    col_exp1, col_exp2 = st.columns(2)
    col_exp1.button("📄 Exportar PDF", type="primary", use_container_width=True)
    col_exp2.button("📊 Exportar Excel", use_container_width=True)