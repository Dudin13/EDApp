import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc, Circle
import io

def draw_pitch(ax, color='green', line_color='white'):
    ax.set_facecolor(color)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 68)
    ax.set_aspect('equal')
    ax.axis('off')

    lw = 1.5
    lc = line_color

    rect = patches.Rectangle((0, 0), 105, 68, linewidth=lw, edgecolor=lc, facecolor='none')
    ax.add_patch(rect)
    ax.plot([52.5, 52.5], [0, 68], color=lc, linewidth=lw)
    circle = Circle((52.5, 34), 9.15, color=lc, fill=False, linewidth=lw)
    ax.add_patch(circle)
    ax.scatter([52.5], [34], color=lc, s=20)

    ax.add_patch(patches.Rectangle((0, 13.84), 16.5, 40.32, linewidth=lw, edgecolor=lc, facecolor='none'))
    ax.add_patch(patches.Rectangle((0, 24.84), 5.5, 18.32, linewidth=lw, edgecolor=lc, facecolor='none'))
    ax.scatter([11], [34], color=lc, s=20)
    ax.add_patch(Arc((11, 34), 18.3, 18.3, angle=0, theta1=308, theta2=52, color=lc, linewidth=lw))

    ax.add_patch(patches.Rectangle((88.5, 13.84), 16.5, 40.32, linewidth=lw, edgecolor=lc, facecolor='none'))
    ax.add_patch(patches.Rectangle((99.5, 24.84), 5.5, 18.32, linewidth=lw, edgecolor=lc, facecolor='none'))
    ax.scatter([94], [34], color=lc, s=20)
    ax.add_patch(Arc((94, 34), 18.3, 18.3, angle=0, theta1=128, theta2=232, color=lc, linewidth=lw))

    ax.add_patch(patches.Rectangle((-2, 30.34), 2, 7.32, linewidth=lw, edgecolor=lc, facecolor='none'))
    ax.add_patch(patches.Rectangle((105, 30.34), 2, 7.32, linewidth=lw, edgecolor=lc, facecolor='none'))

    return ax


def render():
    st.header("🗺️ Mapa Táctico")

    if not st.session_state.get("analysis_done"):
        st.info("⚠️ Primero realiza un análisis en la sección 'Subir y Analizar Vídeo'")
        return

    results = st.session_state.get("mock_results", {})
    player = results.get("player_name", "Jugador")

    viz_type = st.selectbox("Tipo de visualización", [
        "Heatmap de posiciones",
        "Mapa de pases",
        "Mapa de pases progresivos",
        "Zonas de recuperación",
        "Zonas de pérdida",
        "Tiros"
    ])

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("Opciones")
        show_zones = st.checkbox("Mostrar zonas del campo", value=True)

    with col1:
        np.random.seed(42)
        fig, ax = plt.subplots(figsize=(12, 8))
        draw_pitch(ax, color='#2d5a27', line_color='white')

        if viz_type == "Heatmap de posiciones":
            x_pos = np.clip(np.random.normal(65, 15, 200), 0, 105)
            y_pos = np.clip(np.random.normal(34, 12, 200), 0, 68)
            ax.hexbin(x_pos, y_pos, gridsize=20, cmap='YlOrRd', alpha=0.6, extent=[0, 105, 0, 68])
            ax.set_title(f"Heatmap de posiciones — {player}", color='white', fontsize=14, pad=10)

        elif viz_type in ["Mapa de pases", "Mapa de pases progresivos"]:
            n = 28 if viz_type == "Mapa de pases" else 8
            x_start = np.random.normal(60, 12, n)
            y_start = np.random.normal(34, 10, n)
            x_end = x_start + np.random.normal(8, 5, n)
            y_end = y_start + np.random.normal(0, 6, n)
            color = '#FFD700' if viz_type == "Mapa de pases progresivos" else 'lightblue'
            for i in range(n):
                ax.annotate("", xy=(x_end[i], y_end[i]), xytext=(x_start[i], y_start[i]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
            ax.scatter(x_start, y_start, color='white', s=30, zorder=5)
            ax.set_title(f"{viz_type} — {player}", color='white', fontsize=14, pad=10)

        elif viz_type == "Tiros":
            shot_x = np.random.uniform(80, 103, results["shots"])
            shot_y = np.random.uniform(20, 48, results["shots"])
            ax.scatter(shot_x, shot_y, color='red', s=200, zorder=5, marker='*')
            ax.set_title(f"Tiros — {player}", color='white', fontsize=14, pad=10)

        elif viz_type in ["Zonas de recuperación", "Zonas de pérdida"]:
            color = '#00FF00' if "recuperación" in viz_type.lower() else '#FF4444'
            n_events = results["recoveries"] if "recuperación" in viz_type.lower() else results["losses"]
            x_ev = np.random.normal(55, 15, n_events)
            y_ev = np.random.normal(34, 12, n_events)
            ax.scatter(x_ev, y_ev, color=color, s=150, zorder=5, edgecolors='white', linewidth=1.5)
            ax.set_title(f"{viz_type} — {player}", color='white', fontsize=14, pad=10)

        if show_zones:
            for x in [35, 70]:
                ax.axvline(x, color='white', alpha=0.3, linestyle='--', linewidth=1)
            ax.text(17.5, 66, 'Zona defensiva', color='white', alpha=0.5, ha='center', fontsize=9)
            ax.text(52.5, 66, 'Zona media', color='white', alpha=0.5, ha='center', fontsize=9)
            ax.text(87.5, 66, 'Zona ofensiva', color='white', alpha=0.5, ha='center', fontsize=9)

        fig.patch.set_facecolor('#1a1a2e')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        buf.seek(0)
        st.image(buf, use_column_width=True)
        plt.close()