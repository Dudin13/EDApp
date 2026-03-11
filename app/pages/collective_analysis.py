import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc, Circle
import io

def draw_pitch(ax, color='#2d5a27', line_color='white'):
    ax.set_facecolor(color)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 68)
    ax.set_aspect('equal')
    ax.axis('off')
    lw = 1.5
    lc = line_color
    ax.add_patch(patches.Rectangle((0, 0), 105, 68, linewidth=lw, edgecolor=lc, facecolor='none'))
    ax.plot([52.5, 52.5], [0, 68], color=lc, linewidth=lw)
    ax.add_patch(Circle((52.5, 34), 9.15, color=lc, fill=False, linewidth=lw))
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
    st.header("👥 Análisis Colectivo")

    if not st.session_state.get("analysis_done"):
        st.info("⚠️ Primero realiza un análisis en la sección 'Subir y Analizar Vídeo'")
        return

    config = st.session_state.get("analysis_config", {})
    team = config.get("team", "Equipo cedido")
    rival = config.get("rival", "Rival")

    # Selector de equipo
    equipo_sel = st.radio("Equipo a analizar", [team, rival, "Comparativa"], horizontal=True)

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Balón Parado",
        "⚡ Presión Colectiva",
        "⚔️ Patrones de Ataque",
        "🛡️ Análisis Defensivo"
    ])

    # ── TAB 1: BALÓN PARADO ──────────────────────────────────────────
    with tab1:
        st.subheader("🎯 Jugadas a Balón Parado")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Córners a favor", "7")
        col2.metric("Saques de puerta", "12")
        col3.metric("Faltas directas", "5")
        col4.metric("Remates de balón parado", "4")

        st.markdown("---")

        tipo_bp = st.selectbox("Tipo de jugada", [
            "Córners — Zonas de remate",
            "Córners — Trayectorias",
            "Saques de puerta — Zonas destino",
            "Faltas — Tiros directos",
            "Faltas — Jugadas ensayadas"
        ])

        np.random.seed(10)
        fig, ax = plt.subplots(figsize=(12, 8))
        draw_pitch(ax)

        if "Córners" in tipo_bp and "Zonas" in tipo_bp:
            # Zonas de remate en córners
            x_rem = np.random.normal(90, 6, 15)
            y_rem = np.random.normal(34, 10, 15)
            x_rem = np.clip(x_rem, 75, 105)
            y_rem = np.clip(y_rem, 10, 58)
            sc = ax.scatter(x_rem, y_rem, c=np.random.uniform(0, 1, 15),
                           cmap='YlOrRd', s=200, zorder=5, edgecolors='white', linewidth=1.5)
            ax.set_title("Zonas de remate en córners", color='white', fontsize=14, pad=10)

        elif "Córners" in tipo_bp and "Trayectorias" in tipo_bp:
            # Trayectorias de córners
            corners_right = [(105, 0), (105, 68)]
            corners_left = [(0, 0), (0, 68)]
            for _ in range(4):
                start = corners_right[np.random.randint(0, 2)]
                end_x = np.random.uniform(82, 100)
                end_y = np.random.uniform(20, 48)
                ax.annotate("", xy=(end_x, end_y), xytext=start,
                    arrowprops=dict(arrowstyle='->', color='#FFD700', lw=2,
                                   connectionstyle='arc3,rad=0.3'))
            for _ in range(3):
                start = corners_left[np.random.randint(0, 2)]
                end_x = np.random.uniform(5, 23)
                end_y = np.random.uniform(20, 48)
                ax.annotate("", xy=(end_x, end_y), xytext=start,
                    arrowprops=dict(arrowstyle='->', color='lightblue', lw=2,
                                   connectionstyle='arc3,rad=-0.3'))
            ax.set_title("Trayectorias de córners (Amarillo=derecha, Azul=izquierda)",
                        color='white', fontsize=12, pad=10)

        elif "Saques" in tipo_bp:
            # Zonas destino saques de puerta
            x_dest = np.random.normal(55, 20, 12)
            y_dest = np.random.normal(34, 15, 12)
            x_dest = np.clip(x_dest, 5, 100)
            y_dest = np.clip(y_dest, 5, 63)
            for i in range(12):
                ax.annotate("", xy=(x_dest[i], y_dest[i]), xytext=(5, 34),
                    arrowprops=dict(arrowstyle='->', color='#FFD700', lw=1.5, alpha=0.7))
            ax.scatter(x_dest, y_dest, color='#FFD700', s=100, zorder=5, edgecolors='white')
            ax.set_title("Zonas destino — Saques de puerta", color='white', fontsize=14, pad=10)

        elif "Faltas" in tipo_bp:
            # Posiciones de faltas
            x_falt = np.random.uniform(55, 90, 5)
            y_falt = np.random.uniform(15, 53, 5)
            ax.scatter(x_falt, y_falt, color='red', s=200, zorder=5,
                      marker='x', linewidths=3)
            for i, (x, y) in enumerate(zip(x_falt, y_falt)):
                ax.annotate(f"F{i+1}", (x+1, y+1), color='white', fontsize=10)
            ax.set_title("Posiciones de faltas", color='white', fontsize=14, pad=10)

        fig.patch.set_facecolor('#1a1a2e')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        buf.seek(0)
        plt.close()
        st.image(buf, use_column_width=True)

    # ── TAB 2: PRESIÓN COLECTIVA ─────────────────────────────────────
    with tab2:
        st.subheader("⚡ Presión Colectiva del Equipo")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("PPDA", "8.4", "-1.2 vs rival")
        col2.metric("Presiones totales", "87")
        col3.metric("Presiones exitosas", "31", "35.6%")
        col4.metric("Altura media presión", "58m")

        st.markdown("---")

        col_iz, col_der = st.columns(2)

        with col_iz:
            st.markdown("**Zonas de presión**")
            np.random.seed(20)
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            draw_pitch(ax2)
            x_press = np.random.normal(65, 18, 87)
            y_press = np.random.normal(34, 14, 87)
            x_press = np.clip(x_press, 0, 105)
            y_press = np.clip(y_press, 0, 68)
            ax2.hexbin(x_press, y_press, gridsize=15, cmap='Reds',
                      alpha=0.7, extent=[0, 105, 0, 68])
            ax2.set_title("Heatmap de presiones", color='white', fontsize=12)
            fig2.patch.set_facecolor('#1a1a2e')
            buf2 = io.BytesIO()
            plt.savefig(buf2, format='png', dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
            buf2.seek(0)
            plt.close()
            st.image(buf2, use_column_width=True)

        with col_der:
            st.markdown("**PPDA por período**")
            periodos = ['1-15', '16-30', '31-45', '46-60', '61-75', '76-90']
            ppda_vals = [6.2, 7.8, 9.1, 8.4, 10.2, 7.6]
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            fig3.patch.set_facecolor('#1a1a2e')
            ax3.set_facecolor('#1a1a2e')
            ax3.plot(periodos, ppda_vals, 'o-', color='#FFD700', linewidth=2, markersize=8)
            ax3.axhline(y=9, color='red', linestyle='--', alpha=0.5, label='Límite alto (9)')
            ax3.set_xlabel('Período', color='white')
            ax3.set_ylabel('PPDA', color='white')
            ax3.set_title('PPDA por período\n(menor = más presión)', color='white', fontsize=12)
            ax3.tick_params(colors='white')
            ax3.spines['bottom'].set_color('gray')
            ax3.spines['left'].set_color('gray')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.legend(facecolor='#1a1a2e', labelcolor='white')
            buf3 = io.BytesIO()
            plt.savefig(buf3, format='png', dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
            buf3.seek(0)
            plt.close()
            st.image(buf3, use_column_width=True)

    # ── TAB 3: PATRONES DE ATAQUE ────────────────────────────────────
    with tab3:
        st.subheader("⚔️ Patrones de Ataque")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Ataques totales", "34")
        col2.metric("Ataques por banda derecha", "42%")
        col3.metric("Ataques por banda izquierda", "31%")
        col4.metric("Ataques por centro", "27%")

        st.markdown("---")

        patron = st.selectbox("Visualización", [
            "Flujo de ataque",
            "Transiciones ofensivas",
            "Combinaciones frecuentes",
            "Zonas de finalización"
        ])

        np.random.seed(30)
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        draw_pitch(ax4)

        if patron == "Flujo de ataque":
            # Flechas de progresión del balón
            for _ in range(20):
                x_start = np.random.uniform(20, 60)
                y_start = np.random.uniform(10, 58)
                x_end = x_start + np.random.uniform(10, 25)
                y_end = y_start + np.random.uniform(-8, 8)
                x_end = min(x_end, 104)
                color = '#FFD700' if x_end > 80 else 'lightblue'
                ax4.annotate("", xy=(x_end, y_end), xytext=(x_start, y_start),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.8))
            ax4.set_title("Flujo de ataque", color='white', fontsize=14, pad=10)

        elif patron == "Transiciones ofensivas":
            # Punto de recuperación → finalización
            for _ in range(10):
                x_rec = np.random.uniform(30, 65)
                y_rec = np.random.uniform(10, 58)
                x_fin = np.random.uniform(82, 103)
                y_fin = np.random.uniform(20, 48)
                ax4.annotate("", xy=(x_fin, y_fin), xytext=(x_rec, y_rec),
                    arrowprops=dict(arrowstyle='->', color='#00FF00', lw=2,
                                   connectionstyle='arc3,rad=0.1'))
                ax4.scatter([x_rec], [y_rec], color='yellow', s=100, zorder=5)
                ax4.scatter([x_fin], [y_fin], color='red', s=150, zorder=5, marker='*')
            ax4.set_title("Transiciones ofensivas (Amarillo=recuperacion, Rojo=finalizacion)",
                        color='white', fontsize=11, pad=10)

        elif patron == "Zonas de finalización":
            x_fin = np.random.normal(91, 7, 34)
            y_fin = np.random.normal(34, 10, 34)
            x_fin = np.clip(x_fin, 75, 104)
            y_fin = np.clip(y_fin, 10, 58)
            ax4.hexbin(x_fin, y_fin, gridsize=12, cmap='YlOrRd',
                      alpha=0.7, extent=[75, 105, 0, 68])
            ax4.set_title("Zonas de finalización", color='white', fontsize=14, pad=10)

        fig4.patch.set_facecolor('#1a1a2e')
        buf4 = io.BytesIO()
        plt.savefig(buf4, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        buf4.seek(0)
        plt.close()
        st.image(buf4, use_column_width=True)

    # ── TAB 4: ANÁLISIS DEFENSIVO ────────────────────────────────────
    with tab4:
        st.subheader("🛡️ Análisis Defensivo")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Altura línea defensiva", "42m")
        col2.metric("Duelos defensivos", "38")
        col3.metric("Intercepciones", "14")
        col4.metric("Despejes", "9")

        st.markdown("---")

        viz_def = st.selectbox("Visualización defensiva", [
            "Línea defensiva media",
            "Zonas de duelos defensivos",
            "Coberturas y ayudas",
            "Zonas de intercepciones"
        ])

        np.random.seed(40)
        fig5, ax5 = plt.subplots(figsize=(12, 8))
        draw_pitch(ax5)

        if viz_def == "Línea defensiva media":
            # Línea defensiva
            altura = 42
            ax5.axvline(x=altura, color='#FFD700', linewidth=3, linestyle='-', alpha=0.8)
            ax5.axvline(x=altura-5, color='white', linewidth=1, linestyle='--', alpha=0.4)
            ax5.axvline(x=altura+5, color='white', linewidth=1, linestyle='--', alpha=0.4)
            ax5.text(altura, 70, f'Línea media\n{altura}m', color='#FFD700',
                    ha='center', fontsize=11, va='bottom')
            # Posiciones de defensas
            def_positions = [(altura+np.random.uniform(-3,3), y) for y in [15, 25, 43, 53]]
            for x, y in def_positions:
                ax5.scatter([x], [y], color='#FFD700', s=300, zorder=5,
                           edgecolors='white', linewidth=2)
            ax5.set_title("Altura media de la línea defensiva", color='white', fontsize=14, pad=10)

        elif viz_def == "Zonas de duelos defensivos":
            x_due = np.random.normal(45, 15, 38)
            y_due = np.random.normal(34, 14, 38)
            x_due = np.clip(x_due, 0, 105)
            y_due = np.clip(y_due, 0, 68)
            ax5.hexbin(x_due, y_due, gridsize=15, cmap='Blues',
                      alpha=0.7, extent=[0, 105, 0, 68])
            ax5.set_title("Zonas de duelos defensivos", color='white', fontsize=14, pad=10)

        elif viz_def == "Zonas de intercepciones":
            x_int = np.random.normal(42, 12, 14)
            y_int = np.random.normal(34, 12, 14)
            ax5.scatter(x_int, y_int, color='cyan', s=200, zorder=5,
                       edgecolors='white', linewidth=1.5, marker='D')
            ax5.set_title("Zonas de intercepciones", color='white', fontsize=14, pad=10)

        elif viz_def == "Coberturas y ayudas":
            for _ in range(8):
                x1 = np.random.uniform(25, 60)
                y1 = np.random.uniform(10, 58)
                x2 = x1 + np.random.uniform(-10, 10)
                y2 = y1 + np.random.uniform(-8, 8)
                ax5.plot([x1, x2], [y1, y2], color='cyan', lw=1.5, alpha=0.7)
                ax5.scatter([x1], [y1], color='#FFD700', s=150, zorder=5)
                ax5.scatter([x2], [y2], color='white', s=100, zorder=5)
            ax5.set_title("Coberturas y ayudas defensivas", color='white', fontsize=14, pad=10)

        fig5.patch.set_facecolor('#1a1a2e')
        buf5 = io.BytesIO()
        plt.savefig(buf5, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        buf5.seek(0)
        plt.close()
        st.image(buf5, use_column_width=True)