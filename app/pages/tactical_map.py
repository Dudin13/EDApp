import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import io


def render():
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;">
        <div>
            <h2 style="margin:0;font-size:20px;font-weight:700;color:#fff;">Mapa Táctico</h2>
            <p style="margin:2px 0 0;font-size:13px;color:#5a6a7e;">Visualización de posiciones, pases y acciones en el campo</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.get("analysis_done"):
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e2a3a;border-radius:12px;padding:32px;text-align:center;">
            <div style="font-size:36px;margin-bottom:12px;">🗺️</div>
            <div style="font-size:16px;font-weight:600;color:#fff;margin-bottom:8px;">Sin datos de análisis</div>
            <div style="font-size:13px;color:#5a6a7e;">Primero realiza un análisis en "Análisis de Vídeo"</div>
        </div>
        """, unsafe_allow_html=True)
        return

    results = st.session_state.get("mock_results", {})

    # ── Controles ─────────────────────────────────────────────────────────────
    col_ctrl, col_map = st.columns([1, 3])

    with col_ctrl:
        st.markdown('<div class="ws-section-header" style="margin-top:0">Visualización</div>', unsafe_allow_html=True)
        viz_type = st.radio("Tipo", [
            "Heatmap",
            "Mapa de pases",
            "Pases progresivos",
            "Zonas de recuperación",
            "Zonas de pérdida",
            "Tiros"
        ], label_visibility="collapsed")

        st.markdown('<div class="ws-section-header">Opciones</div>', unsafe_allow_html=True)
        show_zones = st.checkbox("Mostrar zonas del campo", value=True)
        show_grid = st.checkbox("Cuadrícula de referencia", value=False)

        # Leyenda de colores
        color_map = {
            "Heatmap": "#ff6b35",
            "Mapa de pases": "#00d4aa",
            "Pases progresivos": "#FFD700",
            "Zonas de recuperación": "#00d4aa",
            "Zonas de pérdida": "#ff4d6d",
            "Tiros": "#ff4d6d",
        }
        accent = color_map.get(viz_type, "#00d4aa")
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1e2a3a;border-radius:8px;padding:12px;margin-top:8px;">
            <div style="font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#5a6a7e;font-weight:600;margin-bottom:8px;">Leyenda</div>
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                <div style="width:12px;height:12px;border-radius:3px;background:{accent};"></div>
                <span style="font-size:12px;color:#8899aa;">{viz_type}</span>
            </div>
            <div style="display:flex;align-items:center;gap:8px;">
                <div style="width:12px;height:2px;background:#5a6a7e;opacity:0.5;"></div>
                <span style="font-size:12px;color:#8899aa;">Líneas de campo</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_map:
        np.random.seed(42)
        
        # Construir campo con mplsoccer
        pitch = Pitch(
            pitch_type='custom', pitch_length=105, pitch_width=68,
            pitch_color='#09090b', line_color='rgba(255,255,255,0.4)',
            linewidth=1.2, spot_scale=0.003
        )
        fig, ax = pitch.draw(figsize=(13, 8.5))
        fig.patch.set_facecolor('#111827')

        if viz_type == "Heatmap":
            hx = st.session_state.get("heatmap_x", [])
            hy = st.session_state.get("heatmap_y", [])
            if hx and len(hx) > 5:
                x_pos = np.array(hx)
                y_pos = np.array(hy)
                data_src = f"{len(x_pos):,} detecciones reales"
            else:
                x_pos = np.clip(np.random.normal(60, 18, 500), 0, 105)
                y_pos = np.clip(np.random.normal(34, 14, 500), 0, 68)
                data_src = "datos anónimos base"
            
            # Heatmap hexbin nativo de mplsoccer o matplotlib
            pitch.kdeplot(
                x_pos, y_pos, ax=ax,
                cmap='YlOrRd', fill=True, n_levels=100,
                alpha=0.6, cut=10, zorder=2
            )
            ax.set_title(f"Heatmap de posiciones · {data_src}", color='#8899aa', fontsize=11, pad=12)

        elif viz_type in ["Mapa de pases", "Pases progresivos", "Tiros", "Zonas de recuperación", "Zonas de pérdida"]:
            events = st.session_state.get("ball_events", [])
            
            # Mapeo de tipos de visualización a acciones del modelo
            action_map = {
                "Mapa de pases": "Pase",
                "Pases progresivos": "Pase",
                "Tiros": "Tiro",
                "Zonas de recuperación": "Recuperación",
                "Zonas de pérdida": "Pérdida"
            }
            target_action = action_map.get(viz_type)
            
            # Filtrar eventos que tengan 'pitch_pos' (nueva versión)
            real_events = [e for e in events if e.get("action") == target_action and "pitch_pos" in e]
            
            if real_events:
                for ev in real_events:
                    px, py = ev["pitch_pos"]
                    
                    # Determinar color según equipo detectado
                    eq_idx = ev.get("equipo", -1)
                    if eq_idx == 0: ev_color = "#ffdd00" # Amarillo base
                    elif eq_idx == 1: ev_color = "#ffffff" # Blanco base
                    else: ev_color = accent
                    
                    if viz_type == "Tiros":
                        pitch.scatter(px, py, ax=ax, color='#ff4d6d', s=200, zorder=5, marker='*', edgecolors='white', linewidths=0.5)
                    elif "Zonas" in viz_type:
                        col = '#00d4aa' if "recuperación" in viz_type.lower() else '#ff4d6d'
                        pitch.scatter(px, py, ax=ax, color=col, s=160, zorder=5, edgecolors='white', linewidths=0.8, alpha=0.85)
                    else:
                        # Para pases, flecha indicativa
                        pitch.arrows(px, py, px+5, py, width=1.8,
                                    headwidth=4, headlength=4, color=ev_color, ax=ax, zorder=2, alpha=0.8)
                        pitch.scatter(px, py, color='white', s=30, zorder=5, alpha=1.0, ax=ax, edgecolors=ev_color)
                ax.set_title(f"{viz_type} · {len(real_events)} acciones reales", color='#8899aa', fontsize=11, pad=12)
            else:
                # Fallback a datos aleatorios si no hay eventos reales
                n = 15
                x_ev = np.random.normal(52, 18, n); y_ev = np.random.normal(34, 13, n)
                pitch.scatter(np.clip(x_ev, 0, 105), np.clip(y_ev, 0, 68), ax=ax, color=accent, s=100, alpha=0.3)
                ax.set_title(f"{viz_type} · Sin datos reales (Vista previa)", color='#8899aa', fontsize=11, pad=12)

        # Zonas verticales opcionales
        if show_zones:
            for x in [35, 70]:
                ax.axvline(x, color='white', alpha=0.15, linestyle='--', lw=1)
            for txt, xc in [('Defensiva', 17.5), ('Mediocampo', 52.5), ('Ofensiva', 87.5)]:
                ax.text(xc, 66.5, txt, color='white', alpha=0.35, ha='center', va='center', fontsize=9, style='italic')

        if show_grid:
            ax.grid(color='white', alpha=0.08, linestyle=':', lw=0.8)

        plt.tight_layout(pad=0.5)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=160, bbox_inches='tight', facecolor='#0e1420')
        buf.seek(0)
        plt.close()
        st.image(buf, use_container_width=True)