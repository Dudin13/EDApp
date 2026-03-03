"""
collective_dashboard.py — Dashboard colectivo estilo Wyscout.

Layout:
  ┌─ Top: Selector de partido ─────────────────────────────────────┐
  ├─ Left (2/3): Métricas colectivas ──┬─ Right (1/3): Vídeo ──────┤
  │  Tabs: KPIs | Mapa | Formación     │  [Video player]           │
  │                                    ├───────────────────────────┤
  │                                    │  Clips de acción          │
  └────────────────────────────────────┴───────────────────────────┘
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc, Circle
import io
from pathlib import Path


# ── Partidos de ejemplo (se van acumulando con análisis reales) ───────────────
PARTIDOS_MOCK = [
    {"id": 1, "local": "ED Atl. Sanluqueño", "visitante": "Real Jaén",      "fecha": "01/03/2026", "comp": "3ª RFEF", "resultado": "2–1"},
    {"id": 2, "local": "ED Atl. Sanluqueño", "visitante": "Lucena CF",      "fecha": "22/02/2026", "comp": "3ª RFEF", "resultado": "0–0"},
    {"id": 3, "local": "Pozoblanco",         "visitante": "ED Atl. Sanluqueño","fecha":"15/02/2026","comp":"3ª RFEF", "resultado": "1–2"},
]


def _draw_pitch_mini(ax, bg='#0a3d1f', lc=(1,1,1,0.75)):
    ax.set_facecolor(bg)
    ax.set_xlim(0, 105); ax.set_ylim(0, 68)
    ax.set_aspect('equal'); ax.axis('off')
    lw = 1.2
    ax.add_patch(patches.Rectangle((0,0), 105, 68, lw=lw, edgecolor=lc, facecolor='none'))
    ax.plot([52.5,52.5],[0,68], color=lc, lw=lw)
    ax.add_patch(Circle((52.5,34), 9.15, color=lc, fill=False, lw=lw))
    ax.scatter([52.5],[34], color=lc, s=10)
    ax.add_patch(patches.Rectangle((0, 13.84), 16.5, 40.32, lw=lw, edgecolor=lc, facecolor='none'))
    ax.add_patch(patches.Rectangle((88.5, 13.84), 16.5, 40.32, lw=lw, edgecolor=lc, facecolor='none'))
    # Bandas de césped
    for x in range(0, 106, 10):
        ax.add_patch(patches.Rectangle((x,0), 5, 68, facecolor='#0d4a25', edgecolor='none', alpha=0.25))
    return ax


def _metric_row(label, value, unit="", color="#00d4aa"):
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                padding:9px 12px;border-bottom:1px solid #1e2a3a;">
        <span style="font-size:12px;color:#8899aa;">{label}</span>
        <span style="font-size:13px;font-weight:700;color:{color};">{value}
            <span style="font-size:10px;color:#5a6a7e;font-weight:400;margin-left:2px;">{unit}</span>
        </span>
    </div>
    """, unsafe_allow_html=True)


def render():
    # ── HEADER ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;">
        <div>
            <h2 style="margin:0;font-size:20px;font-weight:700;color:#fff;">Datos Colectivos</h2>
            <p style="margin:2px 0 0;font-size:13px;color:#5a6a7e;">Análisis táctico y métricas de equipo por partido</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SELECTOR DE PARTIDO (top bar) ─────────────────────────────────────────
    # Combinar partidos mock con el partido real analizado (si existe)
    partidos = list(PARTIDOS_MOCK)
    analysis_cfg = st.session_state.get("analysis_config")
    if analysis_cfg and st.session_state.get("analysis_done"):
        partidos.insert(0, {
            "id": 0,
            "local": analysis_cfg.get("team", "Local"),
            "visitante": analysis_cfg.get("rival", "Visitante"),
            "fecha": str(analysis_cfg.get("match_date", "")),
            "comp": analysis_cfg.get("competition", ""),
            "resultado": "—",
        })

    st.markdown('<div class="ws-section-header" style="margin-top:0">Partidos</div>', unsafe_allow_html=True)

    partido_labels = [f"{p['local']} vs {p['visitante']}  ·  {p['fecha']}" for p in partidos]
    partido_idx = st.session_state.get("col_partido_idx", 0)

    # Tarjetas horizontales de selección de partido
    partido_cols = st.columns(min(len(partidos), 4))
    for i, (col, p) in enumerate(zip(partido_cols, partidos)):
        with col:
            is_sel = partido_idx == i
            border_color = "#00d4aa" if is_sel else "#1e2a3a"
            bg_color = "rgba(0,212,170,0.08)" if is_sel else "#111827"
            label_color = "#00d4aa" if is_sel else "#fff"
            res_color = "#00d4aa" if is_sel else "#8899aa"
            st.markdown(f"""
            <div style="background:{bg_color};border:1px solid {border_color};border-radius:10px;
                        padding:12px 14px;cursor:pointer;transition:all 0.2s;margin-bottom:4px;">
                <div style="font-size:10px;text-transform:uppercase;letter-spacing:1px;
                            color:#5a6a7e;font-weight:600;margin-bottom:4px;">{p['comp']} · {p['fecha']}</div>
                <div style="font-size:13px;font-weight:700;color:{label_color};white-space:nowrap;
                            overflow:hidden;text-overflow:ellipsis;">{p['local']}</div>
                <div style="font-size:11px;color:#5a6a7e;margin:2px 0;">vs</div>
                <div style="font-size:13px;font-weight:700;color:{label_color};white-space:nowrap;
                            overflow:hidden;text-overflow:ellipsis;">{p['visitante']}</div>
                <div style="font-size:20px;font-weight:800;color:{res_color};margin-top:6px;text-align:center;">
                    {p['resultado']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Seleccionar", key=f"sel_partido_{i}", use_container_width=True,
                         type="primary" if is_sel else "secondary"):
                st.session_state["col_partido_idx"] = i
                st.rerun()

    partido_sel = partidos[partido_idx] if partidos else None

    st.markdown("<br>", unsafe_allow_html=True)

    # ── LAYOUT PRINCIPAL: 2 cols (métricas | vídeo+clips) ─────────────────────
    col_left, col_right = st.columns([3, 2], gap="medium")

    # ══════════════════════════════════════════════════════════════════════════
    # COLUMNA IZQUIERDA — MÉTRICAS COLECTIVAS
    # ══════════════════════════════════════════════════════════════════════════
    with col_left:
        tab_kpis, tab_mapa, tab_form = st.tabs([
            "  📊  KPIs Colectivos  ",
            "  🗺️  Mapa Táctico  ",
            "  👥  Formación  ",
        ])

        # ── TAB KPIs ──────────────────────────────────────────────────────────
        with tab_kpis:
            np.random.seed(partido_idx * 7 + 42)

            # KPI cards 3+3
            st.markdown("<br>", unsafe_allow_html=True)
            k1, k2, k3 = st.columns(3)
            k1.metric("Posesión", f"{np.random.randint(42,65)}%", f"{np.random.randint(-5,8)}%")
            k2.metric("Tiros totales", np.random.randint(8,18), np.random.randint(-3,5))
            k3.metric("Tiros a puerta", np.random.randint(3,9), np.random.randint(-2,4))
            k4, k5, k6 = st.columns(3)
            k4.metric("Pases completados", f"{np.random.randint(300,550)}", "")
            k5.metric("Precisión de pase", f"{np.random.randint(72,89)}%", "")
            k6.metric("Duelos ganados", f"{np.random.randint(40,65)}%", "")

            st.markdown('<div class="ws-section-header">Estadísticas avanzadas</div>', unsafe_allow_html=True)

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                _metric_row("xG (goles esperados)", f"{np.random.uniform(0.8,2.5):.2f}")
                _metric_row("xGA (goles esperados concedidos)", f"{np.random.uniform(0.4,1.8):.2f}", color="#ff4d6d")
                _metric_row("Presiones en campo rival", np.random.randint(25,55))
                _metric_row("Presiones exitosas", f"{np.random.randint(30,55)}%", color="#00d4aa")
                _metric_row("Corners a favor", np.random.randint(2,8))
                _metric_row("Faltas recibidas", np.random.randint(8,18))
            with col_s2:
                _metric_row("Recuperaciones", np.random.randint(20,45))
                _metric_row("Pérdidas de balón", np.random.randint(10,25), color="#ff4d6d")
                _metric_row("Línea defensiva media", f"{np.random.randint(35,52)} m")
                _metric_row("Amplitud media ataque", f"{np.random.randint(38,58)} m")
                _metric_row("Sprints del equipo", np.random.randint(80,160))
                _metric_row("Distancia total", f"{np.random.uniform(95,115):.1f} km")

            # Mini radar del equipo
            st.markdown('<div class="ws-section-header">Perfil colectivo</div>', unsafe_allow_html=True)
            cats = ['Posesión', 'Presión', 'Creación', 'Defensa', 'Físico', 'Aéreo']
            vals = [np.random.randint(45, 90) for _ in cats]
            N = len(cats)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            v = vals + vals[:1]

            fig_r, ax_r = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
            fig_r.patch.set_facecolor('#111827')
            ax_r.set_facecolor('#111827')
            ax_r.plot(angles, v, 'o-', lw=2.5, color='#00d4aa')
            ax_r.fill(angles, v, alpha=0.2, color='#00d4aa')
            ax_r.set_xticks(angles[:-1])
            ax_r.set_xticklabels(cats, color='#8899aa', size=9)
            ax_r.set_ylim(0, 100)
            ax_r.set_yticks([25, 50, 75, 100])
            ax_r.set_yticklabels([], size=0)
            ax_r.grid(color='#1e2a3a', alpha=0.8)
            ax_r.spines['polar'].set_color('#1e2a3a')
            buf_r = io.BytesIO()
            plt.savefig(buf_r, format='png', dpi=130, bbox_inches='tight', facecolor='#111827')
            buf_r.seek(0); plt.close()
            st.image(buf_r, use_container_width=True)

        # ── TAB MAPA ──────────────────────────────────────────────────────────
        with tab_mapa:
            np.random.seed(partido_idx * 13)
            viz_opts = ["Heatmap de posesión", "Pases", "Líneas de presión", "Zonas de recuperación"]
            viz = st.radio("Vista", viz_opts, horizontal=True, label_visibility="collapsed")

            fig_m, ax_m = plt.subplots(figsize=(11, 7))
            fig_m.patch.set_facecolor('#0e1420')
            _draw_pitch_mini(ax_m)

            if viz == "Heatmap de posesión":
                x_pos = np.clip(np.random.normal(62, 18, 400), 0, 105)
                y_pos = np.clip(np.random.normal(34, 14, 400), 0, 68)
                ax_m.hexbin(x_pos, y_pos, gridsize=20, cmap='YlOrRd', alpha=0.7, extent=[0,105,0,68])
                ax_m.set_title("Heatmap de posesión de equipo", color='#8899aa', fontsize=10, pad=10)

            elif viz == "Pases":
                n = 35
                xs = np.random.normal(55, 14, n); ys = np.random.normal(34, 12, n)
                xe = xs + np.random.normal(8, 6, n); ye = ys + np.random.normal(0, 7, n)
                for i in range(n):
                    ax_m.annotate("", xy=(xe[i], ye[i]), xytext=(xs[i], ys[i]),
                                  arrowprops=dict(arrowstyle='->', color='#00d4aa', lw=1.4, alpha=0.55))
                ax_m.scatter(xs, ys, color='white', s=18, zorder=5, alpha=0.7)

            elif viz == "Líneas de presión":
                for y_line in [38, 52, 65]:
                    ax_m.axvline(y_line, color='#ff6b35', alpha=0.35, lw=2, linestyle='--')
                ax_m.text(38, 67, 'Bloque bajo', color='#ff6b35', alpha=0.6, ha='center', fontsize=8)
                ax_m.text(52, 67, 'Bloque medio', color='#ff6b35', alpha=0.6, ha='center', fontsize=8)
                ax_m.text(65, 67, 'Pressing alto', color='#ff6b35', alpha=0.6, ha='center', fontsize=8)
                n = 22
                px = np.random.uniform(35, 95, n); py = np.random.uniform(5, 63, n)
                ax_m.scatter(px, py, color='#ff6b35', s=80, alpha=0.55, edgecolors='white', lw=0.5)

            elif viz == "Zonas de recuperación":
                n = 30
                rx = np.clip(np.random.normal(48, 16, n), 5, 100)
                ry = np.clip(np.random.normal(34, 13, n), 5, 63)
                ax_m.scatter(rx, ry, color='#00d4aa', s=100, alpha=0.6, edgecolors='white', lw=0.5)

            # Zonas
            for x in [35, 70]:
                ax_m.axvline(x, color='white', alpha=0.12, lw=1, ls='--')

            plt.tight_layout(pad=0.3)
            buf_m = io.BytesIO()
            plt.savefig(buf_m, format='png', dpi=150, bbox_inches='tight', facecolor='#0e1420')
            buf_m.seek(0); plt.close()
            st.image(buf_m, use_container_width=True)

        # ── TAB FORMACIÓN ─────────────────────────────────────────────────────
        with tab_form:
            np.random.seed(partido_idx + 99)
            st.markdown("<br>", unsafe_allow_html=True)

            # Formaciones típicas
            FORMACIONES = {
                "4-3-3": [(52.5,5), (15,20),(35,15),(70,15),(90,20),
                           (25,38),(52.5,35),(80,38),
                           (20,58),(52.5,62),(85,58)],
                "4-4-2": [(52.5,5), (15,20),(35,15),(70,15),(90,20),
                           (15,40),(38,40),(67,40),(90,40),
                           (35,60),(70,60)],
                "4-2-3-1": [(52.5,5), (15,20),(35,15),(70,15),(90,20),
                             (32,33),(72,33),
                             (15,52),(52.5,50),(90,52),
                             (52.5,63)],
            }
            form_key = st.selectbox("Formación", list(FORMACIONES.keys()), label_visibility="collapsed")
            positions = FORMACIONES[form_key]

            fig_f, ax_f = plt.subplots(figsize=(10, 7))
            fig_f.patch.set_facecolor('#0e1420')
            _draw_pitch_mini(ax_f)

            dorsales_form = [1, 2, 3, 4, 5, 6, 8, 10, 7, 11, 9]
            nombres_form = ["Reyes", "Fernández", "Castillo", "Navarro", "Herrera",
                            "Blanco", "Morales", "Delgado", "Ruiz", "Vega", "Sánchez"]

            for i, (px, py) in enumerate(positions):
                # Círculo jugador
                circle = plt.Circle((px, py), 3.5, color='#00d4aa', zorder=6)
                ax_f.add_patch(circle)
                ax_f.text(px, py, str(dorsales_form[i] if i < len(dorsales_form) else ""),
                          ha='center', va='center', fontsize=8, fontweight='bold',
                          color='#000', zorder=7)
                ax_f.text(px, py - 6, nombres_form[i] if i < len(nombres_form) else "",
                          ha='center', va='center', fontsize=6.5, color='white',
                          fontweight='500', zorder=7,
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='#111827',
                                    edgecolor='none', alpha=0.7))

            ax_f.set_title(f"Formación {form_key}", color='#8899aa', fontsize=10, pad=10)
            plt.tight_layout(pad=0.3)
            buf_f = io.BytesIO()
            plt.savefig(buf_f, format='png', dpi=150, bbox_inches='tight', facecolor='#0e1420')
            buf_f.seek(0); plt.close()
            st.image(buf_f, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # COLUMNA DERECHA — VÍDEO + CLIPS
    # ══════════════════════════════════════════════════════════════════════════
    with col_right:
        # ── Vídeo ─────────────────────────────────────────────────────────────
        st.markdown('<div class="ws-section-header" style="margin-top:0;">Vídeo del partido</div>', unsafe_allow_html=True)

        video_path = st.session_state.get("video_path", "")
        video_available = video_path and Path(video_path).exists()

        if video_available and partido_idx == 0 and st.session_state.get("analysis_done"):
            st.video(video_path)
        else:
            # Placeholder estilo Wyscout
            st.markdown(f"""
            <div style="background:#0d1220;border:1px solid #1e2a3a;border-radius:12px;
                        aspect-ratio:16/9;display:flex;flex-direction:column;align-items:center;
                        justify-content:center;gap:10px;padding:20px;margin-bottom:8px;">
                <div style="width:56px;height:56px;background:#1e2a3a;border-radius:50%;
                            display:flex;align-items:center;justify-content:center;font-size:24px;">
                    🎬
                </div>
                <div style="font-size:13px;font-weight:600;color:#5a6a7e;text-align:center;">
                    {partido_sel['local'] + ' vs ' + partido_sel['visitante'] if partido_sel else 'Sin partido'}<br>
                    <span style="font-size:11px;font-weight:400;">{partido_sel['fecha'] if partido_sel else ''}</span>
                </div>
                <div style="font-size:11px;color:#3a4a5e;text-align:center;">
                    {'Sube el vídeo en Nuevo Análisis para reproducirlo aquí' if not video_available else 'Selecciona el partido analizado para ver el vídeo'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Controles de vídeo decorativos (estáticos)
        st.markdown("""
        <div style="display:flex;gap:8px;margin-bottom:16px;">
            <div style="flex:1;background:#111827;border:1px solid #1e2a3a;border-radius:8px;
                        padding:8px 12px;font-size:11px;color:#5a6a7e;text-align:center;">
                ⏮ Anterior
            </div>
            <div style="flex:1;background:#00d4aa22;border:1px solid #00d4aa44;border-radius:8px;
                        padding:8px 12px;font-size:11px;color:#00d4aa;text-align:center;font-weight:600;">
                ▶ Reproducir
            </div>
            <div style="flex:1;background:#111827;border:1px solid #1e2a3a;border-radius:8px;
                        padding:8px 12px;font-size:11px;color:#5a6a7e;text-align:center;">
                Siguiente ⏭
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Clips de acción ────────────────────────────────────────────────────
        st.markdown('<div class="ws-section-header">Clips de acción</div>', unsafe_allow_html=True)

        ball_events = st.session_state.get("ball_events", [])
        team_local = partido_sel["local"] if partido_sel else "Local"
        team_visit = partido_sel["visitante"] if partido_sel else "Visitante"

        # Clips del análisis real o mock
        if ball_events and partido_idx == 0 and st.session_state.get("analysis_done"):
            clips_data = ball_events[:8]  # max 8 clips aquí
            real_clips = True
        else:
            # Clips de ejemplo
            np.random.seed(partido_idx * 5)
            acciones_mock = ["Gol", "Tiro a puerta", "Pase clave", "Falta", "Córner",
                             "Recuperación", "Contra", "Penalti"]
            clips_data = [
                {
                    "nombre_jugador": np.random.choice(["Sánchez", "Delgado", "Ruiz", "Vega", "Morales"]),
                    "nombre_equipo": np.random.choice([team_local, team_visit]),
                    "minute": np.random.randint(5, 88),
                    "accion": np.random.choice(acciones_mock),
                    "video_second": float(np.random.randint(60, 5400)),
                }
                for _ in range(6)
            ]
            real_clips = False

        ACTION_ICON = {
            "Gol": "⚽", "Tiro a puerta": "🥅", "Pase clave": "🎯",
            "Falta": "🟥", "Córner": "🚩", "Recuperación": "💪",
            "Contra": "⚡", "Penalti": "🎯", "Acción con balón": "⚽",
        }

        for i, clip in enumerate(clips_data[:8]):
            nombre = clip.get("nombre_jugador", "Jugador")
            equipo = clip.get("nombre_equipo", "—")
            minuto = clip.get("minute", 0)
            accion = clip.get("accion", "Acción con balón")
            seg = clip.get("video_second", 0)
            eq_color = "#00d4aa" if equipo == team_local else "#4d9fff"
            icon = ACTION_ICON.get(accion, "⚽")

            # Mini tarjeta de clip
            col_clip_info, col_clip_btn = st.columns([4, 1])
            with col_clip_info:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:10px;padding:8px 10px;
                            background:#111827;border:1px solid #1e2a3a;border-radius:8px;
                            margin-bottom:4px;transition:border-color 0.15s;">
                    <div style="font-size:18px;">{icon}</div>
                    <div style="flex:1;min-width:0;">
                        <div style="font-size:12px;font-weight:700;color:#fff;
                                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                            {nombre}
                        </div>
                        <div style="font-size:10px;color:#5a6a7e;">
                            <span style="color:{eq_color};font-weight:600;">Min {int(minuto)}'</span>
                            {' · ' + accion if not real_clips else ''}
                        </div>
                    </div>
                    <div style="font-size:11px;font-weight:700;color:{eq_color};
                                background:{eq_color}18;padding:2px 8px;border-radius:10px;
                                white-space:nowrap;">{int(minuto)}'</div>
                </div>
                """, unsafe_allow_html=True)
            with col_clip_btn:
                st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                if st.button("▶", key=f"play_clip_col_{i}", help=f"Ver clip min {int(minuto)}"):
                    if video_available and real_clips:
                        st.session_state["preview_clip_second"] = seg
                    else:
                        st.toast(f"Sube el vídeo del partido para reproducir este clip", icon="ℹ️")

        if not clips_data:
            st.markdown("""
            <div style="text-align:center;padding:20px;color:#5a6a7e;font-size:12px;">
                Sin clips disponibles para este partido.<br>Realiza un análisis para generar clips automáticamente.
            </div>
            """, unsafe_allow_html=True)

        # Ver todos los clips
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Ver todos los clips →", use_container_width=True):
            st.session_state["page"] = "partido_clips"
            st.rerun()
