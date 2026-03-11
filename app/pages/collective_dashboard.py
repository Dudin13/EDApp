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
from mplsoccer import Pitch
import io
from pathlib import Path


# ── Partidos de ejemplo (se van acumulando con análisis reales) ───────────────
PARTIDOS_MOCK = [
    {"id": 1, "local": "Cadiz CF", "visitante": "Real Jaén",      "fecha": "01/03/2026", "comp": "3ª RFEF", "resultado": "2–1"},
    {"id": 2, "local": "Cadiz CF", "visitante": "Lucena CF",      "fecha": "22/02/2026", "comp": "3ª RFEF", "resultado": "0–0"},
    {"id": 3, "local": "Pozoblanco",         "visitante": "Cadiz CF","fecha":"15/02/2026","comp":"3ª RFEF", "resultado": "1–2"},
]


def _metric_row(label, value, unit="", color="#00d4aa"):
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                padding:12px 16px;background:rgba(255,255,255,0.02);border-radius:8px;
                margin-bottom:6px;border-left:2px solid {color}44;">
        <span style="font-size:12px;color:#8899aa;font-weight:600;letter-spacing:0.3px;">{label}</span>
        <span style="font-size:14px;font-weight:800;color:{color};text-shadow:0 0 8px {color}33;">{value}
            <span style="font-size:10px;color:#5a6a7e;font-weight:500;margin-left:2px;">{unit}</span>
        </span>
    </div>
    """, unsafe_allow_html=True)


def render():
    # ── CUSTOM CSS ─────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    }
    .event-icon {
        font-size: 20px;
        background: rgba(255, 255, 255, 0.05);
        width: 42px;
        height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        box-shadow: inset 0 0 10px rgba(255,255,255,0.05);
    }
    .status-tag {
        font-size: 10px;
        font-weight: 800;
        text-transform: uppercase;
        padding: 2px 8px;
        border-radius: 4px;
        margin-left: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── HEADER ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;">
        <div>
            <h2 style="margin:0;font-size:24px;font-weight:800;color:#fff;letter-spacing:-0.5px;">Panel Táctico <span style='color:#00d4aa'>Colectivo</span></h2>
            <p style="margin:2px 0 0;font-size:14px;color:#8899aa;font-weight:500;">Sincronización de eventos IA y métricas avanzadas</p>
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
            <div style="background:{bg_color};backdrop-filter:blur(8px);border:1px solid {border_color};
                        border-radius:12px;padding:16px;cursor:pointer;transition:all 0.3s ease;
                        margin-bottom:8px;box-shadow:0 4px 15px rgba(0,0,0,0.2);">
                <div style="font-size:10px;text-transform:uppercase;letter-spacing:1.5px;
                            color:#8899aa;font-weight:700;margin-bottom:6px;opacity:0.8;">{p['comp']} · {p['fecha']}</div>
                <div style="font-size:14px;font-weight:800;color:{label_color};white-space:nowrap;
                            overflow:hidden;text-overflow:ellipsis;letter-spacing:0.3px;">{p['local']}</div>
                <div style="font-size:11px;color:#5a6a7e;margin:2px 0;font-weight:600;">vs</div>
                <div style="font-size:14px;font-weight:800;color:{label_color};white-space:nowrap;
                            overflow:hidden;text-overflow:ellipsis;letter-spacing:0.3px;">{p['visitante']}</div>
                <div style="font-size:24px;font-weight:900;color:{res_color};margin-top:12px;text-align:center;
                            text-shadow:0 0 10px {res_color}33;">
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
            # Si es el partido analizado, usar datos reales
            is_analysis = (partido_idx == 0 and st.session_state.get("analysis_done"))
            results = st.session_state.get("analysis_results", {})
            b_events = st.session_state.get("ball_events", [])
            
            if is_analysis:
                # Calcular KPIs reales
                count_0 = len([e for e in b_events if e.get("equipo") == 0])
                count_1 = len([e for e in b_events if e.get("equipo") == 1])
                total_c = max(1, count_0 + count_1)
                pos_local = int(count_0 / total_c * 100)
                
                shots = len([e for e in b_events if e.get("action") == "Tiro"])
                res_jugadores = results.get("resultados_jugadores", {})
                passes = sum(p.get("passes", 0) for p in res_jugadores.values())
                recoveries = sum(p.get("recoveries", 0) for p in res_jugadores.values())
                dist_total = sum(p.get("distance_km", 0) for p in res_jugadores.values())
                
                st.markdown("<br>", unsafe_allow_html=True)
                k1, k2, k3 = st.columns(3)
                k1.metric("Posesión", f"{pos_local}%", f"{pos_local-50}%")
                k2.metric("Tiros totales", shots, "")
                k3.metric("Recuperaciones", recoveries, "")
                k4, k5, k6 = st.columns(3)
                k4.metric("Pases (detección)", passes, "")
                k5.metric("Distancia Equipo", f"{dist_total:.1f} km", "")
                k6.metric("Detecciones IA", results.get("total_detecciones", 0), "")
            else:
                np.random.seed(partido_idx * 7 + 42)
                # KPI cards 3+3 (Mock)
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
            if is_analysis:
                with col_s1:
                    _metric_row("Eventos analizados", len(b_events))
                    _metric_row("Frames con detección", results.get("frames_analizados", 0))
                with col_s2:
                    n_jug = len(results.get("resultados_jugadores", {}))
                    _metric_row("Distancia media", f"{dist_total/max(1, n_jug):.2f} km")
                    _metric_row("Dorsales identificados", n_jug)
            else:
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

            pitch = Pitch(
                pitch_type='custom', pitch_length=105, pitch_width=68,
                pitch_color='#09090b', line_color='rgba(255,255,255,0.4)',
                linewidth=1.2, spot_scale=0.003
            )
            fig_m, ax_m = pitch.draw(figsize=(11, 7))
            fig_m.patch.set_facecolor('#111827')

            if viz == "Heatmap de posesión":
                hx = st.session_state.get("heatmap_x", [])
                hy = st.session_state.get("heatmap_y", [])
                if hx and is_analysis:
                    x_p = np.array(hx)
                    y_p = np.array(hy)
                else:
                    x_p = np.clip(np.random.normal(62, 18, 400), 0, 105)
                    y_p = np.clip(np.random.normal(34, 14, 400), 0, 68)
                
                pitch.kdeplot(
                    x_p, y_p, ax=ax_m,
                    cmap='YlOrRd', fill=True, n_levels=100,
                    alpha=0.6, cut=10, zorder=2
                )
                ax_m.set_title("Heatmap de posesión de equipo", color='#8899aa', fontsize=10, pad=10)

            elif viz == "Pases":
                pases = [e for e in b_events if e.get("action") == "Pase" and "pitch_pos" in e]
                if pases and is_analysis:
                    for i, p in enumerate(pases[:40]):
                        px, py = p["pitch_pos"]
                        pitch.arrows(px, py, px+3, py, width=1.5,
                                     headwidth=4, headlength=4, color='#00d4aa', ax=ax_m, zorder=2, alpha=0.55)
                    ax_m.set_title(f"Pases detectados · {len(pases)} acciones", color='#8899aa', fontsize=10, pad=10)
                else:
                    n = 35
                    xs = np.random.normal(55, 14, n); ys = np.random.normal(34, 12, n)
                    xe = xs + np.random.normal(8, 6, n); ye = ys + np.random.normal(0, 7, n)
                    for i in range(n):
                        pitch.arrows(xs[i], ys[i], xe[i], ye[i], width=1.5,
                                     headwidth=4, headlength=4, color='#00d4aa', ax=ax_m, zorder=2, alpha=0.55)
                    pitch.scatter(xs, ys, color='white', s=18, zorder=5, alpha=0.7, ax=ax_m)

            elif viz == "Zonas de recuperación":
                recs = [e for e in b_events if e.get("action") == "Recuperación" and "pitch_pos" in e]
                if recs and is_analysis:
                    for r in recs:
                        px, py = r["pitch_pos"]
                        pitch.scatter(px, py, ax=ax_m, color='#00d4aa', s=100, alpha=0.6, edgecolors='white', linewidths=0.5)
                else:
                    n = 30
                    rx = np.clip(np.random.normal(48, 16, n), 5, 100)
                    ry = np.clip(np.random.normal(34, 13, n), 5, 63)
                    pitch.scatter(rx, ry, ax=ax_m, color='#00d4aa', s=100, alpha=0.6, edgecolors='white', linewidths=0.5)

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

            pitch_f = Pitch(
                pitch_type='custom', pitch_length=105, pitch_width=68,
                pitch_color='#09090b', line_color='rgba(255,255,255,0.4)',
                linewidth=1.2, spot_scale=0.003
            )
            fig_f, ax_f = pitch_f.draw(figsize=(10, 7))
            fig_f.patch.set_facecolor('#111827')

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
            start_sec = st.session_state.get("preview_clip_second", 0)
            # st.video con start_time permite saltar a eventos automáticamente
            st.video(video_path, start_time=int(start_sec))
        else:
            p_local = str(partido_sel['local']) if partido_sel else 'Local'
            p_visit = str(partido_sel['visitante']) if partido_sel else 'Visitante'
            st.markdown(f"""
            <div style="background:#0d1220;border:1px solid #1e2a3a;border-radius:12px;
                        aspect-ratio:16/9;display:flex;flex-direction:column;align-items:center;
                        justify-content:center;gap:10px;padding:20px;margin-bottom:8px;">
                <div style="width:56px;height:56px;background:#1e2a3a;border-radius:50%;
                            display:flex;align-items:center;justify-content:center;font-size:24px;">
                    🎬
                </div>
                <div style="font-size:13px;font-weight:600;color:#5a6a7e;text-align:center;">
                    {p_local} vs {p_visit}<br>
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

        # ── Event Timeline (Pro Dashboard) ────────────────────────────────────
        st.markdown('<div class="ws-section-header">Línea de Eventos IA</div>', unsafe_allow_html=True)

        results = st.session_state.get("analysis_results", {})
        ball_events = results.get("ball_events", [])
        
        team_local = partido_sel["local"] if partido_sel else "Local"
        team_visit = partido_sel["visitante"] if partido_sel else "Visitante"

        # Filtro de búsqueda (Open-Vocabulary placeholder)
        search_query = st.text_input("🔍 Buscar evento (ej: 'tiro', 'Sánchez', 'local')", placeholder="Ej: pase, tiro, nombre de jugador...", label_visibility="collapsed")

        # Clips del análisis real o mock
        if ball_events and partido_idx == 0 and st.session_state.get("analysis_done"):
            clips_data = ball_events
            real_clips = True
        else:
            # Clips de ejemplo si no hay análisis
            np.random.seed(partido_idx * 5)
            acciones_mock = ["Gol", "Tiro", "Pase", "Falta", "Córner", "Recuperación"]
            clips_data = [
                {
                    "nombre_jugador": np.random.choice(["Sánchez", "Delgado", "Ruiz", "Vega", "Morales"]),
                    "nombre_equipo": np.random.choice([team_local, team_visit]),
                    "minute": np.random.randint(5, 88),
                    "action": np.random.choice(acciones_mock),
                    "video_second": float(np.random.randint(60, 5400)),
                }
                for _ in range(12)
            ]
            real_clips = False

        # Aplicar filtro inteligente
        if search_query:
            q = search_query.lower()
            clips_data = [c for c in clips_data if (
                q in str(c.get("nombre_jugador", "")).lower() or 
                q in str(c.get("action", "")).lower() or 
                q in str(c.get("nombre_equipo", "")).lower() or
                (q.isdigit() and int(q) == int(c.get("minute", -1)))
            )]

        ACTION_ICON = {
            "Gol": "⚽", "Tiro": "🥅", "Pase Largo": "🎯", "Pase Corto": "🏹", "Pase": "🎯",
            "Falta": "🟥", "Córner": "🚩", "Recuperación": "💪", "Duelo ganado": "💪",
            "Conducción": "🏃", "Parada/Despeje": "👐"
        }

        # Contenedor de scroll para el Timeline
        timeline_html = ""
        for i, clip in enumerate(clips_data):
            nombre = clip.get("nombre_jugador", "Jugador")
            equipo = clip.get("nombre_equipo", "—")
            minuto = clip.get("minute", 0)
            accion = clip.get("action", clip.get("accion", "Acción"))
            seg = clip.get("video_second", 0)
            eq_color = "#00d4aa" if equipo == team_local else "#4d9fff"
            icon = ACTION_ICON.get(accion, "⚽")

            # Tarjeta de evento premium con Glassmorphism
            st.markdown(f"""
            <div class="glass-card" style="display:flex;align-items:center;gap:14px;margin-bottom:10px;border-left:4px solid {eq_color};">
                <div class="event-icon">{icon}</div>
                <div style="flex:1;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-size:14px;font-weight:800;color:#fff;letter-spacing:0.2px;">{nombre}</span>
                        <div style="display:flex;align-items:center;gap:6px;">
                            <span style="font-size:12px;font-weight:900;color:{eq_color};background:{eq_color}15;padding:2px 6px;border-radius:4px;">{int(minuto)}'</span>
                        </div>
                    </div>
                    <div style="display:flex;align-items:center;margin-top:4px;gap:8px;">
                        <span style="font-size:11px;color:#8899aa;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;">{accion}</span>
                        <span style="width:3px;height:3px;background:#3a4a5e;border-radius:50%;"></span>
                        <span style="font-size:11px;color:#5a6a7e;font-weight:500;">{equipo}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Botones de acción debajo de la tarjeta para Streamlit
            c1, c2, c3 = st.columns([1,1,2])
            with c1:
                if st.button("👁️ Ir", key=f"jump_{i}", use_container_width=True):
                    st.session_state["preview_clip_second"] = seg
            with c2:
                if st.button("✂️ Cortar", key=f"cut_{i}", use_container_width=True):
                    st.toast(f"Generando clip de {accion}...", icon="✂️")
            st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)

        if not clips_data:
            st.info("No se han encontrado eventos para esta búsqueda.")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Panel de Clips Maestro →", use_container_width=True):
            st.session_state["page"] = "partido_clips"
            st.rerun()
