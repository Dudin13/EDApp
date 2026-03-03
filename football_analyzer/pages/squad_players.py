"""
squad_players.py — Plantilla de 20 jugadores con ficha individual.
Vista de cuadrícula → detalle Wyscout (Info+Videos | Stats+Pitch+Matches).
"""

import streamlit as st
import numpy as np

# ── Datos de los 20 jugadores de la plantilla ─────────────────────────────────
PLANTILLA = [
    {"dorsal": 1,  "nombre": "Alejandro Reyes",   "posicion": "Portero",      "edad": 28, "nacionalidad": "🇪🇸", "pais": "España",     "altura": "189 cm", "pie": "Derecho"},
    {"dorsal": 13, "nombre": "David Molina",       "posicion": "Portero",      "edad": 23, "nacionalidad": "🇪🇸", "pais": "España",     "altura": "186 cm", "pie": "Derecho"},
    {"dorsal": 2,  "nombre": "Carlos Fernández",   "posicion": "Lateral D",    "edad": 26, "nacionalidad": "🇪🇸", "pais": "España",     "altura": "178 cm", "pie": "Derecho"},
    {"dorsal": 3,  "nombre": "Rubén Castillo",     "posicion": "Lateral I",    "edad": 25, "nacionalidad": "🇪🇸", "pais": "España",     "altura": "176 cm", "pie": "Izquierdo"},
    {"dorsal": 4,  "nombre": "Jorge Navarro",      "posicion": "Central",      "edad": 30, "nacionalidad": "🇪🇸", "pais": "España",     "altura": "185 cm", "pie": "Derecho"},
    {"dorsal": 5,  "nombre": "Miguel Herrera",     "posicion": "Central",      "edad": 27, "nacionalidad": "🇪🇸", "pais": "España",     "altura": "183 cm", "pie": "Izquierdo"},
    {"dorsal": 6,  "nombre": "Sergio Blanco",      "posicion": "Pivote",       "edad": 29, "nacionalidad": "🇪🇸", "pais": "España",     "altura": "181 cm", "pie": "Derecho"},
    {"dorsal": 8,  "nombre": "Iván Morales",       "posicion": "Mediocentro",  "edad": 24, "nacionalidad": "🇪🇸", "pais": "España",     "altura": "177 cm", "pie": "Derecho"},
    {"dorsal": 14, "nombre": "Luis García",        "posicion": "Mediocentro",  "edad": 22, "nacionalidad": "🇪🇸", "pais": "España",     "altura": "175 cm", "pie": "Izquierdo"},
    {"dorsal": 15, "nombre": "Adrián Jiménez",     "posicion": "Interior D",   "edad": 25, "nacionalidad": "🇪🇸", "pais": "España",     "altura": "179 cm", "pie": "Derecho"},
    {"dorsal": 16, "nombre": "Pablo Torres",       "posicion": "Interior I",   "edad": 23, "nacionalidad": "🇪🇸", "pais": "España",     "altura": "177 cm", "pie": "Izquierdo"},
    {"dorsal": 10, "nombre": "Marco Delgado",      "posicion": "Mediapunta",   "edad": 26, "nacionalidad": "🇦🇷", "pais": "Argentina", "altura": "173 cm", "pie": "Izquierdo"},
    {"dorsal": 7,  "nombre": "Javier Ruiz",        "posicion": "Extremo D",    "edad": 24, "nacionalidad": "🇧🇷", "pais": "Brasil",    "altura": "172 cm", "pie": "Izquierdo"},
    {"dorsal": 11, "nombre": "Antonio Vega",       "posicion": "Extremo I",    "edad": 22, "nacionalidad": "🇪🇸", "pais": "España",     "altura": "174 cm", "pie": "Izquierdo"},
    {"dorsal": 9,  "nombre": "Raúl Sánchez",       "posicion": "Delantero",    "edad": 28, "nacionalidad": "🇪🇸", "pais": "España",     "altura": "182 cm", "pie": "Derecho"},
    {"dorsal": 19, "nombre": "Fernando López",     "posicion": "Delantero",    "edad": 21, "nacionalidad": "🇪🇸", "pais": "España",     "altura": "180 cm", "pie": "Derecho"},
    {"dorsal": 20, "nombre": "Tomás Ramos",        "posicion": "Extremo D",    "edad": 20, "nacionalidad": "🇸🇳", "pais": "Senegal",   "altura": "181 cm", "pie": "Derecho"},
    {"dorsal": 17, "nombre": "Héctor Peña",        "posicion": "Extremo I",    "edad": 23, "nacionalidad": "🇪🇸", "pais": "España",     "altura": "173 cm", "pie": "Izquierdo"},
    {"dorsal": 18, "nombre": "Óscar Martín",       "posicion": "Interior D",   "edad": 24, "nacionalidad": "🇪🇸", "pais": "España",     "altura": "178 cm", "pie": "Derecho"},
    {"dorsal": 12, "nombre": "Cristian Flores",    "posicion": "Lateral D",    "edad": 22, "nacionalidad": "🇲🇦", "pais": "Marruecos", "altura": "175 cm", "pie": "Derecho"},
]

POSICION_COLOR = {
    "Portero": "#ff6b35", "Lateral D": "#4d9fff", "Lateral I": "#4d9fff",
    "Central": "#4d9fff", "Pivote": "#00d4aa", "Mediocentro": "#00d4aa",
    "Interior D": "#00d4aa", "Interior I": "#00d4aa", "Mediapunta": "#FFD700",
    "Extremo D": "#a78bfa", "Extremo I": "#a78bfa", "Delantero": "#ff4d6d",
}

POS_XY = {
    "Portero": (8,34), "Lateral D": (25,10), "Lateral I": (25,58),
    "Central": (20,28), "Pivote": (38,34), "Mediocentro": (42,34),
    "Interior D": (52,18), "Interior I": (52,50), "Mediapunta": (62,34),
    "Extremo D": (70,10), "Extremo I": (70,58), "Delantero": (85,34),
}


def _generar_stats(jugador):
    np.random.seed(jugador["dorsal"] * 13)
    pos = jugador["posicion"]
    es_portero = pos == "Portero"
    es_defensa = pos in ("Lateral D", "Lateral I", "Central")
    es_delantero = pos in ("Delantero", "Extremo D", "Extremo I")
    partidos = int(np.random.randint(14, 28))
    minutos = int(partidos * np.random.randint(60, 92))
    goles = 0 if es_portero else (int(np.random.randint(8, 18)) if es_delantero else int(np.random.randint(0, 6)))
    asistencias = 0 if es_portero else (int(np.random.randint(3, 10)) if es_delantero else int(np.random.randint(1, 8)))
    pases = int(np.random.randint(20, 60)) if es_portero else int(np.random.randint(30, 85))
    precision_pase = int(np.random.randint(62, 95))
    duelos = int(np.random.randint(15, 45))
    duelos_ganados = int(duelos * np.random.uniform(0.45, 0.70))
    recuperaciones = int(np.random.randint(10, 40)) if es_defensa else int(np.random.randint(5, 25))
    tiros = 0 if es_portero else int(np.random.randint(5, 30))
    tiros_puerta = int(tiros * np.random.uniform(0.3, 0.55))
    distancia = round(float(np.random.uniform(8.5, 11.5)) * (minutos / 90), 1)
    vel_max = round(float(np.random.uniform(24, 34)), 1)
    nota = round(float(np.random.uniform(6.0, 8.5)), 1)
    paradas = int(np.random.randint(20, 70)) if es_portero else 0
    goles_encajados = int(np.random.randint(10, 30)) if es_portero else 0
    return {
        "partidos": partidos, "minutos": minutos, "goles": goles,
        "asistencias": asistencias, "pases_por_partido": pases,
        "precision_pase": precision_pase, "duelos": duelos,
        "duelos_ganados": duelos_ganados, "recuperaciones": recuperaciones,
        "tiros": tiros, "tiros_puerta": tiros_puerta,
        "distancia_km": distancia, "vel_max": vel_max, "nota": nota,
        "paradas": paradas, "goles_encajados": goles_encajados,
    }


def _render_player_detail(jugador):
    """Vista detalle estilo Wyscout: info+videos izq | stats+pitch+partidos der."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Circle
    import pandas as pd
    import io

    stats = _generar_stats(jugador)
    pos = jugador["posicion"]
    color = POSICION_COLOR.get(pos, "#00d4aa")
    es_portero = pos == "Portero"

    # ── Top bar ──────────────────────────────────────────────────────────────
    col_back, col_info = st.columns([1, 6])
    with col_back:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Plantilla", key="back_to_squad"):
            st.session_state["squad_selected_player"] = None
            st.rerun()
    with col_info:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:14px;padding:8px 0 12px;">
            <div style="font-size:32px;font-weight:900;color:{color};">#{jugador['dorsal']}</div>
            <div>
                <div style="font-size:18px;font-weight:800;color:#fff;">{jugador['nombre']}</div>
                <div style="font-size:12px;color:#5a6a7e;">{jugador['nacionalidad']} {jugador['pais']} ·
                    {jugador['edad']} años · {jugador['altura']} · Pie {jugador['pie']}
                </div>
            </div>
            <span style="background:{color}22;border:1px solid {color}55;color:{color};
                         padding:3px 14px;border-radius:16px;font-size:11px;font-weight:700;
                         text-transform:uppercase;letter-spacing:1px;">{pos}</span>
            <span style="margin-left:auto;font-size:11px;color:#5a6a7e;">
                Temporada 2024-25 · {stats['partidos']} partidos · {stats['minutos']:,} min
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_ov, tab_stats, tab_events = st.tabs([
        "  📋  Resumen  ", "  📊  Estadísticas  ", "  🎬  Vídeos  "
    ])

    # ══ TAB RESUMEN (= Overview de Wyscout) ══
    with tab_ov:
        col_l, col_r = st.columns([1, 2], gap="large")

        # LEFT: General Info + Video types
        with col_l:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#0d1f35,#111827);border:1px solid #1e2a3a;
                        border-radius:12px;padding:20px;text-align:center;margin-bottom:14px;">
                <div style="width:76px;height:76px;border-radius:50%;background:{color}22;
                            border:3px solid {color}55;margin:0 auto 12px;display:flex;
                            align-items:center;justify-content:center;">
                    <span style="font-size:30px;font-weight:900;color:{color};">{jugador['dorsal']}</span>
                </div>
                <div style="font-size:15px;font-weight:700;color:#fff;">{jugador['nombre']}</div>
                <div style="font-size:11px;color:{color};font-weight:600;text-transform:uppercase;
                            letter-spacing:1px;margin-top:2px;">{pos}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:#5a6a7e;margin-bottom:6px;">Información general</div>', unsafe_allow_html=True)
            for label, value in [
                ("Apellido", jugador["nombre"].split()[-1]),
                ("Nombre", jugador["nombre"].split()[0]),
                ("Edad", f"{jugador['edad']} años"),
                ("Pie", jugador["pie"]),
                ("Altura", jugador["altura"]),
                ("País", jugador["pais"]),
                ("Club", "ED Atl. Sanluqueño"),
                ("Posición", pos),
            ]:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e2a3a;">
                    <span style="font-size:12px;color:#00d4aa;font-weight:500;">{label}</span>
                    <span style="font-size:12px;color:#e8eaed;">{value}</span>
                </div>
                """, unsafe_allow_html=True)

            # VIDEO TYPES (like Wyscout)
            st.markdown('<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:#5a6a7e;margin:16px 0 8px;">Vídeos disponibles</div>', unsafe_allow_html=True)
            video_types = [
                ("🎬","Informe automático"), ("✅","Mejores acciones"), ("⚽","Goles"),
                ("🎯","Asistencias"), ("↔️","Pases"), ("📐","Centros"),
                ("➡️","Pases en profundidad"), ("🏃","Conducciones"),
                ("⚔️","Duelos ofensivos"), ("🛡️","Duelos defensivos"),
                ("🥅","Tiros"), ("🔄","Recuperaciones"), ("💨","Aceleraciones"),
                ("🤲","Faltas ganadas"), ("🦵","Faltas cometidas"), ("🤼","Duelos aéreos"),
            ]
            c1, c2 = st.columns(2)
            for i, (icon, label) in enumerate(video_types):
                with (c1 if i % 2 == 0 else c2):
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:5px;padding:5px 7px;
                                background:#111827;border:1px solid #1e2a3a;border-radius:6px;
                                margin-bottom:3px;cursor:pointer;">
                        <span style="font-size:11px;">{icon}</span>
                        <span style="font-size:10px;color:#8899aa;">{label}</span>
                    </div>
                    """, unsafe_allow_html=True)

        # RIGHT: Stats chart + pitch + previous matches
        with col_r:
            st.markdown('<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:#5a6a7e;margin-bottom:10px;">Player Stats</div>', unsafe_allow_html=True)

            cc1, cc2 = st.columns(2)

            # Pie chart distribución de acciones
            with cc1:
                if es_portero:
                    labels_pie = ['Paradas', 'Salidas', 'Distribución', 'Despejes', 'G. enc.']
                    sizes_pie = [45, 15, 20, 12, 8]
                    cpie = ['#00d4aa','#4d9fff','#a78bfa','#FFD700','#ff4d6d']
                else:
                    labels_pie = ['Pases', 'Duelos', 'Recuperac.', 'Asistencias', 'Tiros', 'Conduc.']
                    sizes_pie = [35, 20, 15, 12, 10, 8]
                    cpie = ['#00d4aa','#4d9fff','#a78bfa','#FFD700','#ff4d6d','#ff6b35']

                fig_p, ax_p = plt.subplots(figsize=(4, 4.5))
                fig_p.patch.set_facecolor('#111827')
                ax_p.set_facecolor('#111827')
                wedges, _, autotexts = ax_p.pie(
                    sizes_pie, labels=None, colors=cpie,
                    autopct='%1.0f%%', pctdistance=0.72,
                    wedgeprops=dict(width=0.52, edgecolor='#111827', linewidth=2),
                    startangle=90
                )
                for t in autotexts:
                    t.set_color('white'); t.set_fontsize(8); t.set_fontweight('bold')
                ax_p.legend(wedges, labels_pie, loc='lower center', ncol=2,
                            frameon=False, fontsize=7, labelcolor='#8899aa',
                            bbox_to_anchor=(0.5, -0.22))
                plt.tight_layout()
                buf_pie = io.BytesIO()
                plt.savefig(buf_pie, format='png', dpi=130, bbox_inches='tight', facecolor='#111827')
                buf_pie.seek(0); plt.close()
                st.image(buf_pie, use_container_width=True)

            # Mini campo con posición del jugador
            with cc2:
                fig_f, ax_f = plt.subplots(figsize=(4, 5.5))
                fig_f.patch.set_facecolor('#111827')
                ax_f.set_facecolor('#0a3d1f')
                ax_f.set_xlim(0, 105); ax_f.set_ylim(0, 68)
                ax_f.set_aspect('equal'); ax_f.axis('off')
                lc = (1, 1, 1, 0.6)
                ax_f.add_patch(patches.Rectangle((0,0),105,68,lw=1.2,edgecolor=lc,facecolor='none'))
                ax_f.plot([52.5,52.5],[0,68],color=lc,lw=1.2)
                ax_f.add_patch(Circle((52.5,34),9.15,color=lc,fill=False,lw=1.2))
                ax_f.add_patch(patches.Rectangle((0,13.84),16.5,40.32,lw=1.2,edgecolor=lc,facecolor='none'))
                ax_f.add_patch(patches.Rectangle((88.5,13.84),16.5,40.32,lw=1.2,edgecolor=lc,facecolor='none'))
                for x in range(0,106,10):
                    ax_f.add_patch(patches.Rectangle((x,0),5,68,facecolor='#0d4a25',edgecolor='none',alpha=0.25))
                px, py = POS_XY.get(pos, (52, 34))
                ax_f.add_patch(Circle((px,py),4.5,color=color,zorder=5))
                ax_f.add_patch(Circle((px,py),4.5,color='none',edgecolor='white',lw=1.5,zorder=6))
                ax_f.text(px, py-10, pos[:6], ha='center', va='center', fontsize=7,
                          color='white', fontweight='600', zorder=7)
                buf_fld = io.BytesIO()
                plt.savefig(buf_fld, format='png', dpi=130, bbox_inches='tight', facecolor='#111827')
                buf_fld.seek(0); plt.close()
                st.image(buf_fld, use_container_width=True)

            # Stat highlights (tabla estilo Wyscout)
            for label, val in [
                ("Pie", jugador["pie"]),
                ("Partidos jugados", stats["partidos"]),
                ("Min. jugados", f"{stats['minutos']:,}"),
                ("Goles marcados" if not es_portero else "Goles encajados",
                 stats["goles"] if not es_portero else stats["goles_encajados"]),
                ("Goles / partido", f"{stats['goles']/max(stats['partidos'],1):.1f}" if not es_portero else "—"),
                ("Tarjetas amarillas", int(np.random.randint(0,5))),
                ("Tarjetas rojas", int(np.random.randint(0,2))),
            ]:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e2a3a;">
                    <span style="font-size:12px;color:#00d4aa;font-weight:500;">{label}</span>
                    <span style="font-size:12px;color:#e8eaed;font-weight:600;">{val}</span>
                </div>
                """, unsafe_allow_html=True)

            # PARTIDOS ANTERIORES
            st.markdown('<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:#5a6a7e;margin:18px 0 8px;">Partidos anteriores</div>', unsafe_allow_html=True)
            np.random.seed(jugador["dorsal"] * 17)
            rivales = ["Real Jaén","Lucena CF","Pozoblanco","Écija","Utrera","Osuna"]
            rows = []
            for i in range(6):
                rival = rivales[i % len(rivales)]
                local = bool(np.random.choice([True, False]))
                enc = f"ED Sanluqueño – {rival}" if local else f"{rival} – ED Sanluqueño"
                res = f"{int(np.random.randint(0,3))} – {int(np.random.randint(0,3))}"
                rows.append({
                    "Fecha": f"{int(np.random.randint(1,28)):02d}/{int(np.random.randint(1,3)):02d}/2026",
                    "Partido": enc,
                    "Resultado": res,
                    "Pos.": pos[:5],
                    "Min.": f"{int(np.random.randint(45,90))}'",
                    "G": int(np.random.randint(0,2)) if not es_portero else 0,
                    "A": int(np.random.randint(0,2)) if not es_portero else 0,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ══ TAB ESTADÍSTICAS ══
    with tab_stats:
        np.random.seed(jugador["dorsal"] * 7)
        st.markdown('<div class="ws-section-header">Estadísticas de temporada</div>', unsafe_allow_html=True)
        k1,k2,k3,k4,k5,k6 = st.columns(6)
        k1.metric("Partidos", stats["partidos"])
        k2.metric("Minutos", f"{stats['minutos']:,}")
        k3.metric("Goles" if not es_portero else "Paradas", stats["goles"] if not es_portero else stats["paradas"])
        k4.metric("Asistencias" if not es_portero else "G. enc.", stats["asistencias"] if not es_portero else stats["goles_encajados"])
        k5.metric("Pases/90", stats["pases_por_partido"])
        k6.metric("Nota", stats["nota"])

        st.markdown('<div class="ws-section-header">Perfil de rendimiento</div>', unsafe_allow_html=True)
        metrics = {
            "Pases": stats["precision_pase"],
            "Duelos": int(stats["duelos_ganados"]/max(stats["duelos"],1)*100),
            "Recuperaciones": min(100, stats["recuperaciones"]*3),
            "Velocidad": min(100, int((stats["vel_max"]-20)/14*100)),
            "Goles+Asist.": min(100,(stats["goles"]+stats["asistencias"])*5) if not es_portero else min(100,int(stats["paradas"]/max(stats["paradas"]+stats["goles_encajados"],1)*100)),
            "Distancia": min(100, int(stats["distancia_km"]/15*100)),
        }
        for metric, value in metrics.items():
            cl, cb = st.columns([1, 3])
            with cl:
                st.markdown(f'<div style="font-size:12px;color:#8899aa;margin-top:8px;">{metric}</div>', unsafe_allow_html=True)
            with cb:
                st.markdown(f"""
                <div style="background:#1e2a3a;border-radius:6px;height:10px;margin-top:10px;overflow:hidden;">
                    <div style="width:{value}%;background:{color};height:100%;border-radius:6px;"></div>
                </div>
                <div style="font-size:11px;color:#5a6a7e;text-align:right;margin-top:2px;">{value}/100</div>
                """, unsafe_allow_html=True)

    # ══ TAB VÍDEOS ══
    with tab_events:
        st.markdown('<div class="ws-section-header">Clips del partido</div>', unsafe_allow_html=True)
        ball_events = st.session_state.get("ball_events", [])
        player_events = [e for e in ball_events if jugador["nombre"].split()[-1].lower() in e.get("nombre_jugador","").lower()]

        if player_events:
            for ev in player_events[:10]:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;padding:10px 12px;
                            background:#111827;border:1px solid #1e2a3a;border-radius:8px;margin-bottom:4px;">
                    <div style="font-size:14px;font-weight:700;color:{color};min-width:40px;">
                        Min {int(ev.get('minute',0))}'
                    </div>
                    <div style="font-size:12px;color:#e8eaed;">Acción con balón · {ev.get('nombre_equipo','—')}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            vt_all = [
                ("⚽","Goles"),("🎯","Asistencias"),("↔️","Pases"),("📐","Centros"),
                ("🏃","Conducciones"),("🤼","Duelos aéreos"),("⚔️","Duelos ofensivos"),
                ("🛡️","Duelos defensivos"),("🥅","Tiros"),("💨","Aceleraciones"),
                ("🔄","Recuperaciones"),("🤲","Faltas ganadas"),
            ]
            st.markdown('<div style="font-size:12px;color:#5a6a7e;margin-bottom:12px;">Carga un vídeo para ver los clips reales. Tipos disponibles:</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            for i, (icon, label) in enumerate(vt_all):
                with [c1, c2, c3][i % 3]:
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:6px;padding:8px 10px;
                                background:#111827;border:1px solid #1e2a3a;border-radius:8px;margin-bottom:6px;">
                        <span>{icon}</span>
                        <span style="font-size:12px;color:#8899aa;">{label}</span>
                    </div>
                    """, unsafe_allow_html=True)


def render():
    # Modo detalle de jugador
    selected = st.session_state.get("squad_selected_player")
    if selected is not None:
        jugador = next((j for j in PLANTILLA if j["dorsal"] == selected), None)
        if jugador:
            _render_player_detail(jugador)
            return

    # ── Grid de plantilla ────────────────────────────────────────────────────
    config = st.session_state.get("analysis_config", {})
    nombre_equipo = config.get("team", "Mi Equipo")

    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:24px;">
        <div>
            <h2 style="margin:0;font-size:20px;font-weight:700;color:#fff;">Plantilla — {nombre_equipo}</h2>
            <p style="margin:2px 0 0;font-size:13px;color:#5a6a7e;">20 jugadores · Temporada 2024-25 · Haz clic para ver la ficha</p>
        </div>
        <div style="background:rgba(0,212,170,0.1);border:1px solid rgba(0,212,170,0.25);color:#00d4aa;
                    padding:4px 14px;border-radius:20px;font-size:11px;font-weight:600;text-transform:uppercase;">
            {len(PLANTILLA)} jugadores
        </div>
    </div>
    """, unsafe_allow_html=True)

    grupos_posicion = {
        "Todos": None,
        "🟧 Porteros": ["Portero"],
        "🟦 Defensas": ["Lateral D","Lateral I","Central"],
        "🟩 Centrocampistas": ["Pivote","Mediocentro","Interior D","Interior I","Mediapunta"],
        "🟥 Atacantes": ["Extremo D","Extremo I","Delantero"],
    }
    filtro = st.radio("Filtrar", list(grupos_posicion.keys()), horizontal=True, label_visibility="collapsed")
    posiciones_filtro = grupos_posicion[filtro]
    jugadores_filtrados = [j for j in PLANTILLA if posiciones_filtro is None or j["posicion"] in posiciones_filtro]

    st.markdown("<br>", unsafe_allow_html=True)

    cols_per_row = 4
    for row_start in range(0, len(jugadores_filtrados), cols_per_row):
        row = jugadores_filtrados[row_start: row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, jugador in zip(cols, row):
            with col:
                pos = jugador["posicion"]
                color = POSICION_COLOR.get(pos, "#00d4aa")
                stats = _generar_stats(jugador)
                st.markdown(f"""
                <div style="background:#111827;border:1px solid #1e2a3a;border-radius:14px;
                            overflow:hidden;margin-bottom:4px;">
                    <div style="height:4px;background:{color};"></div>
                    <div style="padding:20px;text-align:center;">
                        <div style="width:64px;height:64px;border-radius:50%;background:{color}22;
                                    border:2px solid {color}55;margin:0 auto 10px;display:flex;
                                    align-items:center;justify-content:center;">
                            <span style="font-size:22px;font-weight:900;color:{color};">{jugador['dorsal']}</span>
                        </div>
                        <div style="font-size:14px;font-weight:700;color:#fff;margin-bottom:4px;
                                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                            {jugador['nombre']}
                        </div>
                        <div style="font-size:10px;font-weight:600;text-transform:uppercase;
                                    letter-spacing:1px;color:{color};margin-bottom:10px;">{pos}</div>
                        <div style="display:flex;justify-content:space-around;border-top:1px solid #1e2a3a;padding-top:10px;">
                            <div style="text-align:center;">
                                <div style="font-size:16px;font-weight:700;color:#fff;">{stats['partidos']}</div>
                                <div style="font-size:9px;color:#5a6a7e;text-transform:uppercase;">PJ</div>
                            </div>
                            <div style="text-align:center;">
                                <div style="font-size:16px;font-weight:700;color:#fff;">{stats['goles'] if pos != 'Portero' else stats['paradas']}</div>
                                <div style="font-size:9px;color:#5a6a7e;text-transform:uppercase;">{'G' if pos != 'Portero' else 'PAR'}</div>
                            </div>
                            <div style="text-align:center;">
                                <div style="font-size:16px;font-weight:700;color:{color};">{stats['nota']}</div>
                                <div style="font-size:9px;color:#5a6a7e;text-transform:uppercase;">NOTA</div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Ver ficha #{jugador['dorsal']}", key=f"player_{jugador['dorsal']}",
                             use_container_width=True):
                    st.session_state["squad_selected_player"] = jugador["dorsal"]
                    st.rerun()
        for empty_col in cols[len(row):]:
            with empty_col:
                st.empty()
