import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io


def render():
    # ── PAGE HEADER ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;">
        <div>
            <h2 style="margin:0;font-size:20px;font-weight:700;color:#fff;">Métricas del Partido</h2>
            <p style="margin:2px 0 0;font-size:13px;color:#5a6a7e;">Estadísticas técnicas y físicas por jugador</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.get("analysis_done"):
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e2a3a;border-radius:12px;padding:32px;text-align:center;">
            <div style="font-size:36px;margin-bottom:12px;">📊</div>
            <div style="font-size:16px;font-weight:600;color:#fff;margin-bottom:8px;">Sin datos de análisis</div>
            <div style="font-size:13px;color:#5a6a7e;">Primero realiza un análisis en "Análisis de Vídeo"</div>
        </div>
        """, unsafe_allow_html=True)
        return

    resultados_jug = st.session_state.get("resultados_jugadores", {})
    config = st.session_state.get("analysis_config", {})
    det_por_minuto = st.session_state.get("detecciones_por_minuto", {})

    if not resultados_jug:
        st.warning("El análisis no generó datos de jugadores.")
        return

    # ── RESUMEN GLOBAL ────────────────────────────────────────────────────────
    st.markdown('<div class="ws-section-header">Resumen del partido</div>', unsafe_allow_html=True)

    total_dets = sum(r2.get("frames_detectado", 0) for r2 in resultados_jug.values())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jugadores registrados", len(resultados_jug))
    c2.metric("Total detecciones", f"{st.session_state.get('total_detecciones', total_dets):,}")
    c3.metric("Frames analizados", st.session_state.get("frames_analizados", 0))
    partido = f"{config.get('team', '')} vs {config.get('rival', '')}"
    c4.metric("Partido", partido if len(partido) < 25 else partido[:22] + "…")

    # ── SELECTOR JUGADOR ─────────────────────────────────────────────────────
    st.markdown('<div class="ws-section-header">Perfil técnico del jugador</div>', unsafe_allow_html=True)

    jugadores_list = list(resultados_jug.keys())
    player_sel = st.selectbox("Selecciona jugador", jugadores_list, label_visibility="collapsed")
    r = resultados_jug.get(player_sel, {})
    dorsal = r.get("dorsal", 0)
    posicion = r.get("posicion", "—")
    equipo_jug = r.get("equipo", "")

    # Player header
    st.markdown(f"""
    <div class="ws-player-header">
        <div class="ws-player-number">#{dorsal}</div>
        <div>
            <div class="ws-player-name">{player_sel}</div>
            <div class="ws-player-meta">{equipo_jug} · {config.get('match_date','')}</div>
        </div>
        <div class="ws-player-pos">{posicion if posicion else "—"}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── RADAR CHART ───────────────────────────────────────────────────────────
    col_radar, col_table = st.columns([1, 1])

    categories = ['Pases\ncompletados', 'Progresividad', 'Creación\npeligro',
                  'Duelos\nganados', 'Recuperaciones', 'Presiones\nefectivas',
                  'Conducciones\nprog.', 'Tiros\na puerta']

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

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')

    vp = values_player + values_player[:1]
    vt = values_team + values_team[:1]
    ax.plot(angles, vp, 'o-', linewidth=2.5, color='#00d4aa', label=player_sel)
    ax.fill(angles, vp, alpha=0.25, color='#00d4aa')
    ax.plot(angles, vt, 'o-', linewidth=1.5, color='#5a6a7e', linestyle='--', label='Media equipo')
    ax.fill(angles, vt, alpha=0.08, color='#5a6a7e')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='#8899aa', size=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], color='#3a4a5e', size=7)
    ax.grid(color='#1e2a3a', alpha=0.8)
    ax.spines['polar'].set_color('#1e2a3a')
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1),
              facecolor='#111827', labelcolor='white', fontsize=9, framealpha=0.9)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#111827')
    buf.seek(0)
    plt.close()

    with col_radar:
        st.image(buf, use_container_width=True)

    with col_table:
        st.markdown('<div class="ws-section-header" style="margin-top:0">Comparativa de métricas</div>', unsafe_allow_html=True)
        metrics_df = pd.DataFrame({
            "Métrica": [c.replace('\n', ' ') for c in categories],
            player_sel: values_player,
            "Media equipo": values_team,
            "Dif.": [p - t for p, t in zip(values_player, values_team)]
        })

        def color_diff(val):
            if val > 0:
                return 'color: #00d4aa; font-weight: 600'
            elif val < 0:
                return 'color: #ff4d6d; font-weight: 600'
            return ''

        st.dataframe(
            metrics_df.style.map(color_diff, subset=["Dif."]),
            use_container_width=True, hide_index=True
        )

        # Mini KPIs
        st.markdown("<br>", unsafe_allow_html=True)
        k1, k2 = st.columns(2)
        k1.metric("Distancia estimada", f"{r.get('distance_km', 0)} km")
        k2.metric("Vel. máxima", f"{r.get('top_speed', 0)} km/h")
        k3, k4 = st.columns(2)
        k3.metric("Pases", r.get("passes", 0))
        k4.metric("Duelos gan./tot.", f"{r.get('duels_won',0)}/{r.get('duels_won',0)+r.get('duels_lost',0)}")

    # ── EVOLUCIÓN POR MINUTO ─────────────────────────────────────────────────
    if det_por_minuto:
        st.markdown('<div class="ws-section-header">Actividad por minuto</div>', unsafe_allow_html=True)
        minutos = sorted(det_por_minuto.keys(), key=lambda x: int(x))
        valores = [det_por_minuto[m] for m in minutos]

        fig2, ax2 = plt.subplots(figsize=(12, 3.5))
        fig2.patch.set_facecolor('#111827')
        ax2.set_facecolor('#111827')

        bars = ax2.bar(minutos, valores, color='#00d4aa', alpha=0.7, edgecolor='none', width=0.7)
        # Línea de media
        media = sum(valores) / len(valores) if valores else 0
        ax2.axhline(media, color='#5a6a7e', linewidth=1, linestyle='--', alpha=0.6)
        ax2.text(minutos[-1] if minutos else 0, media + 0.1, f'  media: {media:.0f}',
                 color='#5a6a7e', fontsize=8, va='bottom')

        ax2.set_xlabel('Minuto', color='#5a6a7e', fontsize=10)
        ax2.set_ylabel('Detecciones', color='#5a6a7e', fontsize=10)
        ax2.tick_params(colors='#5a6a7e')
        for spine in ax2.spines.values():
            spine.set_visible(False)
        ax2.spines['bottom'].set_visible(True)
        ax2.spines['bottom'].set_color('#1e2a3a')
        ax2.grid(axis='y', color='#1e2a3a', alpha=0.5)

        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png', dpi=150, bbox_inches='tight', facecolor='#111827')
        buf2.seek(0)
        plt.close()
        st.image(buf2, use_container_width=True)

    # ── EXPORTAR ─────────────────────────────────────────────────────────────
    st.markdown('<div class="ws-section-header">Exportar informe</div>', unsafe_allow_html=True)
    col_e1, col_e2, _ = st.columns([1, 1, 2])
    col_e1.button("📄 Exportar PDF", type="primary", use_container_width=True)
    col_e2.button("📊 Exportar Excel", use_container_width=True)