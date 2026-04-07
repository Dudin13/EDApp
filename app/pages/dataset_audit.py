"""
dataset_audit.py - Lanzador del EDudin Labeller desde Streamlit

Arranca el servidor FastAPI del labeller en background y
ofrece un boton para abrirlo en el navegador.
No usa iframe: el labeller corre en su propia pestana del navegador,
lo que permite el uso correcto del teclado (Delete, Espacio, etc.)
"""

import streamlit as st
import threading
import time
import socket
import sys
from pathlib import Path

# Asegurar paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LABELLER_URL  = "http://localhost:8570"
LABELLER_PORT = 8570


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _start_labeller_server():
    """Arranca el servidor FastAPI del labeller en un hilo daemon."""
    import uvicorn
    # Importar la app del servidor standalone
    server_path = PROJECT_ROOT / "tools" / "labeller" / "server.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("labeller_server", str(server_path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    uvicorn.run(mod.app, host="127.0.0.1", port=LABELLER_PORT, log_level="warning")


def main():
    st.set_page_config(
        layout="centered",
        page_title="EDudin Labeller",
        page_icon="🔬",
    )

    st.markdown("""
    <style>
    .stMain { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🔬 EDudin Pro Labeller")
    st.markdown("Herramienta de etiquetado manual e IA para el dataset de jugadores.")
    st.divider()

    # ── Estado del servidor ──────────────────────────────────────────────────
    server_running = _port_in_use(LABELLER_PORT)

    col_status, col_btn = st.columns([2, 1])

    with col_status:
        if server_running:
            st.success(f"Servidor activo en {LABELLER_URL}")
        else:
            st.warning("Servidor no iniciado")

    with col_btn:
        if not server_running:
            if st.button("Iniciar servidor", type="primary", use_container_width=True):
                if "labeller_thread" not in st.session_state:
                    t = threading.Thread(target=_start_labeller_server, daemon=True)
                    t.start()
                    st.session_state.labeller_thread = t
                time.sleep(2)
                st.rerun()
        else:
            if st.button("Reiniciar pagina", use_container_width=True):
                st.rerun()

    st.divider()

    if server_running:
        # Boton grande para abrir en nueva pestana
        st.markdown(f"""
        <div style="text-align:center; padding: 30px 0;">
            <a href="{LABELLER_URL}" target="_blank" style="
                display: inline-block;
                background: #00d4aa;
                color: #0f172a;
                font-weight: 800;
                font-size: 18px;
                padding: 16px 48px;
                border-radius: 10px;
                text-decoration: none;
                letter-spacing: -0.3px;
            ">Abrir Labeller en nueva pestana</a>
            <p style="margin-top:12px; color:#8899aa; font-size:13px;">
                Se abre en tu navegador completo — teclado y zoom funcionan correctamente.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Info del flujo
        with st.expander("Flujo de carpetas", expanded=False):
            st.markdown("""
            | Carpeta | Estado |
            |---------|--------|
            | `0_por_entrenar` | Imagenes nuevas sin revisar |
            | `1_entrenadas_ia` | Etiquetadas por la IA (pendiente revision) |
            | `2_entrenadas_manual` | Verificadas manualmente — listas para entrenar |
            | `3_final` | Dataset final de entrenamiento |

            **Atajos en el labeller:**
            - `Click` en imagen → SAM autodetecta el jugador mas cercano
            - `Arrastrar` → Dibuja caja manual
            - `Click derecho / Espacio+arrastrar` → Pan (mover vista)
            - `Ctrl+Scroll` → Zoom
            - `Delete / Backspace` → Borrar caja seleccionada
            """)

        with st.expander("Actualizar modelo de IA", expanded=False):
            st.markdown("""
            El labeller usa el modelo en `assets/weights/` con esta prioridad:

            1. `detect_players_v4_clean.pt` (recomendado — el nuevo)
            2. `detect_players_v3.pt`
            3. `detect_players.pt`

            Para entrenar el modelo v4:
            ```bash
            python ml/training/prepare_dataset.py
            python ml/training/train_players_v4.py
            ```
            El modelo se guarda automaticamente en `assets/weights/` al terminar.
            """)
            weights_dir = PROJECT_ROOT / "assets" / "weights"
            models = list(weights_dir.glob("*.pt"))
            if models:
                st.markdown("**Modelos disponibles:**")
                for m in sorted(models):
                    size_mb = m.stat().st_size / 1024**2
                    st.code(f"{m.name}  ({size_mb:.0f} MB)")
    else:
        st.info("Pulsa 'Iniciar servidor' para arrancar el labeller.")
        st.markdown("""
        **Alternativa — ejecutar directamente desde terminal:**
        ```bash
        python tools/labeller/server.py
        ```
        O hacer doble clic en `tools/labeller/run_labeller.bat`
        """)


if __name__ == "__main__":
    main()
