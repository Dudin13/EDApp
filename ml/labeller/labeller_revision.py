"""
labeller_revision.py
====================
Version standalone del labeller para revision externa de etiquetas.
No necesita EDApp instalado — funciona desde cualquier carpeta.

Estructura esperada:
    revision_veo/
    ├── imagenes/           <- imagenes VEO a revisar
    ├── etiquetas_auto/     <- etiquetas auto-generadas (referencia, no se tocan)
    ├── etiquetas_ok/       <- aqui se guardan las etiquetas corregidas (se crea sola)
    ├── revisadas/          <- imagenes ya revisadas (se crea sola)
    ├── labeller_revision.py
    ├── labeller_revision.html
    └── instalar_y_lanzar.bat

Uso:
    python labeller_revision.py
    Luego abre: http://localhost:5050
"""

import os
import shutil
from pathlib import Path
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:*", "http://127.0.0.1:*"])

# ── Rutas relativas a donde esta este script ───────────────────────────────
HERE         = Path(__file__).parent.absolute()
IMG_DIR      = HERE / "imagenes"
AUTO_DIR     = HERE / "etiquetas_auto"
OK_DIR       = HERE / "etiquetas_ok"
REVISADAS    = HERE / "revisadas"

# Crear carpetas si no existen
OK_DIR.mkdir(exist_ok=True)
REVISADAS.mkdir(exist_ok=True)

# Intentar cargar YOLO para auto-deteccion (opcional)
model = None
try:
    from ultralytics import YOLO
    model_candidates = [
        HERE / "best.pt",
        HERE / "yolo11m-seg.pt",
    ]
    for mp in model_candidates:
        if mp.exists():
            model = YOLO(str(mp))
            print(f"[OK] Modelo cargado: {mp.name}")
            break
    if model is None:
        print("[WARN] No se encontro modelo .pt — YOLO predict desactivado")
except ImportError:
    print("[WARN] ultralytics no instalado — YOLO predict desactivado")


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(str(HERE), 'labeller_revision.html')


@app.route('/api/images')
def list_images():
    """Lista imagenes pendientes de revisar (no están en revisadas/)."""
    if not IMG_DIR.exists():
        return jsonify([])

    all_imgs    = set(f.name for f in IMG_DIR.glob("*.jpg")) | set(f.name for f in IMG_DIR.glob("*.png"))
    revisadas   = set(f.name for f in REVISADAS.glob("*.jpg")) | set(f.name for f in REVISADAS.glob("*.png"))
    pendientes  = sorted(all_imgs - revisadas)

    return jsonify({
        "images":    pendientes,
        "total":     len(all_imgs),
        "revisadas": len(revisadas),
        "pendientes": len(pendientes),
    })


@app.route('/api/image/<filename>')
def get_image(filename):
    return send_from_directory(str(IMG_DIR), filename)


@app.route('/api/labels/<filename>')
def get_labels(filename):
    """Devuelve etiquetas auto-generadas como referencia."""
    stem       = Path(filename).stem
    label_path = AUTO_DIR / f"{stem}.txt"
    if not label_path.exists():
        return jsonify([])

    labels = []
    for line in label_path.read_text(encoding="utf-8").strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 5:
            labels.append({
                "cls": int(float(parts[0])),
                "cx":  float(parts[1]),
                "cy":  float(parts[2]),
                "w":   float(parts[3]),
                "h":   float(parts[4]),
            })
    return jsonify(labels)


@app.route('/api/save', methods=['POST'])
def save_labels():
    """
    Guarda etiquetas corregidas en etiquetas_ok/ y mueve
    la imagen a revisadas/ para marcarla como procesada.
    """
    data     = request.json
    filename = data.get("filename", "")
    labels   = data.get("labels", [])

    if not filename:
        return jsonify({"error": "filename requerido"}), 400

    stem = Path(filename).stem

    # Guardar etiquetas corregidas con sufijo _CORREGIDA
    lines = []
    for l in labels:
        lines.append(f"{l['cls']} {l['cx']:.6f} {l['cy']:.6f} {l['w']:.6f} {l['h']:.6f}")

    label_out = OK_DIR / f"{stem}_CORREGIDA.txt"
    label_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Mover imagen a revisadas/
    img_src = IMG_DIR / filename
    img_dst = REVISADAS / filename
    if img_src.exists() and not img_dst.exists():
        shutil.move(str(img_src), str(img_dst))

    # Contar progreso
    total    = len(list(IMG_DIR.glob("*.jpg"))) + len(list(REVISADAS.glob("*.jpg")))
    revisadas = len(list(REVISADAS.glob("*.jpg")))

    print(f"[OK] Guardado: {filename} | Progreso: {revisadas}/{total}")
    return jsonify({"ok": True, "revisadas": revisadas, "total": total})


@app.route('/api/skip', methods=['POST'])
def skip_image():
    """Marca una imagen como revisada sin guardar etiqueta (imagen ok, sin cambios)."""
    data     = request.json
    filename = data.get("filename", "")
    if not filename:
        return jsonify({"error": "filename requerido"}), 400

    # Copiar etiqueta auto como valida
    stem       = Path(filename).stem
    auto_label = AUTO_DIR / f"{stem}.txt"
    ok_label   = OK_DIR   / f"{stem}.txt"
    if auto_label.exists() and not ok_label.exists():
        shutil.copy2(str(auto_label), str(ok_label))

    # Mover imagen a revisadas
    img_src = IMG_DIR / filename
    img_dst = REVISADAS / filename
    if img_src.exists() and not img_dst.exists():
        shutil.move(str(img_src), str(img_dst))

    return jsonify({"ok": True})


@app.route('/api/predict', methods=['POST'])
def predict():
    """Auto-deteccion YOLO sobre la imagen actual (si hay modelo disponible)."""
    if model is None:
        return jsonify({"error": "Modelo no disponible"}), 503

    data     = request.json
    filename = data.get("filename", "")
    img_path = IMG_DIR / filename
    if not img_path.exists():
        # Buscar en revisadas tambien
        img_path = REVISADAS / filename
    if not img_path.exists():
        return jsonify({"error": "Imagen no encontrada"}), 404

    import cv2
    img     = cv2.imread(str(img_path))
    results = model(img, imgsz=1280, conf=0.35, verbose=False)

    CLASS_NAMES = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}
    labels = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            cx, cy, w, h = [float(x) for x in box.xywhn[0]]
            labels.append({"cls": cls, "cx": cx, "cy": cy, "w": w, "h": h,
                           "name": CLASS_NAMES.get(cls, "unknown"),
                           "conf": float(box.conf[0])})

    return jsonify(labels)


if __name__ == "__main__":
    print()
    print("=" * 50)
    print("  EDApp — Labeller de Revision")
    print("=" * 50)
    print(f"  Imagenes:      {IMG_DIR}")
    print(f"  Etiquetas OK:  {OK_DIR}")
    print(f"  Revisadas:     {REVISADAS}")
    n_pending = len(list(IMG_DIR.glob("*.jpg"))) if IMG_DIR.exists() else 0
    print(f"  Pendientes:    {n_pending} imagenes")
    print()
    print("  Abre en el navegador: http://localhost:5050")
    print("=" * 50)
    print()
    app.run(host="127.0.0.1", port=5050, debug=False)
