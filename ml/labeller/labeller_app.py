import os
import uuid
import tempfile
import cv2
from flask import Flask, send_from_directory, request, jsonify, after_this_request
from flask_cors import CORS
from pathlib import Path
from ultralytics import YOLO
from ultralytics import SAM
import torch

app = Flask(__name__)

# ✅ CORS habilitado
CORS(app)

# ✅ Rutas raíz
BASE = Path("C:/apped")
DATASET_ROOT = (BASE / "data/datasets").absolute()
MODEL_DIR = BASE / "models"
MODEL_PLAYERS = MODEL_DIR / "players.pt"
MODEL_SAM = MODEL_DIR / "sam2_t.pt"

# ✅ Subsets válidos (Simplificado por petición del usuario)
VALID_SUBSETS = {"veo_frames_raw"}

# ✅ Clases de Fútbol
CLASSES = ["player", "goalkeeper", "referee", "ball", "equipo_a", "equipo_b"]

# Modelos con carga lazy
_models = {}

def get_model(name="players"):
    global _models
    if name not in _models:
        path = MODEL_PLAYERS
        print(f"🔄 Cargando modelo YOLO: {name} ({path})...")
        _models[name] = YOLO(str(path))
    return _models[name]

_sam_model = None

def get_sam_model():
    global _sam_model
    if _sam_model is None:
        print("🔄 Cargando modelo SAM...")
        _sam_model = SAM("sam2_t.pt")
        print("✅ SAM cargado.")
    return _sam_model


@app.route('/')
def index():
    return send_from_directory('.', 'labeller_v2.html')


def safe_resolve(base: Path, *parts) -> Path | None:
    try:
        resolved = (base.joinpath(*parts)).resolve()
        if not str(resolved).startswith(str(base.resolve())):
            return None
        return resolved
    except Exception:
        return None


@app.route('/api/segment', methods=['POST'])
def segment_image():
    try:
        data = request.json
        subset = data.get('subset')
        filename = data.get('filename')
        point = data.get('point')

        if subset not in VALID_SUBSETS:
            return jsonify({"error": "Invalid subset"}), 400

        base_path = DATASET_ROOT / subset / "images"
        img_path = safe_resolve(base_path, filename)

        if img_path is None or not img_path.exists():
            return jsonify({"error": "Image not found"}), 404

        img = cv2.imread(str(img_path))
        if img is None:
            return jsonify({"error": "Could not read image"}), 500

        h, w = img.shape[:2]
        input_point = [[point['x'] * w, point['y'] * h]]

        is_ball = data.get('is_ball', False)

        sam = get_sam_model()
        
        # Si es balón, limitamos el área de búsqueda de SAM con un cuadro pequeño (60x60 px)
        # para evitar que 'salte' a jugadores cercanos
        predict_kwargs = {
            "points": input_point,
            "labels": [1],
            "verbose": False
        }
        
        if is_ball:
            margin = 30 # px
            px, py = input_point[0]
            predict_kwargs["bboxes"] = [[max(0, px-margin), max(0, py-margin), min(w, px+margin), min(h, py+margin)]]
            print(f"⚽ Ball SAM mode: box constraint applied at {predict_kwargs['bboxes']}")

        results = sam.predict(img, **predict_kwargs)

        if not results or results[0].masks is None:
            return jsonify({"error": "No mask found"}), 400

        if len(results[0].masks.xyn) > 0:
            poly = results[0].masks.xyn[0].tolist()
            return jsonify({"polygon": poly})

        return jsonify({"error": "No mask found"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/images/<subset>')
def list_images(subset):
    if subset not in VALID_SUBSETS:
        return jsonify({"error": "Invalid subset"}), 400

    img_dir = DATASET_ROOT / subset / "images"
    lab_dir = DATASET_ROOT / subset / "labels"

    if not img_dir.exists():
        return jsonify({"error": "Images directory not found"}), 404

    result = []
    # Escanear JPG y PNG
    extensions = ("*.jpg", "*.png")
    files = []
    for ext in extensions:
        files.extend(img_dir.glob(ext))
    
    manual_count = 0
    for f in sorted(files):
        label_file = lab_dir / (f.stem + ".txt")
        corregida = label_file.exists()
        
        # Nueva lógica: ¿Es revisión manual humana?
        revisado_manual = False
        if corregida:
            try:
                # Leemos solo la primera línea para chequear el flag
                content = label_file.read_text(encoding="utf-8").strip()
                if content.startswith("# revisado_manual"):
                    revisado_manual = True
                    manual_count += 1
            except Exception:
                pass
                
        result.append({
            "name": f.name, 
            "corregida": corregida,
            "revisado_manual": revisado_manual
        })

    return jsonify(result)


@app.route('/api/image/<subset>/<path:filename>')
def get_image(subset, filename):
    if subset not in VALID_SUBSETS:
        return jsonify({"error": "Invalid subset"}), 400

    safe_path = (DATASET_ROOT / subset / "images").absolute()
    return send_from_directory(str(safe_path), filename)


@app.route('/api/predict_data/<subset>/<path:filename>')
def predict_data(subset, filename):
    if subset not in VALID_SUBSETS:
        return jsonify({"error": "Invalid subset"}), 400

    base_path = DATASET_ROOT / subset / "images"
    img_path = safe_resolve(base_path, filename)

    if img_path is None or not img_path.exists():
        return jsonify({"error": "Image not found"}), 404

    # ── Step 1: YOLO detection ──────────────────────────────────────────
    model = get_model()
    results = model.predict(img_path, conf=0.25, imgsz=1280, verbose=False)[0]
    detections = []

    if results.masks is not None:
        for mask, cls in zip(results.masks.xyn, results.boxes.cls):
            detections.append({"cls": int(cls), "points": mask.tolist()})
    else:
        for box, cls in zip(results.boxes.xyxyn, results.boxes.cls):
            x1, y1, x2, y2 = box.tolist()
            detections.append({
                "cls": int(cls),
                "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            })

    # ── Step 2: Optional SAM2 refinement ───────────────────────────────
    sam2_refine = request.args.get("sam2_refine", "false").lower() == "true"

    if sam2_refine and len(detections) > 0:
        try:
            img_cv = cv2.imread(str(img_path))
            h, w = img_cv.shape[:2]

            # Build pixel bboxes for SAM2 from YOLO normalised coords
            sam_bboxes = []
            for d in detections:
                xs = [p[0] for p in d["points"]]
                ys = [p[1] for p in d["points"]]
                sam_bboxes.append([
                    min(xs) * w, min(ys) * h,
                    max(xs) * w, max(ys) * h
                ])

            print(f"⚽ SAM2 refining {len(sam_bboxes)} bboxes...")
            sam = get_sam_model()
            sam_results = sam.predict(img_cv, bboxes=sam_bboxes, verbose=False)

            refined = []
            for i, (res, orig_d) in enumerate(zip(sam_results, detections)):
                if res.masks is not None and len(res.masks.xyn) > 0:
                    poly = res.masks.xyn[0].tolist()
                    refined.append({"cls": orig_d["cls"], "points": poly})
                else:
                    # SAM didn't find a mask → keep YOLO bbox
                    refined.append(orig_d)

            detections = refined
            print(f"✅ SAM2 done: {len(detections)} refined detections")

        except Exception as e:
            print(f"⚠️ SAM2 refine failed, falling back to YOLO: {e}")
            import traceback; traceback.print_exc()
            # detections already has YOLO results, just proceed

    return jsonify(detections)


@app.route('/api/save_labels', methods=['POST'])
def save_labels():
    data = request.json
    subset = data.get('subset')
    filename = data.get('filename')
    labels = data.get('labels')

    if subset not in VALID_SUBSETS:
        return jsonify({"error": "Invalid subset"}), 400

    base_path = DATASET_ROOT / subset / "labels"
    label_filename = Path(filename).stem + ".txt"
    lab_path = safe_resolve(base_path, label_filename)
    
    if lab_path is None:
        return jsonify({"error": "Invalid file path"}), 403

    lab_path.parent.mkdir(parents=True, exist_ok=True)

    # Añadimos el flag de revisión manual humana
    final_content = ["# revisado_manual"] + labels

    with open(lab_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(final_content))

    return jsonify({"status": "success"})


@app.route('/api/load_labels/<subset>/<path:filename>')
def load_labels(subset, filename):
    if subset not in VALID_SUBSETS:
        return jsonify({"error": "Invalid subset"}), 400

    base_path = DATASET_ROOT / subset / "labels"
    label_filename = Path(filename).stem + ".txt"
    lab_path = safe_resolve(base_path, label_filename)
    
    if lab_path is None or not lab_path.exists():
        return jsonify({"labels": []})

    try:
        # Cargamos el archivo y filtramos los comentarios (líneas que empiezan con #)
        all_lines = lab_path.read_text(encoding="utf-8").splitlines()
        yolo_lines = [l.strip() for l in all_lines if l.strip() and not l.strip().startswith("#")]
        return jsonify({"labels": yolo_lines})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)
