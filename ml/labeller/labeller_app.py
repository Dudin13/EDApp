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

# ✅ CORS restringido a localhost
CORS(app, origins=["http://localhost:*", "http://127.0.0.1:*"])

# ✅ Rutas actualizadas para la estructura de raíz
BASE = Path(os.environ.get("APPED_ROOT", "C:/apped"))
DATASET_ROOT = (BASE / "data/datasets").absolute()
MODEL_PATH = BASE / "yolo11m-seg.pt"

# ✅ Subsets válidos
VALID_SUBSETS = {"train", "val", "test", "super_focused_50", "nuevas_muestras_marzo", "dataset", "combined"}

# ✅ Clases de Fútbol
CLASSES = ["Equipo Local", "Equipo Visitante", "Portero", "Árbitro", "Balón"]

print(f"Dataset Root: {DATASET_ROOT}")
model = YOLO(MODEL_PATH)

# ✅ SAM con carga lazy
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

        print(f"🔮 SAM Request: Subset={subset}, File={filename}, Point={point}")

        if subset not in VALID_SUBSETS:
            return jsonify({"error": "Invalid subset"}), 400

        if not point or 'x' not in point or 'y' not in point:
            return jsonify({"error": "Invalid point data"}), 400

        if subset == "combined":
            img_path = DATASET_ROOT / filename
        else:
            if subset == "super_focused_50":
                base_path = DATASET_ROOT / "super_focused_50/train/images"
            elif subset == "nuevas_muestras_marzo":
                base_path = DATASET_ROOT / "nuevas_muestras_marzo"
            elif subset == "dataset":
                base_path = DATASET_ROOT / "dataset/images/train"
            else:
                base_path = DATASET_ROOT / subset / "images"
            img_path = safe_resolve(base_path, filename)

        if img_path is None or not img_path.exists():
            print(f"❌ Image not found: {img_path}")
            return jsonify({"error": f"Image not found at {img_path}"}), 404

        img = cv2.imread(str(img_path))
        if img is None:
            return jsonify({"error": "Could not read image"}), 500

        h, w = img.shape[:2]
        input_point = [[point['x'] * w, point['y'] * h]]
        print(f"🎯 Pixel Point: {input_point}")

        sam = get_sam_model()
        results = sam.predict(img, points=input_point, labels=[1], verbose=False)

        if not results or results[0].masks is None:
            print("⚠️ SAM returned no masks.")
            return jsonify({"error": "No mask found"}), 400

        if len(results[0].masks.xyn) > 0:
            poly = results[0].masks.xyn[0].tolist()
            print(f"✨ SAM Success: {len(poly)} points found.")
            return jsonify({"polygon": poly})

        print("⚠️ SAM masks list is empty.")
        return jsonify({"error": "No mask found"}), 400
    except Exception as e:
        print(f"💥 FATAL ERROR IN SEGMENT: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/images/<subset>')
def list_images(subset):
    if subset not in VALID_SUBSETS:
        return jsonify({"error": "Invalid subset"}), 400

    if subset == "combined":
        # All images from all sources
        folders = [
            DATASET_ROOT / "super_focused_50/train/images",
            DATASET_ROOT / "nuevas_muestras_marzo",
            DATASET_ROOT / "dataset/images/train"
        ]
        images = []
        for folder in folders:
            if folder.exists():
                # Store relative to DATASET_ROOT for retrieval
                rel_path = folder.relative_to(DATASET_ROOT)
                images.extend([str(rel_path / f.name) for f in folder.glob("*.jpg")] + 
                              [str(rel_path / f.name) for f in folder.glob("*.png")])
        return jsonify(sorted(images))

    if subset == "super_focused_50":
        img_dir = DATASET_ROOT / "super_focused_50/train/images"
    elif subset == "nuevas_muestras_marzo":
        img_dir = DATASET_ROOT / "nuevas_muestras_marzo"
    elif subset == "dataset":
        img_dir = DATASET_ROOT / "dataset/images/train"
    else:
        img_dir = DATASET_ROOT / subset / "images"
        
    if not img_dir.exists():
        return jsonify({"error": "Images directory not found"}), 404

    images = [f.name for f in img_dir.glob("*.jpg")] + [f.name for f in img_dir.glob("*.png")]
    return jsonify(sorted(images))


@app.route('/api/image/<subset>/<path:filename>')
def get_image(subset, filename):
    if subset not in VALID_SUBSETS:
        return jsonify({"error": "Invalid subset"}), 400

    if subset == "combined":
        return send_from_directory(str(DATASET_ROOT), filename)

    if subset == "super_focused_50":
        safe_path = (DATASET_ROOT / "super_focused_50" / "train" / "images").absolute()
    elif subset == "nuevas_muestras_marzo":
        safe_path = (DATASET_ROOT / "nuevas_muestras_marzo").absolute()
    elif subset == "dataset":
        safe_path = (DATASET_ROOT / "dataset" / "images" / "train").absolute()
    else:
        safe_path = (DATASET_ROOT / subset / "images").absolute()
        
    return send_from_directory(str(safe_path), filename)


@app.route('/api/predict/<subset>/<path:filename>')
def predict(subset, filename):
    if subset not in VALID_SUBSETS:
        return jsonify({"error": "Invalid subset"}), 400

    if subset == "combined":
        img_path = DATASET_ROOT / filename
    else:
        if subset == "super_focused_50":
            base_path = DATASET_ROOT / "super_focused_50/train/images"
        elif subset == "nuevas_muestras_marzo":
            base_path = DATASET_ROOT / "nuevas_muestras_marzo"
        elif subset == "dataset":
            base_path = DATASET_ROOT / "dataset/images/train"
        else:
            base_path = DATASET_ROOT / subset / "images"
        img_path = safe_resolve(base_path, filename)

    if img_path is None or not img_path.exists():
        return "Image not found", 404

    results = model.predict(img_path, conf=0.15, verbose=False)[0]

    pred_filename = f"pred_{uuid.uuid4().hex}.jpg"
    pred_path = Path(tempfile.gettempdir()) / pred_filename

    img_plotted = results.plot(boxes=True, masks=True)
    cv2.imwrite(str(pred_path), img_plotted)

    @after_this_request
    def cleanup(response):
        try:
            os.remove(pred_path)
        except Exception:
            pass
        return response

    return send_from_directory(str(pred_path.parent), pred_filename)


@app.route('/api/predict_data/<subset>/<path:filename>')
def predict_data(subset, filename):
    if subset not in VALID_SUBSETS:
        return jsonify({"error": "Invalid subset"}), 400

    if subset == "combined":
        img_path = DATASET_ROOT / filename
    else:
        if subset == "super_focused_50":
            base_path = DATASET_ROOT / "super_focused_50/train/images"
        elif subset == "nuevas_muestras_marzo":
            base_path = DATASET_ROOT / "nuevas_muestras_marzo"
        elif subset == "dataset":
            base_path = DATASET_ROOT / "dataset/images/train"
        else:
            base_path = DATASET_ROOT / subset / "images"
        img_path = safe_resolve(base_path, filename)

    if img_path is None or not img_path.exists():
        return jsonify({"error": "Image not found"}), 404

    results = model.predict(img_path, conf=0.25, verbose=False)[0]
    detections = []
    
    if results.masks is not None:
        for mask, cls in zip(results.masks.xyn, results.boxes.cls):
            detections.append({
                "cls": int(cls),
                "points": mask.tolist()
            })
    else:
        for box, cls in zip(results.boxes.xyxyn, results.boxes.cls):
            x1, y1, x2, y2 = box.tolist()
            detections.append({
                "cls": int(cls),
                "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            })

    return jsonify(detections)


@app.route('/api/save_labels', methods=['POST'])
def save_labels():
    data = request.json
    subset = data.get('subset')
    filename = data.get('filename')
    labels = data.get('labels')

    if subset not in VALID_SUBSETS:
        return jsonify({"error": "Invalid subset"}), 400

    if labels is None or not isinstance(labels, list):
        return jsonify({"error": "Invalid labels format"}), 400

    label_filename = Path(filename).stem + ".txt"
    
    if subset == "combined":
        # Resolve where to save based on image location
        img_path = DATASET_ROOT / filename
        base_path = img_path.parent / "labels"
    elif subset == "super_focused_50":
        base_path = DATASET_ROOT / "super_focused_50/train/labels"
    elif subset == "nuevas_muestras_marzo":
        base_path = DATASET_ROOT / "nuevas_muestras_marzo/labels"
    elif subset == "dataset":
        base_path = DATASET_ROOT / "dataset/labels/train"
    else:
        base_path = DATASET_ROOT / subset / "labels"
        
    lab_path = safe_resolve(base_path, label_filename)
    if lab_path is None:
        return jsonify({"error": "Invalid file path"}), 403

    lab_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lab_path, 'w') as f:
        f.write("\n".join(labels))

    return jsonify({"status": "success"})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
