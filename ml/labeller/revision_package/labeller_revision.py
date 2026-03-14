import os
import json
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Configuración de rutas relativas para portabilidad
BASE_DIR = Path(__file__).parent
IMG_DIR = BASE_DIR / "imagenes"
LAB_DIR = BASE_DIR / "etiquetas_auto"

# Clases (YOLO format)
CLASSES = ["player", "goalkeeper", "referee", "ball", "equipo_a", "equipo_b"]

@app.route('/')
def index():
    return send_from_directory('.', 'labeller_revision.html')

@app.route('/api/images')
def list_images():
    if not IMG_DIR.exists():
        return jsonify([])
    images = [f.name for f in IMG_DIR.glob("*.jpg")] + [f.name for f in IMG_DIR.glob("*.png")]
    return jsonify(sorted(images))

@app.route('/api/image/<filename>')
def get_image(filename):
    return send_from_directory(str(IMG_DIR), filename)

@app.route('/api/load_labels/<filename>')
def load_labels(filename):
    label_file = LAB_DIR / (Path(filename).stem + ".txt")
    if not label_file.exists():
        return jsonify({"labels": []})
    
    try:
        lines = label_file.read_text(encoding="utf-8").splitlines()
        return jsonify({"labels": [l.strip() for l in lines if l.strip()]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/save_labels', methods=['POST'])
def save_labels():
    data = request.json
    filename = data.get('filename')
    labels = data.get('labels')
    
    if not filename or labels is None:
        return jsonify({"error": "Datos incompletos"}), 400
        
    label_file = LAB_DIR / (Path(filename).stem + ".txt")
    LAB_DIR.mkdir(exist_ok=True)
    
    try:
        label_file.write_text("\n".join(labels), encoding="utf-8")
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("-------------------------------------------")
    print("  EDUDIN REVISION TOOL - SERVIDOR ACTIVO")
    print("  Abra: http://127.0.0.1:5000")
    print("-------------------------------------------")
    app.run(port=5000, host='0.0.0.0')
