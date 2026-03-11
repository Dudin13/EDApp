import torch
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import os

model_config = "C:/apped/ED/Lib/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py"
model_weights = "groundingdino_swint_ogc.pth"

if not os.path.exists(model_weights):
    import urllib.request
    print("Descargando modelo Grounding DINO...")
    url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    urllib.request.urlretrieve(url, model_weights)
    print("Descargado.")

print("Cargando modelo...")
model = load_model(model_config, model_weights)

frames_dir = "C:/apped/frames_etiquetado"
output_dir = "C:/apped/frames_anotados"
labels_dir = "C:/apped/labels_yolo"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

TEXT_PROMPT = "football player . goalkeeper . referee . ball"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

frames = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
print(f"Procesando {len(frames)} frames...")

for i, fname in enumerate(frames):
    fpath = os.path.join(frames_dir, fname)
    image_source, image = load_image(fpath)
    h, w = image_source.shape[:2]

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device="cpu"
    )

    label_file = os.path.join(labels_dir, fname.replace('.jpg', '.txt'))
    clase_map = {'football player': 0, 'goalkeeper': 1, 'referee': 2, 'ball': 3}

    with open(label_file, 'w') as f:
        for box, phrase in zip(boxes, phrases):
            clase = 0
            for key, val in clase_map.items():
                if key in phrase.lower():
                    clase = val
                    break
            cx, cy, bw, bh = box.tolist()
            f.write(f"{clase} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    annotated = annotate(image_source=image_source, boxes=boxes,
                        logits=logits, phrases=phrases)
    cv2.imwrite(os.path.join(output_dir, fname), annotated)
    print(f"[{i+1}/{len(frames)}] {fname} — {len(boxes)} detecciones")

print(f"\nListo. Etiquetas en: {labels_dir}")
print(f"Imagenes anotadas en: {output_dir}")