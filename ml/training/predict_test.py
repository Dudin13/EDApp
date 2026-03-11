"""
predict_test.py — Probar el modelo entrenado sobre una imagen o un frame de vídeo.

Uso:
    python train_yolo/predict_test.py --image C:/apped/frame_test.jpg
    python train_yolo/predict_test.py --video C:/ruta/video.mp4 --frame 3000
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODEL_PATH = Path(__file__).parent / "runs" / "detect" / "train" / "weights" / "best.pt"
CLASS_NAMES = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}
COLORS = {
    "player": (0, 220, 0),
    "goalkeeper": (255, 165, 0),
    "referee": (0, 0, 255),
    "ball": (0, 255, 255),
}


def predict_image(image_path: str, confidence: float = 0.4, save: bool = True):
    if not MODEL_PATH.exists():
        print(f"❌ Modelo no encontrado: {MODEL_PATH}")
        print("   Ejecuta primero: python train_yolo/train.py")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ pip install ultralytics")
        sys.exit(1)

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ No se pudo leer: {image_path}")
        sys.exit(1)

    model = YOLO(str(MODEL_PATH))
    results = model(img, conf=confidence, verbose=False)

    count = 0
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = CLASS_NAMES.get(cls, "unknown")
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = COLORS.get(name, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{name} {conf:.2f}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            count += 1

    print(f"✅ {count} detecciones en {image_path}")

    if save:
        out_path = str(Path(image_path).parent / (Path(image_path).stem + "_yolo_pred.jpg"))
        cv2.imwrite(out_path, img)
        print(f"   Guardado: {out_path}")

    return img, count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prueba el modelo YOLOv8 entrenado")
    parser.add_argument("--image", type=str, default=str(ROOT / "frame_test.jpg"))
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--frame", type=int, default=3175)
    parser.add_argument("--conf", type=float, default=0.4)
    args = parser.parse_args()

    if args.video:
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"❌ No se pudo leer el frame {args.frame} del vídeo")
            sys.exit(1)
        tmp = str(ROOT / "tmp_frame.jpg")
        cv2.imwrite(tmp, frame)
        predict_image(tmp, confidence=args.conf)
    else:
        predict_image(args.image, confidence=args.conf)
