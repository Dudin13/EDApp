"""
predict_test.py — Probar el modelo entrenado y generar informe visual HTML.

Uso:
    # Imagen unica
    python ml/training/predict_test.py --image C:/apped/data/samples/frame_test.jpg

    # Frame de un video
    python ml/training/predict_test.py --video C:/ruta/video.mp4 --frame 3000

    # Todas las imagenes de validacion (genera informe completo)
    python ml/training/predict_test.py --val

    # Cambiar umbral de confianza
    python ml/training/predict_test.py --val --conf 0.25
"""

import argparse
import sys
import cv2
import base64
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ── Rutas ─────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent.parent          # C:/apped
TRAIN_DIR  = Path(__file__).parent                        # ml/training
RUNS_DIR   = TRAIN_DIR / "runs"
VAL_DIR    = ROOT / "data" / "datasets" / "hybrid_dataset" / "valid" / "images"
OUTPUT_DIR = ROOT / "data" / "samples"

# Buscar el mejor modelo disponible (players_v1 primero, luego cualquier otro)
def find_best_model() -> Path | None:
    candidates = [
        RUNS_DIR / "segment" / "players_v1" / "weights" / "best.pt",
        RUNS_DIR / "segment" / "players_v1" / "weights" / "last.pt",
        RUNS_DIR / "detect"  / "players_v1" / "weights" / "best.pt",
    ]
    # Busqueda generica si no se encuentra ninguno
    for pt in RUNS_DIR.rglob("best.pt"):
        candidates.append(pt)
    for pt in RUNS_DIR.rglob("last.pt"):
        candidates.append(pt)

    for path in candidates:
        if path.exists():
            return path
    return None

# ── Clases y colores ───────────────────────────────────────────────────────
CLASS_NAMES = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}
COLORS_BGR  = {
    "player":     ( 50, 200,  50),   # verde
    "goalkeeper": ( 50, 165, 255),   # naranja
    "referee":    (255,  50,  50),   # azul
    "ball":       (  0, 255, 255),   # amarillo
    "unknown":    (180, 180, 180),   # gris
}
COLORS_HEX = {
    "player":     "#32c832",
    "goalkeeper": "#ffa532",
    "referee":    "#3232ff",
    "ball":       "#ffff00",
    "unknown":    "#b4b4b4",
}


# ── Deteccion sobre una imagen ─────────────────────────────────────────────

def predict_image(model, img_bgr, confidence: float = 0.25) -> tuple:
    """
    Corre el modelo sobre una imagen BGR y devuelve:
      - img_annotated: imagen con bboxes dibujadas
      - detections:    lista de dicts {name, conf, bbox}
      - counts:        dict {clase: cantidad}
    """
    results    = model(img_bgr, conf=confidence, verbose=False)
    detections = []
    counts     = defaultdict(int)

    annotated = img_bgr.copy()

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls  = int(box.cls[0])
            name = CLASS_NAMES.get(cls, "unknown")
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = COLORS_BGR.get(name, (180, 180, 180))

            # Rectangulo
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Etiqueta con fondo
            label     = f"{name} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            label_y   = max(y1 - 4, th + 4)
            cv2.rectangle(annotated, (x1, label_y - th - 4), (x1 + tw + 4, label_y), color, -1)
            cv2.putText(annotated, label, (x1 + 2, label_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            detections.append({"name": name, "conf": conf,
                                "bbox": (x1, y1, x2, y2)})
            counts[name] += 1

    return annotated, detections, dict(counts)


def frame_to_b64(img_bgr, max_width: int = 800) -> str:
    """Redimensiona y convierte a base64 para el HTML."""
    h, w = img_bgr.shape[:2]
    if w > max_width:
        scale = max_width / w
        img_bgr = cv2.resize(img_bgr, (max_width, int(h * scale)))
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode()


# ── Generador de informe HTML ──────────────────────────────────────────────

def generate_html_report(results_data: list, model_path: Path,
                         confidence: float, output_path: Path):
    """
    Genera un informe HTML con:
    - Resumen global por clase
    - Cada imagen con sus detecciones
    - Barra de color por clase
    """
    total_counts = defaultdict(int)
    for r in results_data:
        for cls, cnt in r["counts"].items():
            total_counts[cls] += cnt

    total_dets = sum(total_counts.values())
    n_images   = len(results_data)

    # ── Tarjetas de resumen ────
    summary_cards = ""
    for cls in ["player", "goalkeeper", "referee", "ball"]:
        cnt   = total_counts.get(cls, 0)
        color = COLORS_HEX.get(cls, "#888")
        avg   = cnt / n_images if n_images else 0
        summary_cards += f"""
        <div class="card">
            <div class="card-dot" style="background:{color}"></div>
            <div class="card-cls">{cls}</div>
            <div class="card-count">{cnt}</div>
            <div class="card-avg">avg {avg:.1f}/img</div>
        </div>"""

    # ── Filas de imagenes ────
    image_rows = ""
    for r in results_data:
        det_badges = ""
        for cls in ["player", "goalkeeper", "referee", "ball"]:
            cnt = r["counts"].get(cls, 0)
            if cnt:
                color = COLORS_HEX.get(cls, "#888")
                det_badges += f'<span class="badge" style="background:{color}">{cls} {cnt}</span>'
        if not det_badges:
            det_badges = '<span class="badge" style="background:#555">sin detecciones</span>'

        image_rows += f"""
        <div class="img-row">
            <div class="img-meta">
                <div class="img-name">{r['name']}</div>
                <div class="img-badges">{det_badges}</div>
                <div class="img-total">{sum(r['counts'].values())} detecciones totales</div>
            </div>
            <img src="data:image/jpeg;base64,{r['b64']}" class="pred-img">
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>EDApp — Informe de Predicciones</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e0e0e0; margin: 0; padding: 20px; }}
  h1   {{ color: #fff; margin-bottom: 4px; }}
  .meta {{ color: #888; font-size: 13px; margin-bottom: 24px; }}
  .summary {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 32px; }}
  .card {{ background: #1e2130; border-radius: 10px; padding: 16px 20px; min-width: 140px; }}
  .card-dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 6px; }}
  .card-cls {{ font-size: 13px; color: #aaa; margin-bottom: 4px; }}
  .card-count {{ font-size: 32px; font-weight: bold; color: #fff; }}
  .card-avg {{ font-size: 12px; color: #666; }}
  .img-row {{ background: #1e2130; border-radius: 10px; padding: 16px; margin-bottom: 16px;
              display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap; }}
  .img-meta {{ min-width: 180px; }}
  .img-name {{ font-weight: bold; font-size: 14px; margin-bottom: 8px; word-break: break-all; }}
  .img-badges {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }}
  .badge {{ border-radius: 4px; padding: 2px 8px; font-size: 12px; color: #000; font-weight: bold; }}
  .img-total {{ font-size: 12px; color: #666; }}
  .pred-img {{ max-width: 800px; width: 100%; border-radius: 6px; }}
  .legend {{ display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }}
  .leg-item {{ display: flex; align-items: center; gap: 6px; font-size: 13px; }}
  .leg-dot {{ width: 14px; height: 14px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>EDApp — Informe de Predicciones</h1>
<div class="meta">
  Modelo: {model_path.name} &nbsp;|&nbsp;
  Ruta: {model_path} &nbsp;|&nbsp;
  Confianza: {confidence:.0%} &nbsp;|&nbsp;
  Imagenes: {n_images} &nbsp;|&nbsp;
  Total detecciones: {total_dets} &nbsp;|&nbsp;
  Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>

<div class="legend">
  {''.join(f'<div class="leg-item"><div class="leg-dot" style="background:{COLORS_HEX[c]}"></div>{c}</div>' for c in ["player","goalkeeper","referee","ball"])}
</div>

<div class="summary">
  <div class="card">
    <div class="card-cls">total detecciones</div>
    <div class="card-count">{total_dets}</div>
    <div class="card-avg">{total_dets/n_images:.1f}/img</div>
  </div>
  {summary_cards}
</div>

<div id="images">
  {image_rows}
</div>

</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"[+] Informe guardado: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prueba el modelo YOLO entrenado")
    parser.add_argument("--image",  type=str,   default=None,  help="Ruta a imagen")
    parser.add_argument("--video",  type=str,   default=None,  help="Ruta a video")
    parser.add_argument("--frame",  type=int,   default=3000,  help="Numero de frame del video")
    parser.add_argument("--val",    action="store_true",       help="Evaluar sobre imagenes de validacion")
    parser.add_argument("--conf",   type=float, default=0.25,  help="Umbral de confianza (0-1)")
    parser.add_argument("--limit",  type=int,   default=20,    help="Max imagenes en modo --val")
    parser.add_argument("--model",  type=str,   default=None,  help="Ruta manual al modelo .pt")
    parser.add_argument("--no-html",action="store_true",       help="No generar informe HTML")
    args = parser.parse_args()

    # ── Cargar modelo ────
    model_path = Path(args.model) if args.model else find_best_model()
    if model_path is None or not model_path.exists():
        print("[!] No se encontro ningun modelo entrenado.")
        print("    Entrena primero con: python ml/training/train_unified.py --target players")
        print("    O especifica la ruta con: --model ruta/al/best.pt")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[!] pip install ultralytics")
        sys.exit(1)

    print(f"[+] Cargando modelo: {model_path}")
    model = YOLO(str(model_path))

    # ── Recopilar imagenes a procesar ────
    images_to_process = []

    if args.val:
        if not VAL_DIR.exists():
            print(f"[!] No se encontro el directorio de validacion: {VAL_DIR}")
            sys.exit(1)
        img_files = sorted(VAL_DIR.glob("*.jpg"))[:args.limit]
        img_files += sorted(VAL_DIR.glob("*.png"))[:max(0, args.limit - len(img_files))]
        images_to_process = img_files[:args.limit]
        print(f"[+] Modo validacion: {len(images_to_process)} imagenes")

    elif args.video:
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"[!] No se pudo leer el frame {args.frame}")
            sys.exit(1)
        tmp = ROOT / "tmp_frame.jpg"
        cv2.imwrite(str(tmp), frame)
        images_to_process = [tmp]
        print(f"[+] Frame {args.frame} extraido del video")

    elif args.image:
        images_to_process = [Path(args.image)]

    else:
        # Intentar usar imagenes de data/samples
        samples = list((ROOT / "data" / "samples").glob("*.jpg"))
        if samples:
            images_to_process = samples[:5]
            print(f"[+] Usando {len(images_to_process)} imagenes de data/samples/")
        else:
            print("[!] Especifica --image, --video, o --val")
            parser.print_help()
            sys.exit(1)

    if not images_to_process:
        print("[!] No se encontraron imagenes para procesar")
        sys.exit(1)

    # ── Procesar imagenes ────
    results_data = []
    total_counts = defaultdict(int)

    for i, img_path in enumerate(images_to_process):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[!] No se pudo leer: {img_path}")
            continue

        annotated, detections, counts = predict_image(model, img, args.conf)

        # Guardar imagen anotada
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"{img_path.stem}_pred.jpg"
        cv2.imwrite(str(out_path), annotated)

        for cls, cnt in counts.items():
            total_counts[cls] += cnt

        # Resumen en consola
        summary = " | ".join(f"{k}:{v}" for k, v in counts.items()) or "sin detecciones"
        print(f"  [{i+1}/{len(images_to_process)}] {img_path.name} → {summary}")

        results_data.append({
            "name":   img_path.name,
            "counts": counts,
            "b64":    frame_to_b64(annotated) if not args.no_html else "",
        })

    # ── Resumen final en consola ────
    print("\n" + "="*50)
    print(f"RESUMEN — {len(results_data)} imagenes procesadas")
    print("="*50)
    for cls in ["player", "goalkeeper", "referee", "ball"]:
        cnt = total_counts.get(cls, 0)
        avg = cnt / len(results_data) if results_data else 0
        bar = "█" * min(int(avg), 40)
        print(f"  {cls:<12} {cnt:>5} total  {avg:>5.1f}/img  {bar}")
    print("="*50)

    # ── Generar informe HTML ────
    if not args.no_html and results_data:
        html_path = ROOT / "predicciones_edapp.html"
        generate_html_report(results_data, model_path, args.conf, html_path)
        print(f"\n[+] Abre el informe en el navegador:")
        print(f"    {html_path}")


if __name__ == "__main__":
    main()
