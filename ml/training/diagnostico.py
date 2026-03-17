"""
diagnostico.py
==============
Analiza el estado del dataset y del modelo best.pt.
Genera un reporte HTML en C:/apped/diagnostico_edapp.html

Ejecutar desde la raiz del repo:
    python ml/training/diagnostico.py
"""

import os
import random
import base64
from pathlib import Path
from collections import Counter
from datetime import datetime

BASE         = Path(os.environ.get("APPED_ROOT", "C:/apped"))
DATASET_ROOT = BASE / "data" / "datasets" / "hybrid_dataset"
TRAIN_IMGS   = DATASET_ROOT / "train" / "images"
TRAIN_LABS   = DATASET_ROOT / "train" / "labels"
VAL_IMGS     = DATASET_ROOT / "valid" / "images"
VAL_LABS     = DATASET_ROOT / "valid" / "labels"
RUNS_DIR     = BASE / "ml" / "training" / "runs"
OUTPUT_HTML  = BASE / "diagnostico_edapp.html"

CLASS_NAMES  = {0: "Goalkeeper", 1: "Player", 2: "ball", 3: "referee"}
CLASS_COLORS = {0: "#f59e0b",    1: "#3b82f6", 2: "#10b981", 3: "#ef4444"}


# ── Analisis dataset ───────────────────────────────────────────────────────

def count_images(folder):
    if not folder.exists():
        return 0
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})

def analyze_labels(labels_dir):
    if not labels_dir.exists():
        return {"total": 0, "empty": 0, "corrupt": 0, "class_counts": Counter(), "instances": 0}
    total = empty = corrupt = instances = 0
    class_counts = Counter()
    for lf in labels_dir.glob("*.txt"):
        total += 1
        try:
            lines = [l.strip() for l in lf.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
            if not lines:
                empty += 1
                continue
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    class_counts[int(float(parts[0]))] += 1
                    instances += 1
                else:
                    corrupt += 1
        except Exception:
            corrupt += 1
    return {"total": total, "empty": empty, "corrupt": corrupt,
            "class_counts": class_counts, "instances": instances}

def find_best_pts():
    models = []
    if not RUNS_DIR.exists():
        return models
    for pt in sorted(RUNS_DIR.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
        size_mb = pt.stat().st_size / 1024 / 1024
        mtime   = datetime.fromtimestamp(pt.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        metrics = {}
        results_csv = pt.parent.parent / "results.csv"
        if results_csv.exists():
            try:
                import csv
                rows = list(csv.DictReader(results_csv.read_text(encoding="utf-8").splitlines()))
                if rows:
                    metrics = {k.strip(): v.strip() for k, v in rows[-1].items()}
            except Exception:
                pass
        models.append({"path": pt, "size_mb": size_mb, "modified": mtime, "metrics": metrics})
    return models

def run_val(model_path, data_yaml):
    try:
        from ultralytics import YOLO
        model   = YOLO(str(model_path))
        metrics = model.val(data=str(data_yaml), verbose=False, workers=0)
        result  = {
            "map50":    round(float(metrics.box.map50), 3),
            "map5095":  round(float(metrics.box.map),   3),
            "precision":round(float(metrics.box.mp),    3),
            "recall":   round(float(metrics.box.mr),    3),
            "per_class": {},
        }
        if hasattr(metrics.box, "ap_class_index"):
            for i, cls_idx in enumerate(metrics.box.ap_class_index):
                ap = float(metrics.box.ap[i]) if i < len(metrics.box.ap) else 0.0
                result["per_class"][int(cls_idx)] = round(ap, 3)
        return result
    except ImportError:
        return {"error": "ultralytics no instalado: pip install ultralytics"}
    except Exception as e:
        return {"error": str(e)}

def predict_samples(model_path, n=6):
    """Corre prediccion sobre n imagenes y devuelve imagen con boxes dibujadas en base64."""
    try:
        import cv2
        from ultralytics import YOLO
        model  = YOLO(str(model_path))
        imgs   = list(TRAIN_IMGS.glob("*.jpg")) + list(TRAIN_IMGS.glob("*.png"))
        if not imgs:
            return []
        sample  = random.sample(imgs, min(n, len(imgs)))
        results = []
        colors  = {0: (251,146,60), 1: (59,130,246), 2: (16,185,129), 3: (239,68,68)}
        for img_path in sample:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            r = model.predict(str(img_path), verbose=False, conf=0.20, workers=0)[0]
            n_det = 0
            if r.boxes is not None:
                for box in r.boxes:
                    cls  = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1,y1,x2,y2 = [int(v) for v in box.xyxy[0]]
                    col = colors.get(cls, (200,200,200))
                    cv2.rectangle(img, (x1,y1), (x2,y2), col, 2)
                    label = f"{CLASS_NAMES.get(cls,'?')} {conf:.2f}"
                    cv2.putText(img, label, (x1, max(y1-6,10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
                    n_det += 1
            # Resize para el reporte
            max_w = 480
            if w > max_w:
                scale = max_w / w
                img   = cv2.resize(img, (max_w, int(h*scale)))
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64 = base64.b64encode(buf).decode()
            results.append({"name": img_path.name, "b64": b64, "n_det": n_det})
        return results
    except ImportError as e:
        return [{"error": f"Falta libreria: {e}"}]
    except Exception as e:
        return [{"error": str(e)}]


# ── HTML ───────────────────────────────────────────────────────────────────

def color_val(v, good=0.6, warn=0.35):
    if v >= good:  return "#10b981"
    if v >= warn:  return "#f59e0b"
    return "#ef4444"

def bar_html(v, max_v=1.0):
    pct = min(100, int(v / max_v * 100))
    col = color_val(v)
    return f'<div style="background:#e5e7eb;border-radius:4px;height:8px;margin-top:4px"><div style="background:{col};width:{pct}%;height:100%;border-radius:4px"></div></div>'

def build_html(train_stats, val_stats, models, val_result, samples):
    # -- dataset section
    has_val = val_stats["total"] > 0
    val_warn = "" if has_val else """
    <div style="background:#fef3c7;border:1px solid #f59e0b;border-radius:8px;padding:12px;margin:16px 0;color:#92400e;font-size:14px">
      <b>Sin split de validacion</b> &mdash; ejecuta <code>create_val_split.py</code> antes de entrenar o las metricas de mAP seran falsas.
    </div>"""

    cls_rows = ""
    all_cls = sorted(set(list(train_stats["class_counts"].keys()) + list(val_stats["class_counts"].keys())))
    for cls_id in all_cls:
        name  = CLASS_NAMES.get(cls_id, f"clase_{cls_id}")
        color = CLASS_COLORS.get(cls_id, "#888")
        tr    = train_stats["class_counts"].get(cls_id, 0)
        vr    = val_stats["class_counts"].get(cls_id, 0)
        cls_rows += f"""<tr>
          <td style="padding:6px 8px"><span style="display:inline-block;width:10px;height:10px;background:{color};border-radius:2px;margin-right:6px;vertical-align:middle"></span>{name}</td>
          <td style="padding:6px 8px;text-align:right">{tr:,}</td>
          <td style="padding:6px 8px;text-align:right">{vr:,}</td>
          <td style="padding:6px 8px;text-align:right;font-weight:600">{tr+vr:,}</td>
        </tr>"""

    # -- model section
    model_cards = ""
    for m in models[:3]:
        met   = m["metrics"]
        map50 = None
        for k in met:
            if "mAP50" in k and "95" not in k:
                try: map50 = float(met[k]); break
                except: pass
        badge = ""
        if map50 is not None:
            col   = color_val(map50)
            badge = f'<span style="background:{col};color:#fff;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600">mAP50: {map50:.3f}</span>'
        model_cards += f"""
        <div style="border:1px solid #e5e7eb;border-radius:8px;padding:14px;margin:8px 0;font-size:13px">
          <div style="color:#6b7280;margin-bottom:6px;word-break:break-all">{m['path']}</div>
          <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
            <span>{m['size_mb']:.1f} MB</span>
            <span style="color:#9ca3af">|</span>
            <span>{m['modified']}</span>
            {badge}
          </div>
        </div>"""
    if not model_cards:
        model_cards = '<p style="color:#ef4444">No se encontro ningun best.pt en train_yolo/runs/</p>'

    # -- validation section
    val_html = ""
    if val_result:
        if "error" in val_result:
            val_html = f'<div style="color:#ef4444;font-size:14px">Error al validar: {val_result["error"]}</div>'
        else:
            pc_rows = ""
            for cls_id, ap in sorted(val_result.get("per_class", {}).items()):
                name  = CLASS_NAMES.get(cls_id, f"clase_{cls_id}")
                color = CLASS_COLORS.get(cls_id, "#888")
                pc_rows += f"""<tr>
                  <td style="padding:6px 8px"><span style="display:inline-block;width:10px;height:10px;background:{color};border-radius:2px;margin-right:6px;vertical-align:middle"></span>{name}</td>
                  <td style="padding:6px 8px;text-align:right;font-weight:600;color:{color_val(ap)}">{ap:.3f}</td>
                  <td style="padding:6px 8px;width:200px">{bar_html(ap)}</td>
                </tr>"""
            kpis = ""
            for label, val, g, w in [
                ("mAP50",    val_result["map50"],    0.60, 0.35),
                ("mAP50-95", val_result["map5095"],  0.40, 0.20),
                ("Precision",val_result["precision"],0.65, 0.40),
                ("Recall",   val_result["recall"],   0.60, 0.35),
            ]:
                col = color_val(val, g, w)
                kpis += f"""<div style="background:#f9fafb;border-radius:8px;padding:14px;text-align:center">
                  <div style="font-size:26px;font-weight:700;color:{col}">{val:.3f}</div>
                  <div style="font-size:12px;color:#6b7280;margin-top:2px">{label}</div>
                </div>"""
            val_html = f"""
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0">{kpis}</div>
            <table style="width:100%;border-collapse:collapse;font-size:13px;margin-top:8px">
              <thead><tr style="color:#6b7280;border-bottom:1px solid #e5e7eb">
                <th style="text-align:left;padding:6px 8px">Clase</th>
                <th style="text-align:right;padding:6px 8px">AP</th>
                <th style="padding:6px 8px">Barra</th>
              </tr></thead>
              <tbody>{pc_rows}</tbody>
            </table>"""

    # -- samples section
    sample_html = ""
    for s in samples:
        if "error" in s:
            sample_html += f'<div style="color:#ef4444;font-size:13px">{s["error"]}</div>'
        else:
            sample_html += f"""
            <div style="text-align:center">
              <img src="data:image/jpeg;base64,{s['b64']}" style="max-width:100%;border-radius:6px;border:1px solid #e5e7eb">
              <div style="font-size:11px;color:#6b7280;margin-top:4px">{s['name']} &mdash; {s['n_det']} detecciones</div>
            </div>"""

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<title>Diagnostico EDApp</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0 }}
  body {{ font-family: system-ui, sans-serif; background:#f3f4f6; color:#111827; padding:24px }}
  .card {{ background:#fff; border-radius:12px; padding:24px; margin-bottom:20px; box-shadow:0 1px 3px rgba(0,0,0,.08) }}
  h1 {{ font-size:22px; font-weight:700; margin-bottom:4px }}
  h2 {{ font-size:16px; font-weight:600; margin-bottom:16px; padding-bottom:8px; border-bottom:2px solid #f3f4f6 }}
  table {{ width:100%; border-collapse:collapse; font-size:13px }}
  th {{ text-align:left; padding:6px 8px; color:#6b7280; border-bottom:1px solid #e5e7eb; font-weight:500 }}
  code {{ background:#f3f4f6; padding:1px 6px; border-radius:4px; font-size:12px }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px }}
  .grid3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px }}
  @media(max-width:700px) {{ .grid2,.grid3 {{ grid-template-columns:1fr }} }}
</style>
</head><body>

<div class="card">
  <h1>Diagnostico EDApp</h1>
  <p style="color:#6b7280;font-size:13px;margin-top:4px">Generado el {now} &mdash; Club de Cantera</p>
</div>

<div class="grid2">
  <div class="card">
    <h2>Dataset</h2>
    {val_warn}
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px">
      <div style="background:#eff6ff;border-radius:8px;padding:12px;text-align:center">
        <div style="font-size:28px;font-weight:700;color:#3b82f6">{train_stats['total']:,}</div>
        <div style="font-size:12px;color:#6b7280">imagenes train</div>
      </div>
      <div style="background:{'#f0fdf4' if has_val else '#fef2f2'};border-radius:8px;padding:12px;text-align:center">
        <div style="font-size:28px;font-weight:700;color:{'#10b981' if has_val else '#ef4444'}">{val_stats['total']:,}</div>
        <div style="font-size:12px;color:#6b7280">imagenes val</div>
      </div>
    </div>
    <table>
      <thead><tr><th>Clase</th><th style="text-align:right">Train</th><th style="text-align:right">Val</th><th style="text-align:right">Total</th></tr></thead>
      <tbody>{cls_rows}</tbody>
    </table>
    <div style="margin-top:12px;font-size:12px;color:#6b7280">
      Labels vacias train: {train_stats['empty']} &nbsp;|&nbsp; Corruptas: {train_stats['corrupt']}
    </div>
  </div>

  <div class="card">
    <h2>Modelos encontrados</h2>
    {model_cards}
  </div>
</div>

<div class="card">
  <h2>Metricas del mejor modelo (validacion real)</h2>
  {val_html if val_html else '<p style="color:#9ca3af;font-size:14px">No se pudo ejecutar la validacion.</p>'}
</div>

<div class="card">
  <h2>Predicciones de muestra (conf > 0.20)</h2>
  <div class="grid3">{sample_html if sample_html else '<p style="color:#9ca3af">Sin imagenes de muestra.</p>'}</div>
  <div style="margin-top:12px;font-size:12px;color:#6b7280">
    <span style="display:inline-block;width:10px;height:10px;background:#f59e0b;border-radius:2px;margin-right:4px;vertical-align:middle"></span>Goalkeeper
    <span style="display:inline-block;width:10px;height:10px;background:#3b82f6;border-radius:2px;margin:0 4px 0 12px;vertical-align:middle"></span>Player
    <span style="display:inline-block;width:10px;height:10px;background:#10b981;border-radius:2px;margin:0 4px 0 12px;vertical-align:middle"></span>Ball
    <span style="display:inline-block;width:10px;height:10px;background:#ef4444;border-radius:2px;margin:0 4px 0 12px;vertical-align:middle"></span>Referee
  </div>
</div>

</body></html>"""


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  EDApp - Diagnostico del dataset y modelo")
    print("="*50)

    print("\n[1/4] Analizando dataset...")
    train_stats = analyze_labels(TRAIN_LABS)
    train_stats["total"] = count_images(TRAIN_IMGS)
    val_stats   = analyze_labels(VAL_LABS)
    val_stats["total"]   = count_images(VAL_IMGS)
    print(f"  Train: {train_stats['total']} imagenes, {train_stats['instances']} instancias")
    print(f"  Val:   {val_stats['total']} imagenes, {val_stats['instances']} instancias")

    print("\n[2/4] Buscando modelos entrenados...")
    models = find_best_pts()
    for m in models:
        print(f"  {m['path'].name}  ({m['size_mb']:.1f} MB)  {m['modified']}")
    if not models:
        print("  No se encontro ningun best.pt")

    val_result = {}
    if models:
        data_yaml = DATASET_ROOT / "data.yaml"
        if data_yaml.exists():
            print(f"\n[3/4] Validando {models[0]['path'].name}...")
            val_result = run_val(models[0]["path"], data_yaml)
            if "error" in val_result:
                print(f"  Error: {val_result['error']}")
            else:
                print(f"  mAP50: {val_result['map50']}  |  mAP50-95: {val_result['map5095']}")
        else:
            print("\n[3/4] Sin data.yaml, saltando validacion")
    else:
        print("\n[3/4] Sin modelo, saltando validacion")

    print("\n[4/4] Generando imagenes de muestra...")
    samples = []
    if models and TRAIN_IMGS.exists():
        samples = predict_samples(models[0]["path"], n=6)
        print(f"  {len(samples)} imagenes procesadas")
    else:
        print("  Sin modelo o sin imagenes")

    print("\nGenerando reporte HTML...")
    html = build_html(train_stats, val_stats, models, val_result, samples)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nReporte guardado en: {OUTPUT_HTML}")
    print("Abrelo en el navegador para ver los resultados.\n")
