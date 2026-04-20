"""
detector.py — Motor de detección con arquitectura de 3 modelos especializados.

ARQUITECTURA (inspirada en notebook Kaggle akrambelha):
  - PLAYER_MODEL: YOLO para jugadores, porteros y árbitros (resolución estándar)
  - BALL_MODEL:   YOLO para el balón a resolución 1280px (60% mejor detección)
  - PITCH_MODEL:  YOLO pose para keypoints del campo (homografía automática)

CLASIFICACIÓN DE EQUIPOS:
  - Modo 'siglip': SigLIP embeddings + UMAP + KMeans (SOTA, robusto con VEO)
  - Modo 'kmeans': KMeans RGB fallback (rápido, funciona sin GPU grande)
  - Portero: asignado por proximidad geométrica al centroide de su equipo

COMPATIBILIDAD: API pública (detect_frame, classify_team, auto_detect_team_colors)
  totalmente compatible con video_processor.py existente.

MODELOS PREENTRENADOS (descargar con scripts/download_kaggle_models.py):
  - detect_players.pt  -> jugadores/portero/arbitro
  - detect_ball.pt     -> balon a 1280px
  - pose_field.pt      -> keypoints del campo (opcional, para calibracion auto)
"""

import cv2
import numpy as np
import os
import logging
import threading
from pathlib import Path
from collections import Counter
from sklearn.cluster import KMeans

from modules.calibration_pnl import PnLCalibrator
from modules.identity_reader import IdentityReader

logger = logging.getLogger(__name__)

_calibrator = PnLCalibrator()
_id_reader  = IdentityReader()

# ── Fix 5: Rutas de modelos — portables, sin hardcodear c:/apped ───────────
# Calculamos rutas relativas al propio archivo (detector.py está en app/modules/).
# Buscamos en varios candidatos por orden de prioridad.
_THIS_DIR  = Path(__file__).resolve().parent          # app/modules/
_APP_ROOT  = _THIS_DIR.parent                         # app/
_REPO_ROOT = _APP_ROOT.parent                         # raíz del repo

_MODEL_SEARCH_PATHS = [
    _THIS_DIR,                                        # junto al detector.py (legacy)
    _REPO_ROOT / "models",                           # la nueva carpeta centralizada
    _REPO_ROOT / "ml" / "models",                    # estructura ml/models/
    _REPO_ROOT / "assets" / "weights",               # estructura assets/weights/
    _REPO_ROOT,                                       # raíz del repo (fallback)
]

def _find_model(filename: str) -> Path:
    """Busca un archivo de modelo en los candidatos predefinidos. Retorna el
    primer Path que exista, o el primer candidato + filename si ninguno existe
    (permite mensajes de error con la ruta esperada)."""
    for base in _MODEL_SEARCH_PATHS:
        candidate = base / filename
        if candidate.exists():
            return candidate
    return _MODEL_SEARCH_PATHS[1] / filename  # ruta esperada aunque no exista

PLAYER_MODEL_PATH = _find_model("players.pt")
BALL_MODEL_PATH   = _find_model("ball.pt")
PITCH_MODEL_PATH  = _find_model("pitch.pt")
YOLO_LEGACY_PATH  = _find_model("best_football_seg.pt")
YOLO_COCO_MODEL   = str(_find_model("yolov8n.pt"))

PLAYER_ID=0; GOALKEEPER_ID=1; REFEREE_ID=2; BALL_ID=3
VALID_CLASSES = {"player","goalkeeper","referee","ball"}

ROBOFLOW_API_KEY   = os.getenv("ROBOFLOW_API_KEY","")
ROBOFLOW_WORKSPACE = "roboflow-jvuqo"
ROBOFLOW_PROJECT   = "football-players-detection-3zvbc"
ROBOFLOW_VERSION   = 1

_player_model=None; _ball_model=None; _pitch_model=None
_coco_model=None; _roboflow_model=None
_player_lock=threading.Lock(); _ball_lock=threading.Lock()
_pitch_lock=threading.Lock(); _coco_lock=threading.Lock()
_roboflow_lock=threading.Lock()


def _load_player_model():
    global _player_model
    with _player_lock:
        if _player_model is None:
            from ultralytics import YOLO
            path = PLAYER_MODEL_PATH if PLAYER_MODEL_PATH.exists() else YOLO_LEGACY_PATH
            if path.exists():
                _player_model = YOLO(str(path))
            else:
                _player_model = _load_coco_model()
    return _player_model


def _load_ball_model():
    global _ball_model
    with _ball_lock:
        if _ball_model is None:
            from ultralytics import YOLO
            if BALL_MODEL_PATH.exists():
                _ball_model = YOLO(str(BALL_MODEL_PATH))
                logger.info("Modelo balon: detect_ball.pt (1280px)")
            elif YOLO_LEGACY_PATH.exists():
                _ball_model = YOLO(str(YOLO_LEGACY_PATH))
                logger.warning("detect_ball.pt no encontrado, usando best_football_seg.pt para balon")
    return _ball_model


def _load_pitch_model():
    global _pitch_model
    with _pitch_lock:
        if _pitch_model is None:
            from ultralytics import YOLO
            if PITCH_MODEL_PATH.exists():
                _pitch_model = YOLO(str(PITCH_MODEL_PATH))
                logger.info("Modelo campo: pose_field.pt")
    return _pitch_model


def _load_coco_model():
    global _coco_model
    with _coco_lock:
        if _coco_model is None:
            from ultralytics import YOLO
            _coco_model = YOLO(YOLO_COCO_MODEL)
    return _coco_model


def _load_roboflow():
    global _roboflow_model
    with _roboflow_lock:
        if _roboflow_model is None:
            from roboflow import Roboflow
            rf = Roboflow(api_key=ROBOFLOW_API_KEY)
            project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
            _roboflow_model = project.version(ROBOFLOW_VERSION).model
    return _roboflow_model


def kaggle_models_available():
    return PLAYER_MODEL_PATH.exists() and BALL_MODEL_PATH.exists()

def yolo_model_available():
    return PLAYER_MODEL_PATH.exists() or YOLO_LEGACY_PATH.exists()


def _build_detection_dict(frame, x_abs, y_abs, w, h, clase, confidence,
                           extract_dorsal=False):
    return {
        "x":int(x_abs),"y":int(y_abs),"w":int(w),"h":int(h),
        "bbox": (int(x_abs - w/2), int(y_abs - h/2), int(x_abs + w/2), int(y_abs + h/2)),
        "clase":clase,"confianza":round(confidence,3),"conf":round(confidence,3),
        "torso_color":None,  # Eval. Lazy: Solo calcula KMeans si un clasificador lo pide explicitamente
        "pitch_coords":_calibrator.transform_point(int(x_abs),int(y_abs)),
        "dorsal":None,"mask":None,
    }


def detect_frame_kaggle(frame, confidence=0.3, imgsz=None):
    """
    Deteccion con 3 modelos especializados del notebook Kaggle.
    PLAYER_MODEL: jugadores/porteros/arbitros a resolucion estandar.
    BALL_MODEL:   balon a 1280px — detecta balones de ~8px en VEO panoramica.
    """
    import supervision as sv
    h_frame,w_frame=frame.shape[:2]
    detecciones=[]
    id_to_name={BALL_ID:"ball",GOALKEEPER_ID:"goalkeeper",PLAYER_ID:"player",REFEREE_ID:"referee"}

    # 1. Jugadores / porteros / arbitros
    pm=_load_player_model()
    if pm is not None:
        try:
            import torch
            use_half = torch.cuda.is_available()
            p_imgsz = imgsz if imgsz else 640
            r=pm.predict(frame,conf=0.40,verbose=False,imgsz=p_imgsz, half=use_half)[0]
            import supervision as sv
            d=sv.Detections.from_ultralytics(r).with_nms(threshold=0.5,class_agnostic=True)
            for i,xyxy in enumerate(d.xyxy):
                x1,y1,x2,y2=map(int,xyxy)
                w=x2-x1; h=y2-y1; cx=(x1+x2)//2; cy=(y1+y2)//2
                cls_id=int(d.class_id[i]) if d.class_id is not None else PLAYER_ID
                conf_v=float(d.confidence[i]) if d.confidence is not None else 0.40
                clase=id_to_name.get(cls_id,"player")
                if clase=="ball": continue
                if cy<h_frame*0.10 or w*h<40 or h<20: continue
                if h>0 and (h/max(w,1))>7.5: continue
                detecciones.append(_build_detection_dict(frame,cx,cy,w,h,clase,conf_v))

        except Exception as e:
            logger.error(f"Error modelo jugadores: {e}")

    # 2. Balon a 1280px
    bm=_load_ball_model()
    if bm is not None:
        try:
            import torch
            use_half = torch.cuda.is_available()
            scale=1280/w_frame
            fhd=cv2.resize(frame,(1280,int(h_frame*scale)))
            rb=bm.predict(fhd,conf=0.1,verbose=False,imgsz=1280, half=use_half)[0]
            db=sv.Detections.from_ultralytics(rb).with_nms(threshold=0.3,class_agnostic=True)
            for i,xyxy in enumerate(db.xyxy):
                x1,y1,x2,y2=map(int,xyxy)
                x1=int(x1/scale); y1=int(y1/scale)
                x2=int(x2/scale); y2=int(y2/scale)
                w=x2-x1; h=y2-y1; cx=(x1+x2)//2; cy=(y1+y2)//2
                cls_id=int(db.class_id[i]) if db.class_id is not None else BALL_ID
                conf_v=float(db.confidence[i]) if db.confidence is not None else 0.1
                if id_to_name.get(cls_id,"ball")!="ball": continue
                if h>0 and (h/max(w,1))>3.0: continue
                if w>int(w_frame*0.07) or cy<h_frame*0.15: continue
                det=_build_detection_dict(frame,cx,cy,max(w,20),max(h,20),"ball",conf_v,extract_dorsal=False)
                detecciones.append(det)
        except Exception as e:
            logger.error(f"Error modelo balon: {e}")

    return detecciones


def detect_frame_yolo(frame, confidence=0.45):
    model=_load_player_model()
    h_frame,w_frame=frame.shape[:2]
    results=model(frame,conf=0.10,verbose=False)
    class_names={0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}
    detecciones=[]
    for r in results:
        boxes=r.boxes; masks=r.masks
        if boxes is None: continue
        for i,box in enumerate(boxes):
            cls=int(box.cls[0]); clase=class_names.get(cls,"player")
            box_conf=float(box.conf[0])
            x1,y1,x2,y2=map(int,box.xyxy[0])
            w=x2-x1; h=y2-y1; cx=(x1+x2)//2; cy=(y1+y2)//2
            if clase=="ball":
                if box_conf<0.15 or cy<h_frame*0.30: continue
                if h>0 and (h/max(w,1))>3.0: continue
                if w>int(w_frame*0.06): continue
            elif clase in ("player","goalkeeper","referee"):
                if box_conf<confidence or cy<h_frame*0.22: continue
                if h>0 and (h/max(w,1))>6.5 or w*h<500: continue
            else:
                if box_conf<confidence: continue
            det=_build_detection_dict(frame,cx,cy,w,h,clase,float(box.conf[0]))
            if masks is not None and i<len(masks.xy):
                poly=masks.xy[i]
                if len(poly)>0: det["mask"]=poly.astype(np.int32)
            detecciones.append(det)
    return detecciones


def detect_frame_roboflow(frame, confidence=40, overlap=25):
    import tempfile
    h_img,w_img=frame.shape[:2]
    y_min=int(h_img*0.20); y_max=int(h_img*0.85)
    x_min=int(w_img*0.01); x_max=int(w_img*0.99)
    zona=frame[y_min:y_max,x_min:x_max]
    _,w_zona=zona.shape[:2]
    mitad_izq=zona[:,:w_zona//2]; mitad_der=zona[:,w_zona//2:]
    tmp=tempfile.gettempdir()
    ti=os.path.join(tmp,"_ed_izq.jpg"); td=os.path.join(tmp,"_ed_der.jpg")
    if not cv2.imwrite(ti,mitad_izq) or not cv2.imwrite(td,mitad_der): return []
    try:
        m=_load_roboflow()
        ri=m.predict(ti,confidence=confidence,overlap=overlap)
        rd=m.predict(td,confidence=confidence,overlap=overlap)
    except Exception as e:
        logger.error(f"Roboflow error: {e}"); return []
    dets=[]
    for p in ri.predictions:
        if p["class"] not in VALID_CLASSES: continue
        dets.append(_build_detection_dict(frame,int(p["x"])+x_min,int(p["y"])+y_min,
                    int(p["width"]),int(p["height"]),p["class"],p["confidence"]))
    for p in rd.predictions:
        if p["class"] not in VALID_CLASSES: continue
        dets.append(_build_detection_dict(frame,int(p["x"])+x_min+w_zona//2,int(p["y"])+y_min,
                    int(p["width"]),int(p["height"]),p["class"],p["confidence"]))
    return dets


def detect_frame_coco(frame, confidence=0.35):
    model=_load_coco_model()
    results=model(frame,conf=confidence,classes=[0],verbose=False)
    dets=[]
    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2=map(int,box.xyxy[0])
            w=x2-x1; h=y2-y1; cx=(x1+x2)//2; cy=(y1+y2)//2
            if w*h<800: continue
            dets.append(_build_detection_dict(frame,cx,cy,w,h,"player",float(box.conf[0]),extract_dorsal=False))
    return dets


def detect_frame(frame, mode="auto", confidence=40, imgsz=None):
    """
    Fachada principal. Orden de prioridad:
      kaggle/auto+modelos -> yolo -> roboflow -> coco
    """
    conf_float=confidence/100.0 if isinstance(confidence,int) and confidence>1 else float(confidence)
    conf_int=int(conf_float*100)
    if mode=="kaggle" or (mode=="auto" and kaggle_models_available()):
        return detect_frame_kaggle(frame,confidence=conf_float,imgsz=imgsz)
    if mode=="yolo" or (mode=="auto" and yolo_model_available()):
        return detect_frame_yolo(frame,confidence=conf_float)
    if mode in ("roboflow","auto"):
        result=detect_frame_roboflow(frame,confidence=conf_int)
        if result is not None: return result
        return detect_frame_coco(frame,confidence=conf_float)
    return detect_frame_coco(frame,confidence=conf_float)


def detect_pitch_homography(frame):
    """Homografia automatica con pose_field.pt. Retorna H (3x3) o None."""
    pm=_load_pitch_model()
    if pm is None: return None
    try:
        import supervision as sv
        from sports.configs.soccer import SoccerPitchConfiguration
        CONFIG=SoccerPitchConfiguration()
        result=pm.predict(frame,conf=0.3,verbose=False)[0]
        kp=sv.KeyPoints.from_ultralytics(result)
        mask=kp.confidence[0]>0.5
        frame_pts=kp.xy[0][mask]
        pitch_pts=np.array(CONFIG.vertices)[mask]
        if len(frame_pts)<4: return None
        H,_=cv2.findHomography(pitch_pts.astype(np.float32),frame_pts.astype(np.float32),cv2.RANSAC,5.0)
        return H
    except Exception as e:
        logger.error(f"Homografia automatica error: {e}"); return None


# Clasificacion de equipos

def _extract_torso_rgb(frame, x, y, w, h):
    x1=max(0,int(x-w*0.25)); x2=min(frame.shape[1],int(x+w*0.25))
    y1=max(0,int(y-h*0.3));  y2=min(frame.shape[0],int(y+h*0.1))
    roi=frame[y1:y2,x1:x2]
    if roi.size==0 or roi.shape[0]<2 or roi.shape[1]<2: return None
    roi_rgb=cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
    pixels=roi_rgb.reshape(-1,3)
    if len(pixels)<4: return None
    km=KMeans(n_clusters=2,random_state=42,n_init=3)
    labels=km.fit_predict(pixels)
    counts=Counter(labels)
    dom=counts.most_common(1)[0][0]
    dc=km.cluster_centers_[dom]
    r,g,b=dc
    is_green_rgb=(g>r+30 and g>b+30)
    hsv=cv2.cvtColor(np.uint8([[[int(b),int(g),int(r)]]]),cv2.COLOR_BGR2HSV)[0][0]
    is_green_hsv=(35<=hsv[0]<=85 and hsv[1]>60)
    if is_green_rgb or is_green_hsv:
        if len(counts)>1:
            dc=km.cluster_centers_[counts.most_common(2)[1][0]]
    return tuple(map(int,dc))


class TeamClassifierSigLIP:
    """
    Clasificador SOTA con SigLIP + UMAP + KMeans.
    Robusto con VEO, iluminacion artificial y camisetas de colores similares.
    Requiere: pip install transformers umap-learn
    """
    def __init__(self):
        self.model=None; self.processor=None
        self.reducer=None; self.clustering=None
        self._fitted=False; self.device=None

    def _load(self):
        if self.model is not None: return True
        try:
            import torch
            from transformers import AutoProcessor,SiglipVisionModel
            import umap
            self.device="cuda" if torch.cuda.is_available() else "cpu"
            self.model=SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224").to(self.device)
            self.processor=AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
            self.reducer=umap.UMAP(n_components=3,random_state=42)
            self.clustering=KMeans(n_clusters=2,random_state=42,n_init="auto")
            logger.info(f"SigLIP cargado en {self.device}")
            return True
        except ImportError as e:
            logger.warning(f"SigLIP no disponible: {e}"); return False

    def _embeddings(self,crops):
        import torch
        from PIL import Image
        pils=[Image.fromarray(cv2.cvtColor(c,cv2.COLOR_BGR2RGB)) if isinstance(c,np.ndarray) else c for c in crops]
        embs=[]
        with torch.no_grad():
            for i in range(0,len(pils),32):
                b=pils[i:i+32]
                inp=self.processor(images=b,return_tensors="pt").to(self.device)
                out=self.model(**inp)
                embs.append(torch.mean(out.last_hidden_state,dim=1).cpu().numpy())
        return np.concatenate(embs)

    def fit(self,crops):
        if not self._load() or len(crops)<4: return []
        embs=self._embeddings(crops)
        proj=self.reducer.fit_transform(embs)
        labels=self.clustering.fit_predict(proj)
        self._fitted=True
        return labels.tolist()

    def predict(self,crops):
        if not self._fitted or not crops: return [0]*len(crops)
        embs=self._embeddings(crops)
        proj=self.reducer.transform(embs)
        return self.clustering.predict(proj).tolist()


_siglip_classifier=TeamClassifierSigLIP()


def _get_crop(frame,det):
    x,y,w,h=det["x"],det["y"],det["w"],det["h"]
    x1=max(0,x-w//2); y1=max(0,y-h//2)
    x2=min(frame.shape[1],x+w//2); y2=min(frame.shape[0],y+h//2)
    c=frame[y1:y2,x1:x2]
    return c if c.size>0 else None


def resolve_goalkeeper_team(players_dets, goalkeeper_dets):
    """Asigna portero al equipo geometricamente mas cercano (centroide del equipo)."""
    if not goalkeeper_dets or not players_dets: return [0]*len(goalkeeper_dets)
    t0=np.array([(d["x"],d["y"]) for d in players_dets if d.get("equipo")==0])
    t1=np.array([(d["x"],d["y"]) for d in players_dets if d.get("equipo")==1])
    if not len(t0) or not len(t1): return [0]*len(goalkeeper_dets)
    c0=t0.mean(axis=0); c1=t1.mean(axis=0)
    result=[]
    for gk in goalkeeper_dets:
        p=np.array([gk["x"],gk["y"]])
        result.append(0 if np.linalg.norm(p-c0)<np.linalg.norm(p-c1) else 1)
    return result


def auto_detect_team_colors(frame, detections, mode="kmeans"):
    """
    Detecta y separa los dos equipos.
    mode='siglip': SigLIP embeddings (robusto, requiere transformers+umap)
    mode='kmeans': KMeans RGB (rapido, siempre disponible)
    """
    pdets=[d for d in detections if d.get("clase") in ("player","goalkeeper")]
    if len(pdets)<4: return {}

    if mode=="siglip":
        crops=[c for d in pdets if (c:=_get_crop(frame,d)) is not None]
        if len(crops)>=4:
            labels=_siglip_classifier.fit(crops)
            if labels:
                c0=[]; c1=[]
                for det,lbl in zip(pdets,labels):
                    c=det.get("torso_color") or _extract_torso_rgb(frame,det["x"],det["y"],det["w"],det["h"])
                    if c: (c0 if lbl==0 else c1).append(c)
                t0=tuple(map(int,np.mean(c0,axis=0))) if c0 else (255,0,0)
                t1=tuple(map(int,np.mean(c1,axis=0))) if c1 else (0,0,255)
                return {"team_0":t0,"team_1":t1,"_siglip_fitted":True}

    colors=[det.get("torso_color") or _extract_torso_rgb(frame,det["x"],det["y"],det["w"],det["h"]) for det in pdets]
    colors=[c for c in colors if c is not None]
    if len(colors)<4: return {}
    km=KMeans(n_clusters=2,n_init=10,random_state=42)
    km.fit(np.array(colors,dtype=np.float32))
    centers=km.cluster_centers_
    return {"team_0":tuple(map(int,centers[0])),"team_1":tuple(map(int,centers[1]))}


def classify_team(frame, det, team_colors=None):
    """
    Clasifica: 0 (local), 1 (visitante), 2 (arbitro), -1 (ignorado).
    Usa SigLIP si esta entrenado, KMeans RGB si no.
    """
    if det.get("clase")=="referee": return 2
    if det["w"]*det["h"]<500: return -1
    if team_colors and team_colors.get("_siglip_fitted") and _siglip_classifier._fitted:
        crop=_get_crop(frame,det)
        if crop is not None:
            labels=_siglip_classifier.predict([crop])
            return labels[0] if labels else 0
    color_rgb=det.get("torso_color") or _extract_torso_rgb(frame,det["x"],det["y"],det["w"],det["h"])
    if color_rgb is None: return 0
    r,g,b=color_rgb
    if r<40 and g<40 and b<40: return 2
    if team_colors and "team_0" in team_colors and "team_1" in team_colors:
        c0=np.array(team_colors["team_0"]); c1=np.array(team_colors["team_1"])
        ct=np.array(color_rgb)
        return 0 if np.linalg.norm(ct-c0)<np.linalg.norm(ct-c1) else 1
    return 0
