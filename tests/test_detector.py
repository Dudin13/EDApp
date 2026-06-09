import pytest
import cv2
import numpy as np
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_APP_ROOT = str(Path(_REPO_ROOT) / "app")
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

from modules.detector import _load_player_model, detect_frame

def test_detector_classes():
    # Carga players.pt
    model = _load_player_model()
    assert model is not None, "Error al cargar el modelo players.pt"
    
    # Verificar que las clases base existan
    classes_values = list(model.names.values())
    assert "player" in classes_values, "Falta la clase 'player' en el modelo"
    
    # Detecta sobre 1 frame conocido
    image_path = Path(_REPO_ROOT) / "data" / "samples" / "test_detection.jpg"
    if image_path.exists():
        frame = cv2.imread(str(image_path))
    else:
        # Fallback a un frame negro
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
    dets = detect_frame(frame, mode="auto")
    
    # Verifica que devuelve las 4 clases correctas como conjunto de validación
    valid_classes = {"player", "goalkeeper", "referee", "ball"}
    for d in dets:
        assert d["clase"] in valid_classes, f"Clase desconocida: {d['clase']}"
