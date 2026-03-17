import cv2
import sys
import os
from pathlib import Path

# Fix PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.config.settings import settings
from core.logger import logger
from ai.detector.detector import DetectorEngine

def verify_foundation():
    logger.info("Iniciando verificación de cimientos...")
    
    # 1. Verificar settings
    logger.info(f"BASE_DIR: {settings.BASE_DIR}")
    logger.info(f"WEIGHTS_DIR: {settings.WEIGHTS_DIR}")
    
    # 2. Verificar Detector
    try:
        detector = DetectorEngine()
        logger.info("DetectorEngine instanciado correctamente.")
    except Exception as e:
        logger.error(f"Error instanciando DetectorEngine: {e}")
        return False
        
    logger.info("Verificación de cimientos completada con éxito.")
    return True

if __name__ == "__main__":
    if verify_foundation():
        sys.exit(0)
    else:
        sys.exit(1)
