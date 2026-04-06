#!/usr/bin/env python3
"""
test_event_detection.py - Testing suite para el sistema de detección de eventos
"""

import sys
import numpy as np
from pathlib import Path

# Agregar paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "app"))
sys.path.insert(0, str(project_root / "core"))

from modules.event_spotter_tdeed import AdvancedEventDetector
from modules.training_validator import EventDetectionTrainer
from core.config.settings import settings

def test_event_detection():
    """Test básico del detector de eventos."""
    print("🧪 Testing Event Detection System...")

    detector = AdvancedEventDetector()

    # Simular algunos frames con eventos
    test_frames = [
        {"ball_pos": (500, 100), "pitch_pos": (52, 2), "tracks": {}, "frame_second": 10.0},   # Gol equipo A
        {"ball_pos": (300, 50), "pitch_pos": (15, 5), "tracks": {}, "frame_second": 25.0},   # Corner
        {"ball_pos": (700, 400), "pitch_pos": (85, 60), "tracks": {}, "frame_second": 40.0}, # Corner opuesto
    ]

    detected_events = []
    for frame_data in test_frames:
        events = detector.detect_advanced_events(**frame_data)
        detected_events.extend(events)

    print(f"✅ Detectados {len(detected_events)} eventos:")
    for event in detected_events:
        print(f"  - {event.action} en {event.timestamp:.1f}s (conf: {event.confidence:.2f})")

    return detected_events

def test_validation_system():
    """Test del sistema de validación."""
    print("\n🧪 Testing Validation System...")

    validator = EventDetectionTrainer(settings.OUTPUT_DIR / "test_validation")

    # Crear datos sintéticos
    synthetic = validator.create_synthetic_dataset(
        str(settings.BASE_DIR / "data" / "samples" / "sample_video.mp4")
    )

    print(f"✅ Generados {len(synthetic)} eventos sintéticos")
    print(f"Distribución: {synthetic['event_type'].value_counts().to_dict()}")

    # Simular predicciones
    mock_predictions = synthetic.copy()
    mock_predictions["confidence"] = np.random.uniform(0.5, 1.0, len(mock_predictions))

    # Validar
    results = validator.validate_predictions(mock_predictions, synthetic)

    print(f"✅ Accuracy: {results['accuracy']:.2%}")

    return results

if __name__ == "__main__":
    print("🚀 Iniciando tests del sistema de detección de eventos...\n")

    try:
        events = test_event_detection()
        validation = test_validation_system()

        print("
🎉 Todos los tests pasaron exitosamente!"        print(f"📊 Eventos detectados: {len(events)}")
        print(f"📈 Accuracy de validación: {validation['accuracy']:.2%}")

    except Exception as e:
        print(f"❌ Error en tests: {e}")
        import traceback
        traceback.print_exc()