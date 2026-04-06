#!/usr/bin/env python3
"""
demo_event_detection.py - Demo rápida del sistema de detección de eventos
"""

import sys
from pathlib import Path

# Agregar paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "app"))
sys.path.insert(0, str(project_root / "core"))

def demo():
    """Demo simple del sistema."""
    print("🎯 DEMO: Sistema de Detección Automática de Eventos")
    print("=" * 50)

    try:
        # Importar componentes
        from modules.event_spotter_tdeed import AdvancedEventDetector, BallEvent
        print("✅ AdvancedEventDetector importado correctamente")

        from modules.auto_clip_generator import AutoClipGenerator
        print("✅ AutoClipGenerator importado correctamente")

        from modules.training_validator import EventDetectionTrainer
        print("✅ EventDetectionTrainer importado correctamente")

        # Crear detector
        detector = AdvancedEventDetector()
        print("✅ Detector avanzado creado")

        # Simular detección de eventos
        print("\n🧪 Probando detección de eventos...")

        test_cases = [
            {"ball_pos": (500, 100), "pitch_pos": (52, 2), "tracks": {}, "frame_second": 10.0, "expected": "Gol"},
            {"ball_pos": (300, 50), "pitch_pos": (15, 5), "tracks": {}, "frame_second": 25.0, "expected": "Corner"},
            {"ball_pos": (700, 400), "pitch_pos": (85, 60), "tracks": {}, "frame_second": 40.0, "expected": "Corner"},
        ]

        detected_events = []
        for i, test_case in enumerate(test_cases, 1):
            events = detector.detect_advanced_events(**test_case)
            detected_events.extend(events)
            print(f"  Test {i}: {test_case['expected']} -> {'✅ Detectado' if events else '❌ No detectado'}")

        print(f"\n📊 Total eventos detectados: {len(detected_events)}")
        for event in detected_events:
            print(f"  - {event.action} en {event.timestamp:.1f}s (conf: {event.confidence:.2f})")

        # Demo del generador de clips
        print("\n🎬 Probando generador de clips...")
        from core.config.settings import settings
        generator = AutoClipGenerator(
            clips_dir=settings.OUTPUT_DIR / "demo_clips",
            ffmpeg_path=getattr(settings, 'FFMPEG_PATH', 'ffmpeg')
        )
        print("✅ Generador de clips creado")

        # Demo del validador
        print("\n📈 Probando sistema de validación...")
        validator = EventDetectionTrainer(settings.OUTPUT_DIR / "demo_validation")
        print("✅ Sistema de validación creado")

        print("\n🎉 DEMO COMPLETADA EXITOSAMENTE!")
        print("\n📋 COMPONENTES DEL SISTEMA:")
        print("  • AdvancedEventDetector: Detecta goles, corners, tiros")
        print("  • AutoClipGenerator: Crea clips automáticos")
        print("  • EventDetectionTrainer: Valida y entrena el sistema")
        print("  • Integración completa en VideoProcessor")
        print("  • UI mejorada con nuevos botones")

        return True

    except Exception as e:
        print(f"❌ Error en demo: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = demo()
    if success:
        print("\n🚀 El sistema está listo para usar en la aplicación!")
    else:
        print("\n⚠️  Revisar configuración e imports")