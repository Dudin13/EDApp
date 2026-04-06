#!/usr/bin/env python3
"""
test_image_correction.py - Prueba rápida de la funcionalidad de corrección de imágenes
"""

import sys
from pathlib import Path

# Agregar paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "app"))
sys.path.insert(0, str(project_root / "core"))

def test_imports():
    """Verificar que todos los imports funcionan."""
    print("🧪 Probando imports de la página de corrección...")

    try:
        from pages.image_correction import render
        print("✅ Página image_correction importada correctamente")

        from core.config.settings import settings
        print("✅ Configuración cargada correctamente")

        # Verificar que streamlit_drawable_canvas está disponible
        import streamlit_drawable_canvas
        print("✅ streamlit-drawable-canvas disponible")

        # Verificar que ultralytics está disponible
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO disponible")

        return True

    except Exception as e:
        print(f"❌ Error en imports: {e}")
        return False

def test_directories():
    """Verificar que los directorios existen."""
    print("\n🧪 Verificando directorios de datasets...")

    from core.config.settings import settings

    dataset_options = {
        "Dataset de Muestras": settings.BASE_DIR / "data" / "samples",
        "Imágenes de Entrenamiento": settings.BASE_DIR / "data" / "datasets" / "imagenes_entrenamiento",
        "Validación Híbrida": settings.BASE_DIR / "data" / "datasets" / "hybrid_dataset" / "valid" / "images",
        "Super Focused 50": settings.BASE_DIR / "data" / "datasets" / "super_focused_50" / "train" / "images"
    }

    available_datasets = []
    for name, path in dataset_options.items():
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {name}: {path}")

        if exists:
            # Contar imágenes
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            total_images = 0
            for ext in image_extensions:
                total_images += len(list(path.glob(ext)))
            print(f"      📸 {total_images} imágenes encontradas")
            if total_images > 0:
                available_datasets.append(name)

    if available_datasets:
        print(f"\n✅ Datasets disponibles: {', '.join(available_datasets)}")
        return True
    else:
        print("\n⚠️ No se encontraron imágenes en ningún dataset")
        return False

def test_models():
    """Verificar que los modelos están disponibles."""
    print("\n🧪 Verificando modelos YOLO...")

    from core.config.settings import settings

    model_paths = [
        settings.PLAYER_MODEL_PATH,
        settings.BASE_DIR / "assets" / "weights" / "detect_players_v3.pt",
        settings.BASE_DIR / "yolo11m.pt"
    ]

    available_models = []
    for model_path in model_paths:
        if model_path and model_path.exists():
            print(f"✅ Modelo encontrado: {model_path}")
            available_models.append(str(model_path))
        else:
            print(f"❌ Modelo no encontrado: {model_path}")

    if available_models:
        print(f"\n✅ Modelos disponibles: {len(available_models)}")
        return True
    else:
        print("\n❌ No se encontraron modelos YOLO")
        return False

def main():
    """Función principal de testing."""
    print("🚀 TEST: Sistema de Corrección de Imágenes")
    print("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("Directorios", test_directories),
        ("Modelos", test_models)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Ejecutando test: {test_name}")
        result = test_func()
        results.append(result)

    print("\n" + "=" * 50)
    print("📊 RESULTADOS FINALES:")

    passed = sum(results)
    total = len(results)

    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASÓ" if results[i] else "❌ FALLÓ"
        print(f"  {test_name}: {status}")

    print(f"\n🎯 Tests pasados: {passed}/{total}")

    if passed == total:
        print("\n🎉 ¡Todos los tests pasaron! El sistema está listo.")
        print("\n💡 Para usar la herramienta:")
        print("   1. Ejecuta: streamlit run app/app.py")
        print("   2. Ve al sidebar → Herramientas IA → Corrección de Imágenes")
        print("   3. Selecciona un dataset y comienza a corregir")
    else:
        print("\n⚠️ Algunos tests fallaron. Revisa la configuración.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)