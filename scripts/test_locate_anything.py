import sys
from pathlib import Path
import os
import shutil

# Configurar path para importar modulos de la raiz
root_path = Path(__file__).parent.parent.absolute()
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

def main():
    print("========================================")
    print("  Prueba de modelo LocateAnything")
    print("========================================")
    
    # 1. Comprobar espacio en disco (se requieren ~6GB)
    free_gb = shutil.disk_usage(str(root_path)).free / (1024**3)
    print(f"Espacio libre en disco: {free_gb:.1f} GB")
    
    if free_gb < 10.0:
        print("[WARNING] Espacio en disco bajo. Se requieren al menos 6GB para el modelo.")
    else:
        print("[OK] Espacio suficiente para descargar el modelo de LocateAnything.")

    print("\nScript inicializado y listo para integrar el código de inferencia.")
    # TODO: Añadir código de descarga e inferencia del modelo LocateAnything aquí.

if __name__ == "__main__":
    main()
