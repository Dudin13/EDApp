import os
from pathlib import Path

def clean_yolo_labels(labels_dir):
    """
    Recorre todos los archivos .txt de una carpeta de labels YOLOv8.
    Si una línea tiene 5 valores (clase x y w h), es una caja (bounding box) -> SE ELIMINA.
    Si tiene más de 5 valores (clase x1 y1 x2 y2 ...), es un polígono -> SE MANTIENE.
    """
    labels_path = Path(labels_dir)
    if not labels_path.exists():
        print(f"La ruta {labels_path} no existe.")
        return 0, 0

    archivos_modificados = 0
    lineas_eliminadas = 0

    for txt_file in labels_path.glob("*.txt"):
        with open(txt_file, 'r') as f:
            lineas = f.readlines()
        
        lineas_limpias = []
        modificado = False
        
        for linea in lineas:
            partes = linea.strip().split()
            if len(partes) == 5:
                # Es una caja, la ignoramos (eliminamos)
                lineas_eliminadas += 1
                modificado = True
            elif len(partes) > 5:
                # Es un polígono, lo guardamos
                lineas_limpias.append(linea)

        if modificado:
            archivos_modificados += 1
            with open(txt_file, 'w') as f:
                f.writelines(lineas_limpias)

    return archivos_modificados, lineas_eliminadas

if __name__ == "__main__":
    print("="*60)
    print("LIMPIANDO DATASET MIXTO (Cajas + Polígonos)")
    print("="*60)
    
    # Rutas del dataset de Roboflow
    train_dir = r"C:\apped\roboflow_dataset\train\labels"
    valid_dir = r"C:\apped\roboflow_dataset\valid\labels"
    
    print("Procesando carpeta Train...")
    archivos_train, lineas_train = clean_yolo_labels(train_dir)
    print(f" -> {lineas_train} cajas eliminadas en {archivos_train} archivos modificados.")
    
    print("\nProcesando carpeta Valid...")
    archivos_valid, lineas_valid = clean_yolo_labels(valid_dir)
    print(f" -> {lineas_valid} cajas eliminadas en {archivos_valid} archivos modificados.")
    
    print("\n¡Limpieza completada! El dataset ahora debería tener solo polígonos.")
