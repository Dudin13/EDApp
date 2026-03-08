# 📖 Guía de Entrenamiento: Modelo EDudin

Esta guía detalla el proceso para re-entrenar el cerebro de la IA y mejorar su precisión día tras día.

## 📂 Estructura de Datos
Los datos deben residir en `c:/apped/04_Datasets_Entrenamiento/`.
*   `/images`: Frames de partidos en alta resolución (1024px).
*   `/labels`: Archivos `.txt` en formato YOLO (nc: 8).

## 🛠️ Cómo Lanzar el Entrenamiento
Ejecuta el script maestro:
```bash
python train_yolo/train.py
```

### Parámetros Críticos
*   **imgsz=1024**: No bajar de aquí para mantener la calidad de los dorsales.
*   **epochs=200**: Necesario para que la red converja en situaciones de oclusión.
*   **AdamW**: Recomendado para evitar el olvido catastrófico en modelos grandes.

## 🧹 Limpieza de Datos
Antes de entrenar, usa las utilidades en `train_yolo/utility_scripts/`:
*   `fix_labels.py`: Corrige etiquetas mal formadas.
*   `reorganize_dataset.py`: Asegura la proporción 80/20 entre train y val.

---
> [!IMPORTANT]
> Nunca borres el archivo `EDudin_final.pt` original sin tener un backup. Es la base de todo nuestro conocimiento.
