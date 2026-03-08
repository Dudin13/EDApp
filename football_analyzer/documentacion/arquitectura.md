# 🏗️ Arquitectura del Modelo EDudin

El sistema está diseñado como un **Pipeline Multicapa** donde cada módulo procesa la información y la pasa al siguiente.

## 🛰️ 1. Capa de Percepción (Detección)
*   **Motor**: YOLO11-seg (Evolución de YOLOv8).
*   **Misión**: Segmentar jugadores, árbitros, balón y porterías.
*   **Salida**: Coordenadas en píxeles y máscaras de segmentación.

## 📍 2. Capa de Calibración (Geometría)
*   **Motor**: PnLCalibrator (Basado en PnLCalib).
*   **Misión**: Transformar píxeles (TV) a metros reales (Pitch 2D).
*   **Lógica**: Detección automática de dimensiones basada en marcas FIFA.

## 🆔 3. Capa de Identidad (Identity Reader)
*   **Motor**: PARSeq / ResNet.
*   **Misión**: Leer dorsales y clasificar colores de equipo (Camiseta/Pantalón/Medias).
*   **Salida**: ID persistente vinculado al dorsal real.

## 🧠 4. Capa de Inteligencia (Event Spotting)
*   **Motor**: T-DEED (Temporal Deep Event Detection).
*   **Misión**: Detectar Goles, Faltas, Centros y Córners.
*   **Validación**: Reglas geométricas (Ej: Un penalti debe ocurrir en el área).

## 📊 5. Capa de Visualización (Dashboard)
*   **Motor**: Streamlit + Matplotlib (mplsoccer).
*   **Misión**: Presentar mapa táctico, heatmaps y timeline de eventos.

---
> [!TIP]
> Esta arquitectura modular permite actualizar cualquier capa (ej: cambiar YOLO11 por YOLO12) sin romper el resto del sistema.
