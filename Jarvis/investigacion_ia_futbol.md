# 🧠 Investigación IA y Táctica (Para la próxima sesión)

¡Hola equipo! Aquí recojo los hallazgos de hoy sobre repositorios, librerías y algoritmos Open Source específicos para fútbol que nos van a ayudar a llegar a ese **100% de éxito en el auto-etiquetado y análisis táctico**.

---

## 🏗️ 1. Librerías Core de Python para Datos de Fútbol
Estas librerías nos van a ahorrar reinventar la rueda para dibujar campos, mapas de calor y redes de pases:

*   **Kloppy (`kloppy`)**:
    *   **¿Para qué sirve?**: Es el estándar de oro (Open Source, apoyado por PySport) para leer y estandarizar datos de tracking (coordenadas XY de jugadores) y eventos (pases, tiros).
    *   **Utilidad para ED Analytics**: Si mañana calibramos la cámara y sacamos las coordenadas X, Y de nuestro YOLOv8, podemos meterlas en `Kloppy` y nos genera DataFrames de Pandas listos para calcular métricas tácticas.
*   **Floodlight (`floodlight`)**:
    *   **¿Para qué sirve?**: Un framework para analizar datos de deportes. Carga datos de tracking y calcula modelos computacionales complejos como **Control de Espacio (Space Control - Voronoi)** y **Potencia Metabólica (Metabolic Power)**.
    *   **Utilidad**: Una vez tengamos el tracking, Floodlight dibuja automáticamente las áreas dominadas por cada equipo en el campo.

---

## 🎯 2. Visión por Computadora: Tracking y Homografía
Hemos revisado varios proyectos en GitHub que han resuelto el problema de grabar desde la grada y convertirlo en un mini-mapa táctico:

*   **Proyecto "FootyTacticalAnalysis" / "Football-Analysis"**:
    *   Están usando el mismo stack que nosotros: **YOLOv8** para jugadores/balón/árbitro.
    *   *El truco*: Usan **KMeans Clustering** sobre un recorte (crop) de cada jugador detectado para obtener el color de su camiseta y separarlos en Equipo A y Equipo B automáticamente. No hace falta que el modelo "aprenda" los escudos, solo escanea el color predominante.
*   **Sistema "SocerNet Camera Calibration"**:
    *   Tienen código robusto para detectar las líneas del campo (área, centro del campo, bandas).
    *   Usa `cv2.getPerspectiveTransform()` para calcular la matriz de homografía. Esto significa que podemos pasar del formato "vídeo deformado por la perspectiva" al formato "pizarra táctica 2D cenital" en tiempo real.

---

## ⚽ 3. Estrategia de Perfeccionamiento (100% Acierto) para Mañana

Basado en la investigación, este es el plan para subir del 85% al 100%:

1.  **Refinar Detección y Equipo (KMeans)**: 
    *   Mejorar nuestro `detector.py` para que cuando encuentre la caja (bounding box) de un jugador, extraiga los píxeles del centro de su pecho, aplique KMeans y lo asigne instantáneamente al Cádiz vs Rival por color.
2.  **Tracking Invencible (ByteTrack vs DeepSORT)**: 
    *   Ahora mismo perdemos al jugador si se cruza con otro. Vamos a conectar YOLOv8 con la librería **ByteTrack**, que es rapidísima y maneja las oclusiones grupales mucho mejor (usa los boxes con baja confidencialidad en vez de descartarlos como hace SORT original).
3.  **Movimiento de Cámara (Optical Flow)**:
    *   Como la cámara graba siguiendo el balón, los jugadores parecen moverse aunque estén parados. Implementaremos **Lucas-Kanade Optical Flow** con OpenCV para que el código sepa hacia dónde se mueve el campo, restando ese movimiento a los jugadores para obtener su velocidad real y no perder el tracking.
4.  **Tracking del Balón Específico (Interpolación)**:
    *   El balón a veces desaparece por un rebote rápido. Añadiremos un filtro matemático simple (como un *Filtro de Kalman* 1D o interpolación pandas) para que si el modelo pierde el balón 3 fotogramas por ir muy rápido, rellene el hueco sabiendo la trayectoria que llevaba.

---
Con esta munición técnica guardada en nuestro `Jarvis`, mañana estructuraremos el código para ir atacando estos puntos tácticos reales. ¡Vamos por el buen camino!
