# Roadmap de Mejoras: Football Analyzer V2

Propuestas para llevar la aplicación al siguiente nivel de profesionalismo y utilidad táctica.

## 🎨 Interfaz y Experiencia (UI/UX)
- **Glassmorphism Real**: Implementar efectos de desenfoque de fondo (`backdrop-filter`) en paneles para un look más "Apple/Premium".
- **Modo Enfoque**: Opción para ocultar todos los menús y dejar solo el vídeo y la botonera con un diseño minimalista.
- **Micro-animaciones**: Transiciones suaves entre los pasos del asistente y efectos de "pulsación" en los botones de carga.
- **Dashboard de Bienvenida**: Una pantalla de inicio con estadísticas rápidas de los últimos análisis realizados.

## ✂️ Modo Analista (Scout) V2
- **Hotkeys de Teclado (Analista Pro)**: Asignar teclas (A, S, D, F...) a los botones de etiquetas para análisis a velocidad real sin usar el ratón.
- **Comparador Cinematográfico**: Pantalla partida (Side-by-Side) para comparar dos clips de vídeo o diferentes ángulos de la misma jugada.
- **Seguimiento Dinámico (Auto-Zoom)**: El vídeo hace zoom automático a la zona de acción o al jugador etiquetado durante la reproducción del clip.
- **Dibujo sobre Vídeo en Vivo**: Lápiz táctico que funcione directamente sobre el reproductor de vídeo principal, no solo en los recortes.
- **Informes Tácticos Automáticos**: Generación de un PDF profesional que incluya los clips, las anotaciones dibujadas y estadísticas del partido para entregar al staff técnico.

## 🧠 Inteligencia Artificial y Datos Avanzados
- **Re-Identificación (ReID)**: Mantener el dorsal o ID del jugador aunque salga de plano o se cruce con otros.
- **Calibración de Campo (Homonografía)**: Mapear el vídeo a un plano 2D (Mini-mapa) para calcular distancias reales y velocidades de sprint.
- **Co-Pilot de Análisis**: La IA sugiere etiquetas automáticas (ej: "Detección de Pase") basadas en el movimiento del balón y jugadores.
- **Entrenamiento Continuo (Active Learning)**: Un botón para "corregir" a la IA y usar esas correcciones para mejorar el modelo automáticamente en el futuro.

## 🛠️ Estructura Técnica
- **Refactorización**: Separar la lógica pesada de `scout.py` en módulos más pequeños para mayor velocidad.
- **Optimización de Memoria**: Mejorar la carga de vídeos largos para que el PC no sufra tanto.
