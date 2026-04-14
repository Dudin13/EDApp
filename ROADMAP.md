# EDApp · Roadmap Unificado 2026

Este documento unifica la visión estratégica de negocio con los objetivos técnicos del repositorio.

## 📊 Estado Actual
- **Progreso Global:** ~18%
- **Fases Completadas:** 2
- **Fases Activas:** 2
- **Fases Pendientes:** 5

---

## 🏗️ FASE 0: Base del Sistema
**Estado:** ✅ Completada
- [x] Refactorización principal (app/ml/core/data).
- [x] Entorno CUDA 12.1 funcional.
- [x] Persistencia de eventos y modularización.
- [x] Repositorio limpio y documentado.

---

## 🎯 FASE 1: Detección Robusta (VEO)
**Estado:** 🚀 Activa
- [x] Modelo `players_v3` (mAP50=0.818).
- [/] **Bug Crítico:** Corrección del Tracker (Mismatch de IDs en `detector.py`).
- [ ] Entrenamiento Incremental: Procesar 590 imágenes revisadas manualmente.
- [ ] Mejora del Balón: Entrenamiento a 1280px para cámaras panorámicas.

---

## 📐 FASE 2: Homografía Estable
**Estado:** 🔧 Activa
- [x] Calibración automática de campo operativa.
- [ ] **Sync Proyectado:** Implementar `sskit` (SynLoc) para mayor precisión.
- [ ] Ajustar `world_to_image()` para proyección milimétrica en minimapa.

---

## 👕 FASE 3: TeamClassifier
**Estado:** ⏳ Pendiente de Integración
- [x] Script de clasificación por color desarrollado.
- [ ] Integración lógica en `video_processor.py`.
- [ ] HUD dinámico con colores de equipo reales.

---

## 📡 FASE 4: Radar Táctico 2D
**Estado:** 📅 Pendiente
- [ ] Proyección de tracks en tiempo real usando `cv2.warpPerspective`.
- [ ] Interfaz de minimapa interactiva en Streamlit.

---

## ⚡ FASE 5: Eventos y Métricas (Paralelo)
**5A: Event Detection**
- [ ] Integración de `EventSpotterTDEED`.
- [ ] Detección automática de tiros, centros y posesión.

**5B: Advanced Metrics (Quantic)**
- [ ] Cálculo de compactación de líneas.
- [ ] Análisis de línea de presión y ancho de bloque.

---

## 🖥️ FASE 6: Dashboard y Event Data
**Estado:** 📅 Pendiente
- [ ] Interfaz "Analyst Dashboard" (estilo Football Manager).
- [ ] Generación automática de clips destacados.
- [ ] OCR de dorsales con `PARSeq`.
- [ ] Exportación de reportes PDF.

---

## 🔮 Fases Futuras
- **F7: Pose Estimation:** Análisis biomecánico con esqueleto de 15 keypoints.
- **F8: Game State Reconstruction:** Tracking cross-cámara completo y ReID nivel VAR.

---

*Nota: Para ver el roadmap visual interactivo, ejecuta `Roadmap.bat` en la raíz del proyecto.*
