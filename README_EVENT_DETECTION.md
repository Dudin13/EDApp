# Sistema de Detección Automática de Eventos y Clips

## 🎯 **Resumen del Sistema**

Se ha implementado un sistema completo de **detección automática de eventos + generación de clips + entrenamiento/validación** que extiende el sistema existente.

### ✅ **Componentes Implementados**

#### 1. **AdvancedEventDetector** (`event_spotter_tdeed.py`)
- Extiende `EventSpotterTDEED` con reglas geométricas
- Detecta automáticamente: Goles, Corners, Tiros a puerta
- Zonas del campo predefinidas para clasificación precisa

#### 2. **AutoClipGenerator** (`auto_clip_generator.py`)
- Genera clips automáticamente basados en eventos detectados
- Configuración personalizable por tipo de evento
- Procesamiento en paralelo para eficiencia

#### 3. **EventDetectionTrainer** (`training_validator.py`)
- Sistema de validación y entrenamiento
- Generación de datos sintéticos para testing
- Métricas de accuracy y reportes visuales

#### 4. **Integración en Pipeline**
- `VideoProcessor` actualizado para usar `AdvancedEventDetector`
- Eventos avanzados marcados con `advanced_detection: True`
- Resultados incluyen todos los eventos detectados

#### 5. **UI Mejorada** (`upload_analyze.py`)
- Botones para generar clips automáticos
- Sistema de validación integrado
- Feedback visual del progreso

---

## 🚀 **Cómo Usar el Sistema**

### **1. Análisis de Video**
```bash
# El sistema se activa automáticamente durante el análisis de video
# Los eventos avanzados se detectan junto con los eventos normales
```

### **2. Generación de Clips Automáticos**
Después del análisis, en la UI aparecerán nuevos botones:
- **🎬 Generar Clips Automáticos**: Crea clips para todos los eventos detectados
- **🔍 Validar Sistema**: Ejecuta validación y genera reportes

### **3. Testing Manual**
```python
# Ejecutar tests
python test_event_detection.py
```

---

## 📊 **Eventos Detectados**

### **Eventos Básicos** (existentes)
- Pase, Recepción, Recuperación, Duelo ganado

### **Eventos Avanzados** (nuevos)
- **Gol**: Balón en zona de portería
- **Corner**: Balón sale por línea de fondo
- **Tiro a puerta**: Balón en área de penalty con trayectoria de tiro

---

## 🎬 **Configuración de Clips**

```python
# Configuración por defecto en AutoClipGenerator
clip_configs = {
    "Gol": {"before": 5, "after": 3, "quality": "high"},
    "Corner": {"before": 3, "after": 5, "quality": "medium"},
    "Tiro a puerta": {"before": 2, "after": 4, "quality": "medium"},
    "Falta": {"before": 3, "after": 3, "quality": "low"},
    "Pase clave": {"before": 2, "after": 3, "quality": "low"}
}
```

---

## 📈 **Sistema de Validación**

### **Métricas Trackeadas**
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **Accuracy Temporal**: Error temporal medio en segundos
- **Distribución de Eventos**: Balance de tipos detectados

### **Reportes Generados**
- `training_report.png`: Gráfico visual de accuracy
- `training_report.txt`: Reporte textual detallado

---

## 🔧 **Archivos Modificados**

1. `app/modules/event_spotter_tdeed.py` - Agregado AdvancedEventDetector
2. `app/modules/video_processor.py` - Integración del detector avanzado
3. `app/pages/upload_analyze.py` - Nuevos botones en UI
4. `app/modules/auto_clip_generator.py` - Nuevo módulo
5. `app/modules/training_validator.py` - Nuevo módulo
6. `test_event_detection.py` - Script de testing

---

## 🎯 **Próximos Pasos**

### **Mejoras Inmediatas**
1. **Ajustar thresholds**: Los valores de zona del campo pueden necesitar calibración
2. **Agregar más eventos**: Penaltis, tarjetas, sustituciones
3. **Mejorar calibración**: Hacer las zonas adaptativas al campo detectado

### **Mejoras Avanzadas**
1. **Machine Learning**: Entrenar modelo para detectar eventos complejos
2. **Análisis de trayectoria**: Usar velocidad y aceleración del balón
3. **Contexto temporal**: Analizar secuencias de eventos

---

## 🧪 **Testing**

Para probar el sistema:

1. **Carga un video** en la aplicación
2. **Ejecuta el análisis** completo
3. **Haz clic en "Generar Clips Automáticos"**
4. **Revisa los clips generados** en `output/auto_clips/`
5. **Ejecuta validación** con "Validar Sistema"

Los clips se generan automáticamente con nombres descriptivos como:
- `auto_Gol_001_45.2s.mp4`
- `auto_Corner_002_120.8s.mp4`

---

## 📋 **Solución de Problemas**

### **No se detectan eventos avanzados**
- Verificar que la calibración del campo esté funcionando
- Revisar las coordenadas de zona del campo en `AdvancedEventDetector`

### **Clips no se generan**
- Verificar que FFmpeg esté instalado y accesible
- Comprobar permisos de escritura en `output/auto_clips/`

### **Validación falla**
- Asegurarse de que hay eventos detectados para validar
- Revisar que pandas esté instalado

---

**¡El sistema está listo para usar!** 🎉</content>
<parameter name="filePath">c:\apped\README_EVENT_DETECTION.md