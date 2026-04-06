# 🔍 Sistema de Corrección de Detecciones de Jugadores

## 🎯 **¿Qué es esto?**

Una herramienta integrada en tu aplicación que te permite **comparar las predicciones automáticas de la IA con correcciones manuales**, para mejorar la calidad del dataset de entrenamiento.

## 📋 **¿Por qué lo necesitas?**

Si el modelo YOLO no detecta correctamente a los jugadores, esta herramienta te permite:

- ✅ **Comparar** predicciones automáticas vs correcciones manuales
- ✅ **Corregir** detecciones erróneas dibujando rectángulos
- ✅ **Mejorar** el dataset para futuros entrenamientos
- ✅ **Validar** la calidad del modelo actual

---

## 🚀 **Cómo acceder**

1. **Abre tu aplicación** de EDudin
2. **En el sidebar lateral**, busca la sección **"Herramientas IA"**
3. **Haz clic en** **"🔍 Corrección de Imágenes"**

---

## 🎨 **Interfaz de la herramienta**

### **Layout Principal**
```
┌─────────────────────────────────────────────────┐
│ 🔍 Corrección de Detecciones de Jugadores       │
│ Compara predicciones automáticas vs manuales   │
└─────────────────────────────────────────────────┘

┌─────────────────┬─────────────────┐
│ 🤖 IA Automática │ ✏️ Manual       │
│                 │                 │
│ [Imagen con     │ [Canvas para    │
│  detecciones]   │  dibujar]       │
│                 │                 │
│ 📊 Estadísticas │ 🔧 Controles    │
└─────────────────┴─────────────────┘

🎯 Decisión Final: [Radio buttons] 💾 Guardar
```

### **Elementos de la interfaz**

#### **1. Configuración del Dataset**
- **Selector de dataset**: Elige qué imágenes revisar
- **Navegación**: Anterior/Siguiente imagen
- **Número de imagen**: Input para saltar a imagen específica

#### **2. Columna Izquierda - IA Automática**
- **Imagen con predicciones**: Muestra lo que detectó el modelo YOLO
- **Control de confianza**: Slider para ajustar el threshold de detección
- **Estadísticas**: Número de detecciones encontradas
- **Lista de detecciones**: Detalle de cada objeto detectado

#### **3. Columna Derecha - Corrección Manual**
- **Canvas interactivo**: Dibuja rectángulos sobre la imagen
- **Herramientas de dibujo**: Solo modo rectángulo activado
- **Contador de rectángulos**: Muestra cuántos dibujaste
- **Selector de clase**: Asigna clase a los rectángulos dibujados

#### **4. Decisión Final**
- **Radio buttons** para elegir qué versión usar:
  - Usar predicción automática (IA)
  - Usar corrección manual
  - Mantener etiquetas existentes

---

## 📖 **Cómo usar paso a paso**

### **Paso 1: Seleccionar Dataset**
```python
# Elige el dataset que quieres corregir:
- Dataset de Muestras
- Imágenes de Entrenamiento
- Validación Híbrida
- Super Focused 50
```

### **Paso 2: Revisar Detección Automática**
1. **Mira la columna izquierda**: ¿El modelo detectó correctamente?
2. **Ajusta la confianza** si es necesario (slider 0.1 - 1.0)
3. **Revisa la lista** de detecciones encontradas

### **Paso 3: Corregir Manualmente**
1. **Ve a la columna derecha**
2. **Dibuja rectángulos** sobre jugadores no detectados
3. **Selecciona la clase** apropiada (Team A, Team B, etc.)
4. **Asegúrate** de que los rectángulos cubran bien a los jugadores

### **Paso 4: Tomar Decisión**
1. **Elige la mejor opción**:
   - ✅ **IA correcta** → Usa predicciones automáticas
   - ✏️ **Necesita corrección** → Usa tus dibujos manuales
   - 📁 **Mantener existentes** → Conserva etiquetas previas

### **Paso 5: Guardar**
1. **Haz clic en "💾 Guardar Decisión"**
2. **Las etiquetas se guardan** automáticamente en formato YOLO
3. **Pasa a la siguiente imagen**

---

## 🎯 **Flujo de Trabajo Recomendado**

```
1. Inicia con confianza baja (0.1-0.2) para ver TODO
   ↓
2. Revisa si la IA detectó correctamente
   ↓
3. Si SÍ → Marca "Usar predicción automática"
   ↓
4. Si NO → Dibuja correcciones manuales
   ↓
5. Elige clase apropiada para tus dibujos
   ↓
6. Marca "Usar corrección manual"
   ↓
7. Guarda y pasa a siguiente imagen
```

---

## 📊 **Formatos de Etiquetas**

Las correcciones se guardan en **formato YOLO**:

```
# Formato: clase x_centro y_centro ancho alto
0 0.512500 0.408333 0.075000 0.141667  # Team A
1 0.487500 0.591667 0.062500 0.116667  # Team B
2 0.525000 0.500000 0.050000 0.100000  # Goalkeeper
```

**Coordenadas normalizadas** (0-1) para que funcionen con cualquier resolución.

---

## 🔧 **Solución de Problemas**

### **"No se pudo cargar ningún modelo YOLO"**
- Verifica que tienes modelos en `assets/weights/`
- O que `yolo11m.pt` existe en la raíz

### **"No se encontraron imágenes"**
- Revisa que los directorios existen
- Verifica que hay archivos `.jpg`, `.png`, etc.

### **Canvas no funciona**
- Asegúrate de que `streamlit-drawable-canvas` esté instalado
- Prueba recargar la página

### **No se guardan las etiquetas**
- Verifica permisos de escritura
- Revisa que el directorio `labels/` existe

---

## 📈 **Mejora Continua del Modelo**

Esta herramienta es parte de un **ciclo de mejora**:

```
Detecciones IA → Corrección Manual → Nuevo Entrenamiento → Mejor Modelo
     ↑                                                              ↓
     └─────────────────── Dataset Mejorado ─────────────────────────┘
```

**Cada corrección que haces mejora el dataset para futuros entrenamientos.**

---

## 🎉 **¡Listo para usar!**

Ahora tienes una herramienta profesional para **corregir y mejorar las detecciones de tu modelo YOLO**. Esta funcionalidad integrada te permitirá tener un dataset de entrenamiento de alta calidad.

**¡Mejora tu modelo paso a paso! 🚀**</content>
<parameter name="filePath">c:\apped\README_CORRECCION_IMAGENES.md