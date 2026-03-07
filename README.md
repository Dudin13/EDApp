# ⚽ Football Analyzer (EDApp)

Una potente aplicación de análisis táctico y estadístico de fútbol basada en **Inteligencia Artificial (Computer Vision)** y construida de forma interactiva sobre **Streamlit**.

Diseñada para analistas, entrenadores y aficionados que deseen extraer métricas precisas (mapas de calor, detecciones por minuto, rendimiento por jugador) a partir de un simple archivo de vídeo comercial.

---

## 🚀 Características Principales

### 1. 🔍 Modelos de Visión Artificial Híbridos
Integra detección avanzada de jugadores mediante modelos alojados localmente y en la nube:
- **YOLOv8** para rastreo por defecto súper ligero.
- **Roboflow API** para una precisión líder en la industria al aislar jugadores, balón y árbitro.

### 2. 📋 Subida Flexible de Plantillas (Excel / CSV)
Olvídate de teclear los datos. Carga **documentos Excel (`.xlsx`) o `.csv`** de cada equipo (Local y Visitante).
El sistema parsea automáticamente las columnas (incluso formatos irregulares como 'Dorsal' vs 'Numero') y normaliza las posiciones de los jugadores en la base de datos interna de la app.

### 3. 🎯 Etiquetado Manual Puntero (Point-and-Click)
La IA no siempre es perfecta en el frame inicial, pero tú sí.
- ¡Añade bounding boxes sobre la imagen marcando a golpe de **Clic con el ratón**!
- Elimina cajas IA erróneas.
- **Sistema de Exclusividad de Nombres**: Asigna "Lionel Messi" o al "Árbitro 🟨" a una caja y esa etiqueta desaparecerá del resto de opciones automáticamente para evitar clones de jugadores.

### 4. 🏃‍♂️ Tracker Vectorial
Sistema de *Tracking Simple Evolucionado* que en lugar de usar solapamientos estrictos, emplea matrices de **Distancia Euclídea** sumadas a penalizaciones basadas en clasificación del color de las camisetas para evitar perder la identidad de los jugadores cuando se cruzan a gran velocidad.

### 5. 📊 Dashboard Estadístico Global
Resultados interactivos integrados en Streamlit:
- 🗺️ Mapas de calor de posicionamiento de jugadores.
- 📈 Gráficas de densidad y contacto de balón minuto a minuto.
- 🕒 Resumen final de distancias recorridas y velocidades pico estimadas.

---

## 🛠️ Instalación y Requisitos

**Prerrequisitos**: Python 3.9 o superior.

Clona este repositorio o descarga la carpeta, abre una terminal en la raíz del proyecto e instala las dependencias:

```bash
pip install -r requirements.txt
```

*(Librerías principales: `streamlit`, `opencv-python`, `ultralytics`, `pandas`, `openpyxl`, `streamlit-image-coordinates`, `inference`)*

## 🔑 Configuración

1. Necesitarás una **API Key de Roboflow/Inference** para usar los modelos remotos ultrarrápidos.
2. Crea un archivo `.env` en la raíz (ej. dentro de `football_analyzer/.env`) y añade tu token:

```env
ROBOFLOW_API_KEY=tu_roboflow_key_aqui
```

## 🎮 Cómo usar

Arranca el servidor de Streamlit y ábrelo en tu navegador. 
Ve al directorio principal (`football_analyzer`) y ejecuta:

```bash
python -m streamlit run app.py
```

### Flujo de Trabajo

1. **Clip:** Introduce la URL a un vídeo físico o súbelo directamente en la pestaña "Subir y Analizar".
2. **Plantillas:** (Opcional) Carga tu *plantilla_madrid.xlsx* u otro Excel.
3. **Análisis Frame 1:** Corrige a la IA si no vio el balón o al linier antes de empezar. Un clic = Una caja roja lista para tu magia.
4. **Ver:** Pulsa comenzar y tómate un café mientras la IA procesa todo el clip y genera las métricas en pantalla.

---

## 📝 Autor y Licencia
Proyecto desarrollado de forma iterativa implementando UX/UI moderno para facilitar el análisis futbolístico democratizado. 
Módulos principales ubicados en `football_analyzer/modules`.
