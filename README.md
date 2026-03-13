# EDApp — Football Video Analysis Platform

> **Plataforma de análisis de vídeo de fútbol** desarrollada para el Departamento de Datos de Cádiz CF Cantera.  
> **Football video analysis platform** developed for the Cádiz CF Cantera Data Department.

---

## 🇪🇸 Español

### ¿Qué es EDApp?

EDApp es una aplicación de análisis de vídeo de fútbol construida con Python y Streamlit. Permite detectar jugadores, clasificarlos por equipo, hacer seguimiento de su posición en el campo y extraer métricas de rendimiento a partir de vídeos de partidos.

### Arquitectura del pipeline

El sistema funciona en capas independientes:

| Capa | Módulo | Tecnología | Estado |
|------|--------|------------|--------|
| Detección | `app/modules/detector.py` | YOLO11m-seg | ⚠️ Entrenando |
| Clasificación de equipos | `app/modules/team_classifier.py` | KMeans + HSV | ✅ Listo |
| Tracking | `app/modules/tracker.py` | ByteTrack | ✅ Funcional |
| Calibración de campo | `app/modules/calibration_pnl.py` | PnLCalib + Homografía | ⚠️ Manual |
| Identificación de dorsales | `app/modules/identity_reader.py` | OCR (PARSeq) | ❌ Pendiente |
| Detección de eventos | `app/modules/event_spotter_tdeed.py` | T-DEED | ❌ Pendiente |

### Clases del modelo de detección

El modelo YOLO detecta 4 clases:

```
0: player      — jugador de campo
1: goalkeeper  — portero
2: referee     — árbitro
3: ball        — balón
```

La diferenciación de equipos (equipo A vs equipo B) se realiza en la **Capa 2** mediante clasificación por color de camiseta, sin necesidad de reentrenar el modelo.

### Estructura del proyecto

```
C:\apped\
├── app/                        # Interfaz Streamlit
│   ├── app.py                  # Entrada principal
│   ├── modules/                # Módulos del pipeline
│   │   ├── detector.py         # Detección YOLO
│   │   ├── team_classifier.py  # Clasificación de equipos por color
│   │   ├── tracker.py          # Tracking de jugadores
│   │   ├── video_processor.py  # Pipeline completo
│   │   └── ...
│   └── pages/                  # Páginas de la app
│       ├── upload_analyze.py   # Subir y analizar vídeo
│       ├── player_identification.py  # Identificar jugadores
│       ├── match_metrics.py    # Métricas del partido
│       ├── tactical_map.py     # Mapa táctico
│       └── ...
├── ml/
│   ├── training/               # Scripts de entrenamiento
│   │   ├── train_unified.py    # Entrenamiento unificado (GPU)
│   │   ├── create_val_split.py # Crear split de validación
│   │   ├── diagnostico.py      # Diagnóstico del dataset
│   │   ├── remap_classes.py    # Remapeo de clases (6→4)
│   │   ├── add_ball_dataset.py # Añadir dataset de balones
│   │   └── data.yaml           # Configuración del dataset
│   └── labeller/               # Herramienta de etiquetado
│       └── labeller_app.py     # Etiquetador con SAM2
├── data/
│   └── samples/                # Frames de ejemplo (en repo)
├── experiments/scripts/        # Scripts de prueba y diagnóstico
├── docs/                       # Documentación
├── requirements.txt
├── EDApp_Launch.bat            # Lanzador Windows
└── .gitignore
```

> **Nota:** Los datasets (`data/datasets/`), modelos entrenados (`runs/`, `*.pt`) y entornos virtuales (`venv_cuda/`, `venv_training/`) están excluidos del repositorio por su tamaño.

### Instalación

**Requisitos:**
- Python 3.10+
- NVIDIA GPU con CUDA 12.1 (recomendado: RTX 2070 o superior)
- Windows 10/11 o Linux

**Pasos:**

```powershell
# 1. Clonar el repositorio
git clone https://github.com/Dudin13/EDApp.git
cd EDApp

# 2. Crear entorno virtual con soporte CUDA
python -m venv venv_cuda
.\venv_cuda\Scripts\activate

# 3. Instalar PyTorch con CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Instalar dependencias del proyecto
pip install -r requirements.txt

# 5. Verificar GPU
python -c "import torch; print(torch.cuda.is_available())"
```

**Lanzar la aplicación:**

```powershell
# Windows (doble clic o desde PowerShell)
.\EDApp_Launch.bat

# Manual
.\venv_cuda\Scripts\python.exe -m streamlit run app/app.py
```

### Entrenamiento del modelo

El dataset está en `C:\apped\data\datasets\hybrid_dataset` con 4 clases finales. Para entrenar:

```powershell
# Entrenamiento completo (jugadores + porteros + árbitros + balón)
.\venv_cuda\Scripts\python.exe ml/training/train_unified.py --target players

# Solo balón (mayor resolución)
.\venv_cuda\Scripts\python.exe ml/training/train_unified.py --target ball --imgsz 1280

# Reanudar entrenamiento interrumpido
.\venv_cuda\Scripts\python.exe ml/training/train_unified.py --target players --resume

# Monitorizar progreso en tiempo real
.\venv_cuda\Scripts\python.exe ml/training/progress_monitor.py
```

**Dataset actual:**
- Train: ~2,340 imágenes
- Val: ~590 imágenes
- Clases: player (5888 inst.), goalkeeper (413), referee (638), ball (2211 imágenes añadidas)

### Variables de entorno

Copia `.env.example` a `.env` y configura:

```env
ROBOFLOW_API_KEY=tu_clave_aqui
APPED_ROOT=C:/apped
```

---

## 🇬🇧 English

### What is EDApp?

EDApp is a football video analysis application built with Python and Streamlit. It detects players, classifies them by team, tracks their position on the pitch, and extracts performance metrics from match videos.

### Pipeline Architecture

The system works in independent layers:

| Layer | Module | Technology | Status |
|-------|--------|------------|--------|
| Detection | `app/modules/detector.py` | YOLO11m-seg | ⚠️ Training |
| Team Classification | `app/modules/team_classifier.py` | KMeans + HSV | ✅ Ready |
| Tracking | `app/modules/tracker.py` | ByteTrack | ✅ Working |
| Field Calibration | `app/modules/calibration_pnl.py` | PnLCalib + Homography | ⚠️ Manual |
| Dorsal Reader | `app/modules/identity_reader.py` | OCR (PARSeq) | ❌ Pending |
| Event Detection | `app/modules/event_spotter_tdeed.py` | T-DEED | ❌ Pending |

### Detection Model Classes

The YOLO model detects 4 classes:

```
0: player      — outfield player
1: goalkeeper  — goalkeeper
2: referee     — referee
3: ball        — football
```

Team differentiation (team A vs team B) is handled in **Layer 2** via jersey color classification — no retraining needed between matches.

### Project Structure

```
C:\apped\
├── app/                        # Streamlit interface
│   ├── app.py                  # Main entry point
│   ├── modules/                # Pipeline modules
│   │   ├── detector.py         # YOLO detection
│   │   ├── team_classifier.py  # Color-based team classification
│   │   ├── tracker.py          # Player tracking
│   │   ├── video_processor.py  # Full pipeline
│   │   └── ...
│   └── pages/                  # App pages
│       ├── upload_analyze.py   # Upload & analyze video
│       ├── player_identification.py  # Identify players
│       ├── match_metrics.py    # Match metrics
│       ├── tactical_map.py     # Tactical map
│       └── ...
├── ml/
│   ├── training/               # Training scripts
│   │   ├── train_unified.py    # Unified training (GPU)
│   │   ├── create_val_split.py # Create validation split
│   │   ├── diagnostico.py      # Dataset & model diagnostics
│   │   ├── remap_classes.py    # Class remapping (6→4)
│   │   ├── add_ball_dataset.py # Add ball dataset
│   │   └── data.yaml           # Dataset configuration
│   └── labeller/               # Labelling tool
│       └── labeller_app.py     # Labeller with SAM2
├── data/
│   └── samples/                # Example frames (in repo)
├── experiments/scripts/        # Test & diagnostic scripts
├── docs/                       # Documentation
├── requirements.txt
├── EDApp_Launch.bat            # Windows launcher
└── .gitignore
```

> **Note:** Datasets (`data/datasets/`), trained models (`runs/`, `*.pt`) and virtual environments (`venv_cuda/`, `venv_training/`) are excluded from the repository due to their size.

### Installation

**Requirements:**
- Python 3.10+
- NVIDIA GPU with CUDA 12.1 (recommended: RTX 2070 or better)
- Windows 10/11 or Linux

**Steps:**

```powershell
# 1. Clone the repository
git clone https://github.com/Dudin13/EDApp.git
cd EDApp

# 2. Create virtual environment with CUDA support
python -m venv venv_cuda
.\venv_cuda\Scripts\activate

# 3. Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install project dependencies
pip install -r requirements.txt

# 5. Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

**Launch the app:**

```powershell
# Windows (double click or from PowerShell)
.\EDApp_Launch.bat

# Manual
.\venv_cuda\Scripts\python.exe -m streamlit run app/app.py
```

### Model Training

The dataset is at `C:\apped\data\datasets\hybrid_dataset` with 4 final classes. To train:

```powershell
# Full training (players + goalkeepers + referees + ball)
.\venv_cuda\Scripts\python.exe ml/training/train_unified.py --target players

# Ball only (higher resolution)
.\venv_cuda\Scripts\python.exe ml/training/train_unified.py --target ball --imgsz 1280

# Resume interrupted training
.\venv_cuda\Scripts\python.exe ml/training/train_unified.py --target players --resume

# Monitor training progress in real time
.\venv_cuda\Scripts\python.exe ml/training/progress_monitor.py
```

**Current dataset:**
- Train: ~2,340 images
- Val: ~590 images
- Classes: player (5888 inst.), goalkeeper (413), referee (638), ball (2211 images added)

### Environment Variables

Copy `.env.example` to `.env` and configure:

```env
ROBOFLOW_API_KEY=your_key_here
APPED_ROOT=C:/apped
```

---

## Roadmap

### ✅ Completado
- [x] Reestructuración del proyecto (app / ml / core / data)
- [x] Dataset preparado con 4 clases finales (player, goalkeeper, referee, ball)
- [x] Descarga y fusión del dataset de balones (2.211 imágenes nuevas)
- [x] Script de entrenamiento unificado con soporte GPU (`train_unified.py`)
- [x] Clasificador de equipos por color de camiseta (`team_classifier.py`)
- [x] UI de identificación de jugadores en Streamlit (`player_identification.py`)
- [x] Repositorio limpio y documentado en GitHub

### 🔜 Próximos pasos

**Inmediato — esta noche**
- [ ] Lanzar entrenamiento nocturno con GPU (`--resume`) y dejar correr ~10h

**Mañana — según métricas**
- [ ] Evaluar mAP50 por clase (objetivo: > 0.5 global)
- [ ] Si OK → integrar `TeamClassifier` al `video_processor.py`
- [ ] Si KO → revisar dataset y ajustar hiperparámetros
- [ ] Segunda ronda `--target ball --imgsz 1280` para mejorar detección del balón

**Siguiente semana**
- [ ] Conectar `player_identification.py` al flujo principal de la app
- [ ] Script de pseudo-labeling (el modelo etiqueta frames nuevos automáticamente)
- [ ] Prueba end-to-end con vídeo VEO real de Cádiz CF Cantera

**Medio plazo**
- [ ] OCR de dorsales con PARSeq
- [ ] Detección de eventos con T-DEED (pases, tiros, duelos)
- [ ] Exportación de reportes PDF por partido y por jugador

---

## 📋 Changelog

### 2026-03-13
- **Dataset**: Remapeo completo de 6 clases antiguas a 4 clases finales (`remap_classes.py`)
- **Dataset**: Descarga y fusión de 2.211 imágenes de balones desde Roboflow (`add_ball_dataset.py`)
- **Dataset**: Generación de split de validación real 85/15 (`create_val_split.py`)
- **Modelo**: Diagnóstico completo del estado del dataset y modelos existentes (`diagnostico.py`)
- **Modelo**: Primer entrenamiento unificado lanzado en GPU RTX 2070 SUPER con CUDA 12.1
- **App**: Nuevo módulo de clasificación de equipos por color de camiseta (`team_classifier.py`)
- **App**: Nueva página de identificación manual de jugadores al inicio del análisis (`player_identification.py`)
- **Repo**: Limpieza del repositorio — eliminadas carpetas antiguas (`football_analyzer`, `Jarvis`, `02_Scripts_Pruebas`, `documentacion`)
- **Repo**: README bilingüe completo (ES/EN) con arquitectura, instalación y roadmap

---

*Desarrollado por Eduardo Pérez — Departamento de Datos, Cádiz CF Cantera*  
*Developed by Eduardo Pérez — Data Department, Cádiz CF Cantera*
