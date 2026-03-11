# ⚽ EDudin (ED Analytics)
## AI-Powered Football Video Analysis Platform
### STARTUP TECHNICAL BLUEPRINT

**Cross-Functional Team Blueprint**
AI Researcher · Computer Vision Engineer · Sports Performance Analyst · Python / UX Developer

---

## 01 VISIÓN DEL PRODUCTO

### El Problema que Resolvemos
Actualmente, el análisis avanzado de fútbol y la inteligencia artificial están reservados para la élite de la Champions League. Los clubes modestos, academias, e incluso equipos profesionales de divisiones inferiores no pueden permitirse ejércitos de analistas de vídeo ni licencias de software prohibitivas (Wyscout, Hudl). Analizar un partido requiere entre 40 y 60 horas de un analista etiquetando eventos (pases, tiros, presiones) frame a frame.

**EDudin (desarrollado por ED Analytics)** es una plataforma nativa de IA que democratiza el big data táctico. Usando *Computer Vision* "State-of-the-art" (SOTA), automatizamos la recolección de eventos y el *tracking* físico de los jugadores usando únicamente vídeo (sin necesidad de chalecos GPS), haciéndolo accesible a cualquier nivel del fútbol.

### Problemas Clave y Solución EDudin
| Problema | Solución EDudin |
| :--- | :--- |
| **Etiquetado Manual (Tardanza)** | Nuestra "Capa Inteligente" (T-DEED + YOLO11) detecta pases, tiros y córners 100% automático. |
| **Barrera de Coste** | SaaS económico (Suscripción o pago por partido analizado) permitiendo a clubes amateurs tener métricas pro. |
| **Datos Silos** | EDudin unifica el vídeo (Streamlit Player), los eventos (Event Manifest) y la táctica (Pitch 2D) en una sola pestaña. |
| **Sin Insights de IA** | Los softwares actuales sólo te dicen "qué pasó". EDudin (con LLMs) te sugerirá "por qué pasó" detectando patrones. |

---

## 02 PERFIL DE USARIOS OBJETIVO (TARGET USERS)

EDudin está diseñado para 3 arquetipos principales de clientes potenciales dispuestos a pagar por ahorrarse decenas de horas semanales:

1. **Clubes Semiprofesionales / Academias (B2B)**
   - *Necesidad:* Extraer mapas de calor, redes de pases y vídeos de su equipo sin pagar plataformas de élite ni comprar GPS (ya que EDudin calcula la velocidad de los jugadores calibrando la cámara de TV).
2. **Analistas Libres / Creadores de Contenido Deportivo (B2C)**
   - *Necesidad:* Subir un vídeo de un partido de fin de semana y que EDudin devuelva un archivo CSV con las métricas *Expected Threat (xT)* y los clips mágicamente cortados para subirlos a Twitter/YouTube.
3. **Scouting y Ojeadores (B2B/B2C)**
   - *Necesidad:* Encontrar jugadores que destaquen. A través del *Identity Reader* (lectura de dorsales) de EDudin, pueden rastrear el rendimiento individual de un jugador (#9) específico a lo largo de 90 minutos de forma automatizada.

---

## 03 CORE FEATURES (MVP - Funciones Clave)

![EDudin AI Tracking Mockup](C:\Users\Usuario\.gemini\antigravity\brain\51c7658e-8286-435d-851d-249e1972ad52\edudin_ai_tracking_1773075540493.png)
*(Fig 1. Concepto de la Capa de Percepción EDudin con Bounding Boxes y Tracking Físico)*

### 1. Capa de Percepción y Tracking
*   **Seguimiento Multi-Jugador:** Asignación de IDs persistentes mediante cruces de información (YOLO11 + ByteTrack) para no perder al jugador de vista.
*   **Estimación de Carga Física sin GPS:** Usando calibración geométrica de las líneas del campo (`PnLCalibrator`), el modelo traduce los píxeles a metros/segundo para ofrecer datos de *Sprints* y *Distancia Recorrida*.

### 2. Detección Inteligente de Eventos (Spotting)
*   Integración de bases de conocimiento (SoccerNet-v2). El modelo analiza ventanas temporales para marcar **automáticamente** los disparos a puerta, tiros, faltas y tarjetas.
*   El analista sólo tiene que entrar al programa y **validar o ajustar** lo que la IA ha hecho (*Flywheel System* para reentrenar).

### 3. Dashboard Táctico Glassmorphism ("Collective Dashboard")
![EDudin Tactical Dashboard](C:\Users\Usuario\.gemini\antigravity\brain\51c7658e-8286-435d-851d-249e1972ad52\edudin_dashboard_1773075519576.png)
*(Fig 2. Panel Analítico y Mapa Táctico 2D en Streamlit / Matplotlib)*
*   **Visualización Top-Down:** El partido convertido a un mapa 2D interactivo (`mplsoccer`).
*   **Event Timeline Modular:** Posibilidad de cargar la primera y la segunda parte del partido para una correcta cronología asíncrona.
*   **Filtrado de Peligro (xT / Progressive Actions):** Extracción de vídeo automática. El usuario le pide a EDudin "Dame los clips de los 5 pases progresivos más peligrosos del equipo visitante", y se descargan en formato MP4.

---

## 04 ROADMAP DE STARTUP (ESTRATEGIA EDUDIN)

### Fase 1 — Alpha / Prueba de Concepto (M1-M3)
- [x] *Core Loop* desarrollado: Pipeline base subiendo vídeo local, detecciones con YOLO11 y visualizador en Streamlit básico.
- [ ] Integración de datasets masivos de entrenamiento (DFL, SoccerNet, MS COCO) para afinar el Modelo "EDudin v2" y reducir fallos del balón.

### Fase 2 — MVP Comercial & Beta Privada (M4-M6)
- [ ] Refinamiento del UI en Streamlit con diseño Premium (Glassmorphism, Dark Mode).
- [ ] Inclusión de recortes de video independientes (ClipMaker feature) y métricas avanzadas como Redes de Pases automáticas.
- [ ] Fichaje de 3-5 equipos B locales (Tercera RFEF / División Honor Juvenil) para que prueben el software gratis y nos corrijan fallos (alimentación humana de la IA).

### Fase 3 — Lanzamiento Cloud B2B (M7-M12)
- [ ] Migración de la inferencia pesada (los modelos PyTorch) de local a un servidor GPU en la Nube (AWS EC2 / RunPod).
- [ ] Implementación de "Generative AI" táctica (ej: Integrar la API de Gemini para que el bot de EDudin responda preguntas naturales sobre el partido basadas en el JSON detectado).
- [ ] Modelo de subscripción SaaS (Mínima: 50€/mes - Máxima Team: 250€/mes).

---

## 05 LO QUE NOS HACE ÚNICOS (VENTAJAS COMPETITIVAS)

1. **Arquitectura No-Billion Dollar:** No dependemos de servidores gigantes. Toda la arquitectura en Python y Streamlit hace que sea rapidísimo iterar.
2. **Sistema Híbrido Analista-IA *(The Data Flywheel)* :** Aceptamos que ningún modelo es perfecto hoy. EDudin promueve que el Scout corrija los fallos (con teclas rápidas). Cada clic que hace un usuario mejorando una predicción fallida de EDudin, guarda la corrección para reentrenar nuestra propia IA en el futuro, creando un foso competitivo (*moat*) imposible de replicar.
3. **Clip Engine:** Cortar el flujo de vídeo para el analista no es tedioso; simplemente se filtran eventos por "Peligrosidad" y se exportan clips al instante sin editores de vídeo en medio.
