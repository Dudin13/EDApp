# ED Analytics - Business Plan (Draft)

## 1. Resumen Ejecutivo
ED Analytics es una plataforma web avanzada de análisis de fútbol táctico impulsada por Inteligencia Artificial (IA) y Visión por Computadora (CV). Diseñada para analistas de fútbol, entrenadores y equipos técnicos, la herramienta automatiza el proceso de "scouting" y análisis de partidos. Usando modelos de *deep learning* de vanguardia (YOLO11x, ByteTrack, KMeans), ED Analytics transforma vídeo "crudo" en datos tácticos accionables (coordenadas, mapas de calor, redes de pases, tracking de jugadores e identificación de equipos), reduciendo drásticamente el tiempo de análisis manual de horas a minutos.

**Misión:** Democratizar el análisis táctico de élite, proporcionando a equipos de todo los niveles herramientas tecnológicas que tradicionalmente solo estaban al alcance de los presupuestos de la Champions League.

## 2. Definición del Problema
*   **Tiempo Consumido:** Los analistas de vídeo modernos pasan un promedio de 6-8 horas etiquetando manualmente acciones (pases, tiros, faltas) por cada hora de metraje de partido.
*   **Ausencia de Datos Cuantitativos Inmediatos:** Para los equipos sub-élite o academias, obtener mapas de calor o tracking de velocidad precisa de jugadores requiere de sistemas de hardware costosos (GPS en chalecos o sistemas multicámara en estadios).
*   **Silos de Herramientas:** El análisis de vídeo, la creación de informes de datos y la presentación táctica (pizarras, dibujos sobre vídeo tipo KlipDraw) suelen estar separados en 2 o 3 suscripciones de software diferentes.

## 3. Nuestra Solución (Producto)
ED Analytics es un ecosistema "Todo en Uno" construido sobre Python, Streamlit y PyTorch:
*   **Motor de IA "EDudin v1":** Nuestro modelo especializado en fútbol detecta a los jugadores, sus cruces (oclusiones), al árbitro y la trayectoria del balón a alta velocidad.
*   **Dashboards Analíticos Avanzados:** Módulo Scout para etiquetado asistido con una Botonera Dinámica; Mapas Tácticos para la representación 2D cenital (Heatmaps integrados con `mplsoccer`); y Timelines de Eventos estilo *Glassmorphism*.
*   **Exportación Táctica Total:** Herramienta integrada de dibujo sobre el vídeo (lápiz, flechas) exportable directamente como un clip MP4 con las animaciones incrustadas (motor FFmpeg).

## 4. Mercado Objetivo y Clientes Potenciales (B2B y B2C)
*   **Clientela Principal (Nivel Medio/Sub-Élite):**
    *   Equipos de 2ª y 3ª división, semi-profesionales y academias de alto rendimiento que no pueden pagar suscripciones a *Sportcode* + *Instat* + *Wyscout* pero necesitan los mismos insights.
*   **Usuarios Finales (Productivity B2C):**
    *   Analistas de fútbol freelance, creadores de contenido táctico en YouTube/Twitch, o estudiantes universitarios de ciencia deportiva.

## 5. Modelo de Negocio (Monetización)
Proponemos un modelo **SaaS (Software as a Service) por niveles (Tiers)**:
1.  **Tier Básico (Analista Freelance):** x€/mes. 
    *   Soporte para 1 Liga/Equipo, importación y módulo Scout manual (Botones) e informes estáticos en PDF.
2.  **Tier Pro (Coach / Team Analytics):** y€/mes.
    *   Procesamiento de IA (motor local o en nube ligera), generación de mapas tácticos (`mplsoccer`), y timeline de eventos en vídeo.
3.  **Tier Enterprise (Clubes / Academias):** z€/mes-año.
    *   Tracking completo (ByteTrack), calibración automática de cámara (2D y 3D), API de exportación de datos puros (formato `Kloppy` o JSON compatible con FIFA) y almacenamiento ilimitado de plantillas de scouting.

## 6. Panorama Competitivo
*   **Los Gigantes (Hudl Sportscode, Wyscout, StatsBomb):** Costosos, orientados enteramente a los clubes Tier 1.
*   **Alternativas manuales (Nacsport, LongoMatch):** Muy consolidados en recortes de vídeo manuales pero lentos en adoptar IA generativa o tracking en tiempo real accesible desde una cuenta en navegador.
*   **Nuestra Ventaja (Unique Value Proposition - UVP):** 
    *   Simplicidad radical (una única App UI/UX prémium en navegador web o local).
    *   Procesamiento de IA 100% privado/local si se requiere (el equipo lo puede arrancar en el avión de regreso del partido gracias a que funciona en un portátil con gráfica dedicada usando YOLO local).

## 7. Hoja de Ruta (Roadmap y Adopción de Tecnología)
*   **Q1: Fundamentos Hitos Logrados:** 
    *   Lanzamiento módulo de Scout, exportación FFmpeg y UI Premium. 
    *   Entrenamiento exitoso del modelo Core `EDudin_v1` (YOLO11x).
*   **Q2: Escalada Táctica (Actual):** 
    *   Implementación de ByteTrack + KMeans para diferenciación automática de equipos y seguimiento persistente.
    *   Lanzamiento del Event Dashboard avanzado (Glassmorphism).
*   **Q3: Inteligencia Predictiva:** 
    *   Redes Neuronales de Grafos (GNN) y Clasificadores de Acción (basados en *SoccerNet-v2* o *TrackNet*).
    *   Auto-etiquetado. El analista solo sube el vídeo y la herramienta detecta "Pase Peligroso", "Contraataque", o "Tiro a puerta" automáticamente.
*   **Q4: Continual Learning y Arquitectura Dual (Dual-Model):**
    *   **Test-Time Training (TTT)** (Fine-tuning en caliente) para adaptar dinámicamente el modelo a las condiciones de luz/césped de estadios específicos usando los primeros fotogramas seguros.
    *   **Arquitectura Dual de Seguridad (Shadow Mode):** Mantener un modelo **Base robusto** intocable y un modelo **Activo de aprendizaje diario**. El modelo activo se re-entrena por las noches con los casos difíciles (Hard Examples) corregidos por los analistas. Si el modelo activo sufre de "olvido catastrófico" (se vuelve inestable), el sistema hace un "rollback" limpio al modelo Base.

## 8. Siguientes Pasos
Validar el modelo técnico (finalizar el entrenamiento YOLOv8 actual y comprobar la tasa de acierto) para luego empaquetar una Beta "MVP" que un usuario pueda arrancar de forma unificada desde un instalador `.exe` o contenedor de Docker fácil para Windows.
