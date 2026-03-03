from roboflow import Roboflow
import cv2
import numpy as np

rf = Roboflow(api_key='1FzYekbvWJQGv1HYQRvE')
project = rf.workspace('roboflow-jvuqo').project('football-players-detection-3zvbc')
model = project.version(1).model

cap = cv2.VideoCapture('C:/Users/Usuario/Desktop/VideosDePrueba/j23mirandilla-vs-cccfb-2026-02-21.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 3175)
ret, frame = cap.read()
cap.release()

if ret:
    h_img, w_img = frame.shape[:2]

    CAMPO_Y_MIN = int(h_img * 0.28)
    CAMPO_Y_MAX = int(h_img * 0.68)
    CAMPO_X_MIN = int(w_img * 0.01)
    CAMPO_X_MAX = int(w_img * 0.99)

    zona_campo = frame[CAMPO_Y_MIN:CAMPO_Y_MAX, CAMPO_X_MIN:CAMPO_X_MAX]
    h_zona, w_zona = zona_campo.shape[:2]

    mitad_izq = zona_campo[:, :w_zona//2]
    mitad_der = zona_campo[:, w_zona//2:]

    cv2.imwrite('C:/apped/frame_izq.jpg', mitad_izq)
    cv2.imwrite('C:/apped/frame_der.jpg', mitad_der)

    result_izq = model.predict('C:/apped/frame_izq.jpg', confidence=40, overlap=25)
    result_der = model.predict('C:/apped/frame_der.jpg', confidence=40, overlap=25)

    detecciones = []
    for pred in result_izq.predictions:
        if pred['class'] in ['player', 'goalkeeper', 'referee']:
            detecciones.append({
                'x': int(pred['x']) + CAMPO_X_MIN,
                'y': int(pred['y']) + CAMPO_Y_MIN,
                'w': int(pred['width']),
                'h': int(pred['height']),
                'clase': pred['class']
            })
    for pred in result_der.predictions:
        if pred['class'] in ['player', 'goalkeeper', 'referee']:
            detecciones.append({
                'x': int(pred['x']) + w_zona//2 + CAMPO_X_MIN,
                'y': int(pred['y']) + CAMPO_Y_MIN,
                'w': int(pred['width']),
                'h': int(pred['height']),
                'clase': pred['class']
            })

    print(f'Jugadores detectados: {len(detecciones)}')

    def get_torso_color(img, x, y, w, h):
        x1 = max(0, x - w//4)
        x2 = min(img.shape[1], x + w//4)
        y1 = max(0, y - h//4)
        y2 = min(img.shape[0], y + h//4)
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        return np.mean(hsv, axis=(0, 1))

    equipos = []
    for det in detecciones:
        # Si el modelo ya lo detecta como arbitro
        if det['clase'] == 'referee':
            equipos.append(2)
            continue

        # Filtrar por tamaño minimo
        area = det['w'] * det['h']
        if area < 800:
            equipos.append(-1)
            continue

        color = get_torso_color(frame, det['x'], det['y'], det['w'], det['h'])
        if color is None:
            equipos.append(0)
            continue

        h, s, v = color

        # Arbitro rojo
        if (0 <= h <= 10 or 170 <= h <= 180) and s > 100:
            equipos.append(2)
        # Amarillo: H entre 20-40, S alto
        elif 20 <= h <= 40 and s > 80:
            equipos.append(0)
        # Blanco: S bajo
        elif s < 60:
            equipos.append(1)
        # Verde rayas: H entre 55-90
        elif 55 <= h <= 90 and s > 60:
            equipos.append(1)
        else:
            equipos.append(0)

    frame_result = frame.copy()
    colores_box = [(0, 220, 0), (0, 140, 255), (0, 0, 255)]
    nombres_eq = ['Amarillo', 'Blanco/Verde', 'Arbitro']

    cv2.rectangle(frame_result,
                 (CAMPO_X_MIN, CAMPO_Y_MIN),
                 (CAMPO_X_MAX, CAMPO_Y_MAX),
                 (255, 255, 0), 1)

    for i, det in enumerate(detecciones):
        eq = equipos[i]
        if eq == -1:
            continue
        color = colores_box[eq]
        x1 = det['x'] - det['w']//2
        y1 = det['y'] - det['h']//2
        x2 = det['x'] + det['w']//2
        y2 = det['y'] + det['h']//2
        cv2.rectangle(frame_result, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_result, nombres_eq[eq], (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    cv2.imwrite('C:/apped/test_equipos.jpg', frame_result)
    print('Imagen guardada en C:/apped/test_equipos.jpg')