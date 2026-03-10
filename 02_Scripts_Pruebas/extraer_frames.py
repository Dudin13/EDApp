import cv2
import os

video_path = 'C:/Users/Usuario/Desktop/VideosDePrueba/j23mirandilla-vs-cccfb-2026-02-21.mp4'
output_dir = 'C:/apped/frames_etiquetado'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duracion_min = (total_frames / fps) / 60

print(f'FPS: {fps}')
print(f'Total frames: {total_frames}')
print(f'Duración: {duracion_min:.1f} minutos')

# Extraer 1 frame cada 2 minutos de juego real
# Empezamos en el minuto 2 (cuando ya está el partido en marcha)
inicio = int(2 * 60 * fps)
intervalo = int(2 * 60 * fps)

frames_guardados = 0
frame_actual = inicio

while frame_actual < total_frames and frames_guardados < 50:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_actual)
    ret, frame = cap.read()
    if ret:
        minuto = frame_actual / fps / 60
        nombre = f'frame_{frames_guardados:03d}_min{minuto:.1f}.jpg'
        cv2.imwrite(f'{output_dir}/{nombre}', frame)
        print(f'Guardado: {nombre}')
        frames_guardados += 1
    frame_actual += intervalo

cap.release()
print(f'\nTotal frames extraidos: {frames_guardados}')
print(f'Carpeta: {output_dir}')