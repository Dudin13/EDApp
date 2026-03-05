from ultralytics import YOLO
import cv2

model = YOLO(r"C:\apped\train_yolo\runs\segment\train\weights\best.pt")
cap = cv2.VideoCapture(r"C:\apped\dataset\test_segmentation_video.mp4")

frame_count = 0
ball_found = 0

while cap.isOpened() and frame_count < 200:
    ret, frame = cap.read()
    if not ret: break
    
    # Predecir con confianza muuuy baja para ver si YOLO "ve" algo del balon
    results = model(frame, conf=0.05, verbose=False)
    
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 3: # 3 = ball en nuestro data.yaml
                ball_found += 1
                
    frame_count += 1

print(f"Frames analizados: {frame_count}")
print(f"Balones detectados (con conf=0.05): {ball_found}")

cap.release()
