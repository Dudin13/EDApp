from ultralytics import YOLO
model = YOLO("C:/D/New folder/detect_players.pt")
print(model.names)
