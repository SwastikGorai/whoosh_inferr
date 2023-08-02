from ultralytics import YOLO
model = YOLO("YOLO_Custom_v8m.pt")

model.predict(source="vid.mp4", save=True, conf = 0.37, stream=True, )  # source=0 for capture from webcam/camera. Change value to the desired camera.