from ultralytics import YOLO
import cv2
from picamera2 import Picamera2
model = YOLO("best_float16.tflite", task = 'detect')

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (320, 320), "format": "RGB888"})
picam2.configure(config)
picam2.start()

known_distance = .81 #placeholder distance for images mark took
ball_height = .4
pixel_height = .0000014

frame = picam2.capture_array()
results = model(frame)

for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        if label.lower() == 'ball':
            pixel_height = abs(y1-y2)

focal_length = (pixel_height * known_distance)/ball_height
print(focal_length)

