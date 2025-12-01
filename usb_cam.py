from ultralytics import YOLO
import cv2

# ✅ Use raw string to fix invalid escape sequence
model = YOLO("best_float16.tflite", task = 'detect')

cam = cv2.VideoCapture(0)

focal_length = 700  # in pixels (you’ll want to calibrate this)
#need to calculate focal length once pi cam works


#in meters
ball_height= 0.4
TG_height = 1.39 
SG_height = 1.16
CG_height = 1.13


while True:
    ret, frame = cam.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 640))
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ✅ Define variable first
            ball_distance = None
            CG_distance = None
            TG_distance = None
            SG_distance = None

            if label.lower() == 'ball': 
                pixel_height = abs(y2 - y1)
                if pixel_height > 0:
                    ball_distance = (ball_height * focal_length) / pixel_height
            elif label.lower() == 'orange circle goal' or label.lower() == "yellow circle goal":  
                pixel_height = abs(y2 - y1)
                if pixel_height > 0:
                    CG_distance = (CG_height * focal_length) / pixel_height
            elif label.lower() == 'orange triangle goal' or label.lower() == "yellow triangle goal":  
                pixel_height = abs(y2 - y1)
                if pixel_height > 0:
                    TG_distance = (TG_height * focal_length) / pixel_height
            elif label.lower() == 'orange square goal' or label.lower() == "yellow square goal":  
                pixel_height = abs(y2 - y1)
                if pixel_height > 0:
                    SG_distance = (SG_height * focal_length) / pixel_height

            print(f"""
            Class: {label}
            Confidence: {conf:.2f}
            BBox: [{x1}, {y1}, {x2}, {y2}]
            Ball Distance (m): {ball_distance}
            SG Distance (m): {SG_distance}
            CG Distance (m): {CG_distance}
            TG Distance (m): {TG_distance}
            """)


    cv2.imshow("YOLOv8 Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
