from ultralytics import YOLO
import cv2
from angle_utils import get_object_angles_px

# ✅ Use raw string to fix invalid escape sequence
model = YOLO("best_float16.tflite", task='detect')

# Load and resize image
frame = cv2.imread("test_image_1.JPEG")
frame = cv2.resize(frame, (640, 640))

# Calibration constants
focal_length = 700  # placeholder
focal_length_test_images = 1000  # calculated externally

# Object heights in meters
ball_height = 0.4
TG_height = 1.39
SG_height = 1.16
CG_height = 1.13

# Run inference
results = model(frame)

# Store inference data
inference_data = []

for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        theta_x, theta_y = get_object_angles_px(x_center, y_center)

        # Visual overlay
        cv2.circle(frame, (int(x_center), int(y_center)), 4, (0, 255, 0), -1)
        cv2.putText(frame, f"X:{theta_x:.1f} Y:{theta_y:.1f}",
                    (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

        # Distance estimation
        pixel_height = abs(y2 - y1)
        distance = None
        if pixel_height > 0:
            label_lower = label.lower()
            if label_lower == 'ball':
                distance = (ball_height * focal_length_test_images) / pixel_height
            elif label_lower in ['orange circle goal', 'yellow circle goal']:
                distance = (CG_height * focal_length_test_images) / pixel_height
            elif label_lower in ['orange triangle goal', 'yellow triangle goal']:
                distance = (TG_height * focal_length_test_images) / pixel_height
            elif label_lower in ['orange square goal', 'yellow square goal']:
                distance = (SG_height * focal_length_test_images) / pixel_height

        # Append to 2D array
        inference_data.append([
            label,
            round(conf, 2),
            round(theta_x, 2),
            round(theta_y, 2),
            round(distance, 2) if distance else None
        ])

        # Color mapping and drawing
        color_map = {
            'ball': (255, 0, 0),
            'yellow circle goal': (0, 255, 255),
            'orange circle goal': (0, 165, 255),
            'yellow square goal': (0, 255, 0),
            'orange square goal': (0, 128, 0),
            'yellow triangle goal': (255, 255, 0),
            'orange triangle goal': (0, 140, 255)
        }
        color = color_map.get(label_lower, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Print 2D array
print("Inference Results:")
for row in inference_data:
    print(row)

# Show image
cv2.imshow("YOLOv8 Image Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()