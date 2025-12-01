from ultralytics import YOLO
import cv2
import time
from picamera2 import Picamera2
from calculate_angle import get_object_angles_px

# Load the YOLO model
model = YOLO("best_int8_320.tflite", task='detect')

# Initialize the Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (320, 320), "format": "RGB888"})
picam2.configure(config)
picam2.start()

# Constants
focal_length = 235  # Placeholder, calibrate later
ball_height = 0.4
TG_height = 1.39
SG_height = 1.16
CG_height = 1.13
frame_rate = 10
prev = 0

# Function to process each image
def process_image(frame):
    # Resize to model input shape
    #frame_resized = cv2.resize(frame, (320, 320))

    # Run YOLOv8 inference
    results = model(frame)

    # Store inference data for this frame
    inference_data = []

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Extract bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            # Draw box around detected object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Calculate angles
            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
            theta_x, theta_y = get_object_angles_px(x_center, y_center)

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
                    distance = (ball_height * focal_length) / pixel_height
                elif label_lower in ['orange circle goal', 'yellow circle goal']:
                    distance = (CG_height * focal_length) / pixel_height
                elif label_lower in ['orange triangle goal', 'yellow triangle goal']:
                    distance = (TG_height * focal_length) / pixel_height
                elif label_lower in ['orange square goal', 'yellow square goal']:
                    distance = (SG_height * focal_length) / pixel_height

            inference_data.append([
                label,
                round(conf, 2),
                round(theta_x, 2),
                round(theta_y, 2),
                round(distance, 2) if distance else None
            ])

    print("Frame Inference Data:")
    for row in inference_data:
        print(row)

    # Display the frame
    cv2.imshow("YOLOv8 Pi Camera", frame)

    return cv2.waitKey(1) & 0xFF == ord('q')

while True:
    time_elapsed = time.time() - prev

    if time_elapsed > 1. / frame_rate:
        prev = time.time()

        # Capture a frame from the picamera2
        frame = picam2.capture_array()

        # Check the shape of the frame
        print("Frame shape:", frame.shape)  # Print the shape for debugging

        # Handle conversion appropriately
        if frame.shape[2] == 1:  # If it's a single channel (Y or YUV)
            print("Y OR YUV")
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
        elif frame.shape[2] == 3:  # If it's already BGR
            print("BGR")
            pass  # No conversion needed
        elif frame.shape[2] == 4:  # If it's BGRA
            print("BGRA")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            print("Unexpected frame shape, skipping processing")
            continue

        # Process the captured frame
        if process_image(frame):
            break


# Cleanup
picam2.stop()
cv2.destroyAllWindows()
