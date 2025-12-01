import os
os.environ.setdefault("OMP_NUM_THREADS", "4")  # lightweight hint on many CPUs

from ultralytics import YOLO
import cv2, time, threading, queue

from angle_utils import get_object_angles_px

# ====== Tunables ======
MODEL_PATH = "best_int8_320.tflite"  # try INT8 @ 320 for CPU; fallback to your FP16 if needed
IMGSZ = 320
CAP_W, CAP_H = 640, 360
PROCESS_EVERY_N = 2        # run detector every Nth frame
CONF_THRES = 0.45
IOU_THRES  = 0.50
MAX_FPS    = 20            # UI throttle, not a hard cap

# Distance parameters (your originals)
focal_length = 700
ball_height = 0.4
TG_height   = 1.39
SG_height   = 1.16
CG_height   = 1.13

# ====== Setup ======
cv2.setUseOptimized(True)
cv2.setNumThreads(2)

model = YOLO(MODEL_PATH, task="detect")  # your previous: best_float16.tflite
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
cam.set(cv2.CAP_PROP_FPS, MAX_FPS)

# Capture thread to keep frames flowing
frame_q: "queue.Queue[tuple[bool, any]]" = queue.Queue(maxsize=2)
def capture_loop():
    while True:
        ret, frame = cam.read()
        if not ret:
            frame_q.put((False, None))
            break
        # smallest cheap resize once, then letterbox later in model
        frame_q.put((True, frame))
        # keep queue lean
        while frame_q.qsize() > 1:
            try:
                frame_q.get_nowait()
            except queue.Empty:
                break

cap_th = threading.Thread(target=capture_loop, daemon=True)
cap_th.start()

frame_idx = 0
last_results = None
last_inference_data = []

prev = 0.0
while True:
    ret, frame = frame_q.get()
    if not ret or frame is None:
        print("Camera error")
        break

    # Throttle display loop a bit
    now = time.time()
    if now - prev < 1.0 / MAX_FPS:
        # Drop to keep UI responsive
        pass
    prev = now

    run_detect = (frame_idx % PROCESS_EVERY_N == 0)
    frame_idx += 1

    if run_detect:
        # Ultralytics handles letterbox internally when given imgsz
        # stream=False returns a list; that's fine at low N
        results = model.predict(
            source=frame,
            imgsz=IMGSZ,
            conf=CONF_THRES,
            iou=IOU_THRES,
            verbose=False
        )
        last_results = results
        last_inference_data = []

    # Draw using last_results (even on skipped frames)
    disp = frame.copy()
    if last_results is not None:
        r = last_results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])

                # light rectangle + one line of text (keep it cheap)
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0,255,0), 2)

                # angles
                xc, yc = (x1+x2)/2.0, (y1+y2)/2.0
                theta_x, theta_y = get_object_angles_px(xc, yc)
                cv2.circle(disp, (int(xc), int(yc)), 3, (0,255,0), -1)

                # distance by class
                ph = max(1, abs(y2 - y1))
                dst = None
                ll = label.lower()
                if ll == 'ball':
                    dst = (ball_height * focal_length) / ph
                elif ll in ['orange circle goal', 'yellow circle goal']:
                    dst = (CG_height * focal_length) / ph
                elif ll in ['orange triangle goal', 'yellow triangle goal']:
                    dst = (TG_height * focal_length) / ph
                elif ll in ['orange square goal', 'yellow square goal']:
                    dst = (SG_height * focal_length) / ph

                # one compact overlay
                txt = f"{label} {conf:.1f} | X:{theta_x:.1f} Y:{theta_y:.1f}"
                if dst is not None:
                    txt += f" | D:{dst:.2f}m"
                cv2.putText(disp, txt, (x1, max(0, y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1, cv2.LINE_AA)

                if run_detect:
                    last_inference_data.append([
                        label, round(conf,1), round(theta_x,2), round(theta_y,2),
                        round(dst,2) if dst is not None else None
                    ])

    # Print only when new inference was run
    if run_detect and last_inference_data:
        print("Frame Inference Data:")
        for row in last_inference_data:
            print(row)

    cv2.imshow("YOLO FastCam", disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

