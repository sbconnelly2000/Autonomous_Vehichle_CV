from picamera2 import Picamera2
import time

picam2 = Picamera2()

try:
    picam2.start()
    time.sleep(5)  # Let the camera run for 5 seconds
finally:
    picam2.stop()
