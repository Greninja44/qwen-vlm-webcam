"""Grab one frame from the webcam and save it as camera_test.jpg in the project folder."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import time, cv2
from camera import Camera

try:
    cam = Camera()
except RuntimeError as e:
    sys.exit(f"CAMERA FAIL: {e}")
time.sleep(1.5)
f = cam.latest()
if f is None:
    sys.exit("CAMERA FAIL: no frames received")
out = pathlib.Path(__file__).resolve().parents[1] / "camera_test.jpg"
cv2.imwrite(str(out), f)
print(f"CAMERA OK: /dev/video{cam.index}, {f.shape[1]}x{f.shape[0]}, ~{cam.fps:.0f} fps, saved {out}")
cam.close()
