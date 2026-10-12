"""Webcam reader running in its own thread, independent of inference."""
import threading, time
import cv2
import config


def open_camera():
    indices = [config.CAMERA_INDEX] if config.CAMERA_INDEX is not None else range(4)
    denied = False
    for i in indices:
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAM_FPS)
        ok, _ = cap.read()
        if ok:
            return cap, i
        cap.release()  # e.g. a metadata-only /dev/videoN node
    import glob, os
    nodes = glob.glob("/dev/video*")
    if not nodes:
        raise RuntimeError("No webcam found (no /dev/video*). Check the USB cable / try another port.")
    if not any(os.access(n, os.R_OK | os.W_OK) for n in nodes):
        raise RuntimeError("Camera permission denied. Run: sudo usermod -aG video $USER  (then log out/in)")
    raise RuntimeError("Could not read frames from the camera. Is another program using it? "
                       "(check: fuser /dev/video*)")


class Camera:
    def __init__(self):
        self.cap, self.index = open_camera()
        self.frame = None
        self.fps = 0.0
        self._lock = threading.Lock()
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        n, t0 = 0, time.time()
        while self._run:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self.frame = f
            n += 1
            if time.time() - t0 >= 1:
                self.fps, n, t0 = n / (time.time() - t0), 0, time.time()

    def latest(self):
        with self._lock:
            return None if self.frame is None else self.frame.copy()

    def close(self):
        self._run = False
        time.sleep(0.1)
        self.cap.release()
