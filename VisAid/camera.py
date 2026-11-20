import cv2
import threading
import time

class CameraStream:
    def __init__(self, source=0, width=640, height=480):
        self._source = source
        is_network_stream = isinstance(source, str)

        self.cap = (
            cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if is_network_stream else cv2.VideoCapture(source)
        )

        if not self.cap.isOpened():
            hint = (
                "Check the phone and laptop are on the SAME Wi-Fi network, "
                "DroidCam app is running and showing that URL, and the IP "
                "hasn't changed since you copied it."
                if is_network_stream else
                "Try a different CAMERA_SOURCE in config.py (0, 1, 2...), or "
                "check that no other app is using the camera."
            )
            raise RuntimeError(f"Could not open camera source '{source}'. {hint}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        return self

    def _update(self):
        consecutive_failures = 0
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures >= 20:
                    self.cap.release()
                    self.cap.open(self._source, cv2.CAP_FFMPEG) \
                        if isinstance(self._source, str) else self.cap.open(self._source)
                    consecutive_failures = 0
                time.sleep(0.05)
                continue
            consecutive_failures = 0
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.cap.release()
