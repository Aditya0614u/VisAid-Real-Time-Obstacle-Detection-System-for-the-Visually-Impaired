import pyttsx3
import queue
import threading

import config

class Speaker:
    def __init__(self):
        self._queue = queue.Queue()
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        return self

    def _worker(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", config.TTS_RATE)

        while self._running:
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if text is None:
                break
            engine.say(text)
            engine.runAndWait()

    def say(self, text):
        if not text:
            return
        if self._queue.qsize() < 5:
            self._queue.put(text)

    def stop(self):
        self._running = False
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
