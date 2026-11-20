import torch
from ultralytics import YOLO

import config

class Detector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(config.YOLO_MODEL)
        self.model.to(self.device)
        print(f"[Detector] YOLO loaded on {self.device}")

    def detect(self, frame):
        results = self.model.predict(
            frame,
            conf=config.YOLO_CONFIDENCE,
            verbose=False,
            device=self.device,
        )

        detections = []
        if not results:
            return detections

        r = results[0]
        if r.boxes is None:
            return detections

        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            detections.append({
                "label": label,
                "conf": conf,
                "box": (x1, y1, x2, y2),
            })
        return detections
