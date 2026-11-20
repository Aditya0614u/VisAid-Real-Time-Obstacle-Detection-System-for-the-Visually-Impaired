import cv2
import torch
import numpy as np

import config

class DepthEstimator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = torch.hub.load("intel-isl/MiDaS", config.DEPTH_MODEL_TYPE)
        self.model.to(self.device)
        self.model.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = transforms.small_transform

        print(f"[DepthEstimator] MiDaS ({config.DEPTH_MODEL_TYPE}) loaded on {self.device}")

    @torch.no_grad()
    def estimate(self, frame_bgr):
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)

        prediction = self.model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = prediction.cpu().numpy()

        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min < 1e-6:
            return np.zeros_like(depth_map)
        return (depth_map - d_min) / (d_max - d_min)

    @staticmethod
    def bucket_for_box(depth_map, box):
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = depth_map.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return "far"

        region = depth_map[y1:y2, x1:x2]
        score = float(np.median(region))

        for threshold, label in config.DISTANCE_BUCKETS:
            if score >= threshold:
                return label
        return config.DISTANCE_BUCKETS[-1][1]
