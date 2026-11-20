import time

import config

class DescriptionEngine:
    def __init__(self):
        self._state = {}

    def _position_for_box(self, box, frame_width):
        x1, _, x2, _ = box
        cx = (x1 + x2) / 2.0
        frac = cx / frame_width
        if frac < config.LEFT_BOUND:
            return "on your left"
        elif frac > config.RIGHT_BOUND:
            return "on your right"
        return "ahead"

    def _priority(self, label):
        return config.PRIORITY.get(label, config.DEFAULT_PRIORITY)

    def _bucket_rank(self, bucket_label):
        for i, (_, label) in enumerate(config.DISTANCE_BUCKETS):
            if label == bucket_label:
                return i
        return len(config.DISTANCE_BUCKETS)

    def build_sentence(self, detections, depth_map, frame_width):
        from depth_estimator import DepthEstimator

        enriched = []
        for det in detections:
            label = det["label"]
            box = det["box"]
            position = self._position_for_box(box, frame_width)
            bucket = (
                DepthEstimator.bucket_for_box(depth_map, box)
                if depth_map is not None else "nearby"
            )
            enriched.append({
                "label": label,
                "position": position,
                "bucket": bucket,
                "priority": self._priority(label),
            })

        enriched.sort(key=lambda d: (d["priority"], self._bucket_rank(d["bucket"])))

        now = time.time()
        to_announce = []
        for d in enriched:
            key = (d["label"], d["position"])
            prev = self._state.get(key)

            is_new = prev is None
            distance_changed = prev is not None and prev["bucket"] != d["bucket"]
            cooldown_expired = (
                prev is not None
                and (now - prev["last_spoken"]) >= config.RE_ANNOUNCE_COOLDOWN_SECONDS
            )

            if is_new or distance_changed or cooldown_expired:
                to_announce.append(d)
                self._state[key] = {"bucket": d["bucket"], "last_spoken": now}

            if len(to_announce) >= config.MAX_OBJECTS_PER_ANNOUNCEMENT:
                break

        stale_keys = [
            k for k, v in self._state.items()
            if now - v["last_spoken"] > config.RE_ANNOUNCE_COOLDOWN_SECONDS * 5
        ]
        for k in stale_keys:
            del self._state[k]

        if not to_announce:
            return None

        parts = [
            f"{d['label']} {d['bucket']} {d['position']}"
            for d in to_announce
        ]
        return "; ".join(parts) + "."
