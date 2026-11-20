CAMERA_SOURCE = 1

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

PROCESS_EVERY_N_FRAMES = 5

YOLO_MODEL = "yolov8n.pt"
YOLO_CONFIDENCE = 0.45

DEPTH_MODEL_TYPE = "MiDaS_small"

DISTANCE_BUCKETS = [
    (0.75, "very close"),
    (0.55, "close"),
    (0.35, "nearby"),
    (0.0,  "far"),
]

PRIORITY = {
    "person": 1, "car": 1, "bicycle": 1, "motorcycle": 1, "bus": 1, "truck": 1,
    "stairs": 1,
    "chair": 2, "dining table": 2, "couch": 2, "bed": 2, "door": 2,
    "potted plant": 2, "bench": 2,
    "bottle": 3, "cup": 3, "laptop": 3, "backpack": 3, "handbag": 3,
    "cell phone": 3, "book": 3, "tv": 3,
}
DEFAULT_PRIORITY = 4
MAX_OBJECTS_PER_ANNOUNCEMENT = 3

RE_ANNOUNCE_COOLDOWN_SECONDS = 6.0

LEFT_BOUND = 0.35
RIGHT_BOUND = 0.65

TTS_RATE = 175

SHOW_DEBUG_WINDOW = True
