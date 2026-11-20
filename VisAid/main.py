import sys
import time

import cv2

import config
from camera import CameraStream
from detector import Detector
from depth_estimator import DepthEstimator
from description_engine import DescriptionEngine
from speech import Speaker

BUCKET_COLORS = {
    "very close": (0, 0, 255),
    "close": (0, 128, 255),
    "nearby": (0, 200, 200),
    "far": (0, 200, 0),
}

def draw_debug_overlay(frame, detections, depth_map):
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        bucket = DepthEstimator.bucket_for_box(depth_map, det["box"]) if depth_map is not None else "?"
        color = BUCKET_COLORS.get(bucket, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det['label']} ({bucket}) {det['conf']:.2f}"
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def main():
    print("[Main] Starting camera...")
    try:
        cam = CameraStream(
            config.CAMERA_SOURCE, config.FRAME_WIDTH, config.FRAME_HEIGHT
        ).start()
    except RuntimeError as e:
        print(f"[Main] FATAL: {e}")
        sys.exit(1)

    print("[Main] Loading models (this can take 10-30s the first run)...")
    detector = Detector()
    depth_estimator = DepthEstimator()
    engine = DescriptionEngine()
    speaker = Speaker().start()

    show_debug = config.SHOW_DEBUG_WINDOW
    frame_count = 0

    while cam.read() is None:
        time.sleep(0.05)

    print("[Main] Running. Press 'q' to quit, 'd' to toggle debug overlay.")

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            frame_count += 1
            run_models = (frame_count % config.PROCESS_EVERY_N_FRAMES == 0)

            if run_models:
                detections = detector.detect(frame)
                depth_map = depth_estimator.estimate(frame) if detections else None

                sentence = engine.build_sentence(
                    detections, depth_map, frame.shape[1]
                )
                if sentence:
                    print(f"[Speak] {sentence}")
                    speaker.say(sentence)

                if show_debug:
                    frame = draw_debug_overlay(frame, detections, depth_map)

            if show_debug:
                cv2.imshow("Scene Describer (debug)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("d"):
                    show_debug = not show_debug
                    cv2.destroyAllWindows()
            else:
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")

    finally:
        print("[Main] Shutting down cleanly...")
        cam.stop()
        speaker.stop()
        cv2.destroyAllWindows()
        print("[Main] Done.")

if __name__ == "__main__":
    main()
