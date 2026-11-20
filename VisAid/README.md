# Scene Describer

Real-time scene description for blind users: camera → YOLOv8 (what) →
MiDaS depth (how far) → priority-filtered description engine → speech.

Runs fully locally, tuned for a 6GB GPU (RTX 4050).

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install CUDA-enabled torch FIRST (check your CUDA version — 4050 supports 12.1+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

First run will download `yolov8n.pt` (~6MB) and the MiDaS_small weights
(~80MB) automatically — needs internet the first time only.

## Run

```bash
python main.py
```

- Press `q` to quit
- Press `d` to toggle the debug overlay (bounding boxes + distance buckets)
- Ctrl+C also shuts everything down cleanly (camera released, threads joined)

## How it avoids the usual bugs

| Problem | Fix |
|---|---|
| Video feed lags/freezes over time | Camera runs on its own thread, always serves the *latest* frame only (`camera.py`) |
| TTS freezes the whole app while speaking | TTS runs on its own thread + queue, `say()` is non-blocking (`speech.py`) |
| App narrates every object every frame ("chair chair chair...") | State-memory in `description_engine.py` only re-announces on new object, distance-bucket change, or after a cooldown |
| Fake precise distances ("2.03 meters") | MiDaS gives *relative* depth only — bucketed into very close/close/nearby/far, never presented as metric |
| GPU/CPU overload trying to run every frame through both models | `PROCESS_EVERY_N_FRAMES` in `config.py` throttles model calls while camera still streams smoothly |
| Camera left open / thread leaks on crash | `try/finally` in `main.py` guarantees `cam.stop()` and `speaker.stop()` run even on Ctrl+C or exception |

## Tuning

Everything lives in `config.py`:
- `YOLO_CONFIDENCE` — raise if you get false positives, lower if it's missing things
- `DISTANCE_BUCKETS` — run with the debug overlay on, watch the printed depth scores for objects at known rough distances, adjust thresholds to match
- `PRIORITY` — add/reorder object classes; anything not listed defaults to lowest priority
- `RE_ANNOUNCE_COOLDOWN_SECONDS` — how long before a still-present object gets mentioned again

## Next steps (once this is stable)

1. **OCR** — add EasyOCR to read signs/text on demand (e.g. triggered by a keypress or voice command), don't run every frame, it's slow.
2. **Voice commands** — "what's ahead", "read that sign" via a lightweight wake-word/command library instead of always-on listening.
3. **Metric depth** — if you want real meters later, that needs a stereo camera or depth sensor (e.g. Intel RealSense); monocular depth genuinely cannot give you that.
4. **Custom stairs/curb detection** — COCO (what YOLO is trained on) has no "stairs" class. If stairs/curbs matter for your use case, you'd need a small custom-trained model or a fine-tune — worth doing as a later phase, not now.
