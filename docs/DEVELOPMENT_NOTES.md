# Development Notes

## Current product scope

The repository is intentionally browser-first. The shipped application path is:

```text
web/ → ONNX Runtime Web → YOLOX-S MOT17 → ByteTrack JS
     → Stable ID → Privacy Hold → Pixelation → Canvas
```

The earlier FastAPI/MJPEG prototype, camera server, server-side pixel-art controls, and unrelated upstream deployment demos were removed to keep the repository focused.

## Core runtime decisions

- One inference may be active at a time.
- Slow inference causes frame skipping rather than queue growth.
- Source/session epochs invalidate stale asynchronous results.
- Pause preserves tracker/Stable ID state and starts no new inference.
- Seek invalidates in-flight work and resets tracking at the new media position.
- Automatic video loop resets tracking at the loop boundary.
- WebGPU is preferred; WASM is the fallback.
- WASM runs with one thread so GitHub Pages does not depend on cross-origin isolation headers.
- Local video input uses `blob:` URLs and stays in the browser.
- Stable ID is geometric short-term reassociation, not identity recognition.

## Default privacy parameters

```text
YOLOX test size : 544 × 960
Detector conf   : 0.05
NMS             : 0.7
ByteTrack       : 0.50 / 30 / 0.80
Padding         : 5%
Pixel scale     : 8%
Privacy hold    : 5 ticks
Stable ID gap   : 30
Stable ID IoU   : 0.15
Center distance : 120 px
```

## Supporting Python files

Only the support path remains:

- `tools/export_yolox_onnx.py`
- `tools/setup_ort_web.py`
- `tools/serve_web.py`
- `eval_wildtrack.py`
- the minimal YOLOX source required by export/evaluation

## Validation

Static/browser-module checks:

```powershell
python -m unittest tests.test_web -v
python -m unittest tests.test_docs -v
```

Real browser/model checks when local assets are available:

```powershell
python tests/smoke_web.py
python tests/smoke_loading.py
python tests/smoke_playback.py
python tests/smoke_privacy_fps.py
python tests/smoke_pixel_art.py
```

The GitHub Pages workflow publishes only `web/`. Deployment requires `web/models/yolox_s_mot17.onnx` to exist in the checked-out repository or be supplied by the deployment workflow.
