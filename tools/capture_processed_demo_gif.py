"""Capture a README-ready GIF from the real browser privacy pipeline.

The script drives the same browser app used by the portfolio benchmark, imports a
local CC0 benchmark video through the file picker, waits for YOLOX + ByteTrack +
privacy masking to be active, then samples the processed output canvas and
encodes those frames as a GIF.

Example:
    python -m pip install Pillow
    python tools/download_benchmark_videos.py --only shengli_night_market_cc0
    python tools/setup_ort_web.py
    python tools/capture_processed_demo_gif.py

Default output:
    docs/assets/shengli_night_market_cc0-processed-demo.gif
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "videos.json"
SMOKE_PATH = ROOT / "tests" / "smoke_web.py"


def load_smoke_helpers():
    spec = importlib.util.spec_from_file_location("privacy_smoke_helpers", SMOKE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SMOKE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_manifest(video_id):
    items = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for item in items:
        if item["id"] == video_id:
            return item
    known = ", ".join(item["id"] for item in items)
    raise SystemExit(f"unknown --video-id {video_id!r}; choose one of: {known}")


def decode_canvas_png(data_url, Image):
    prefix = "data:image/png;base64,"
    if not isinstance(data_url, str) or not data_url.startswith(prefix):
        raise RuntimeError("processed canvas did not return a PNG data URL")
    raw = base64.b64decode(data_url[len(prefix):])
    image = Image.open(io.BytesIO(raw))
    image.load()
    return image.convert("RGB")


def resize_frame(image, width, Image):
    if width <= 0 or image.width <= width:
        return image
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def main():
    parser = argparse.ArgumentParser(
        description="Capture an animated GIF from the real processed browser canvas.")
    parser.add_argument("--video-id", default="shengli_night_market_cc0")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.seconds <= 0:
        raise SystemExit("--seconds must be greater than 0")
    if not 1 <= args.fps <= 15:
        raise SystemExit("--fps must be between 1 and 15")
    if args.width < 160:
        raise SystemExit("--width must be at least 160 pixels")

    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Pillow is required to encode the GIF. Run: python -m pip install Pillow"
        ) from exc

    item = load_manifest(args.video_id)
    video_path = ROOT / "benchmarks" / "videos" / item["local_filename"]
    model_path = ROOT / "web" / "models" / "yolox_s_mot17.onnx"
    ort_path = ROOT / "web" / "vendor" / "onnxruntime" / "ort.webgpu.min.js"
    output_path = (
        Path(args.output)
        if args.output
        else ROOT / "docs" / "assets" / f"{item['id']}-processed-demo.gif"
    )
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    if not video_path.is_file():
        raise SystemExit(
            f"benchmark video missing: {video_path}\n"
            f"Run: python tools/download_benchmark_videos.py --only {item['id']}"
        )
    if not model_path.is_file():
        raise SystemExit(f"model missing: {model_path}")
    if not ort_path.is_file():
        raise SystemExit(
            "ONNX Runtime Web assets are missing. Run: python tools/setup_ort_web.py"
        )

    smoke = load_smoke_helpers()
    chrome = smoke.find_chrome()
    if chrome is None:
        raise SystemExit("Chrome/Edge not found; set CHROME=<browser path>")

    web_port = smoke.free_port()
    debug_port = smoke.free_port()
    server = smoke.start_server(web_port)
    profile = smoke.tempfile.mkdtemp(prefix="privacy-demo-gif-")
    process = None
    page = None

    try:
        process, version = smoke.start_browser(chrome, profile, debug_port)
        page = smoke.new_page(debug_port, f"http://127.0.0.1:{web_port}/index.html")
        page.call("Runtime.enable")
        page.call("Page.enable")

        page.wait_for(
            "document.querySelector('#model').textContent",
            lambda value: value == "YOLOX-S MOT17",
            timeout=120,
        )

        page.call("DOM.enable")
        document = page.call("DOM.getDocument", {"depth": 1}, primitive=False)
        node = page.call(
            "DOM.querySelector",
            {"nodeId": document["root"]["nodeId"], "selector": "#video-input"},
            primitive=False,
        )
        page.call(
            "DOM.setFileInputFiles",
            {"files": [str(video_path)], "nodeId": node["nodeId"]},
        )
        page.evaluate(
            "document.querySelector('#video-input')"
            ".dispatchEvent(new Event('change', {bubbles: true}))"
        )

        page.wait_for(
            "document.querySelector('#camera-state').textContent",
            lambda value: value is not None and "Video" in value,
            timeout=60,
        )
        page.wait_for(
            "privacyDemo.detector.inferenceCount",
            lambda value: isinstance(value, int) and value >= 2,
            timeout=120,
        )

        # Fresh page defaults should already enable these. Set them explicitly so
        # a future UI-default change cannot accidentally create an unprotected demo.
        page.evaluate("""(() => {
          const privacy = document.getElementById('privacy-button');
          const stable = document.getElementById('stable-button');
          const overlay = document.getElementById('overlay-button');
          const comparison = document.getElementById('comparison-button');
          if (!privacyDemo.CONFIG.privacy.enabled) privacy.click();
          if (!privacyDemo.stableIds.enabled) stable.click();
          if (!privacyDemo.state.showOverlay) overlay.click();
          if (privacyDemo.state.comparison) comparison.click();
        })()""")

        # Do not begin the portfolio clip until the browser has actually rendered
        # at least one privacy region from the detector/tracker path.
        page.wait_for(
            "privacyDemo.pixelator.regions",
            lambda value: isinstance(value, (int, float)) and value > 0,
            timeout=120,
        )
        time.sleep(0.5)

        frame_count = max(1, round(args.seconds * args.fps))
        interval = 1.0 / args.fps
        frames = []
        next_capture = time.perf_counter()

        for _ in range(frame_count):
            now = time.perf_counter()
            if now < next_capture:
                time.sleep(next_capture - now)

            data_url = page.evaluate(
                "document.getElementById('canvas').toDataURL('image/png')",
                timeout=30,
            )
            frame = decode_canvas_png(data_url, Image)
            frames.append(resize_frame(frame, args.width, Image))
            next_capture += interval

        uploads = page.evaluate("privacyDemo.getNetworkReport().imageUploads")
        if uploads != 0:
            raise RuntimeError(
                f"network guard reported {uploads} image/frame uploads during capture"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        duration_ms = max(20, round(1000 / args.fps))

        # Let Pillow create per-frame palettes. This preserves the visible
        # pixelation/overlay while keeping a short README clip reasonably small.
        gif_frames = [
            frame.quantize(
                colors=128,
                method=Image.Quantize.MEDIANCUT,
                dither=Image.Dither.FLOYDSTEINBERG,
            )
            for frame in frames
        ]
        gif_frames[0].save(
            output_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
            disposal=2,
        )

        relative = output_path.relative_to(ROOT) if output_path.is_relative_to(ROOT) else output_path
        print("PASS: processed browser demo GIF created")
        print(f"  video   : {item['title']}")
        print(f"  browser : {version.get('Browser')}")
        print(f"  frames  : {len(gif_frames)} @ {args.fps:g} FPS")
        print(f"  size    : {gif_frames[0].width}x{gif_frames[0].height}")
        print(f"  uploads : {uploads}")
        print(f"  output  : {relative}")
        return 0
    finally:
        if page is not None:
            try:
                page.close()
            except OSError:
                pass
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except Exception:
                process.kill()
        server.shutdown()
        server.server_close()
        smoke.shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
