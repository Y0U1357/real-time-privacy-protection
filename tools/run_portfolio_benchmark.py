"""Run one CC0 browser benchmark and capture portfolio-ready evidence.

This script deliberately measures the real browser app rather than a separate
Python inference path. It imports a locally downloaded benchmark video through
the page's file picker, samples the diagnostics exposed by the UI, verifies the
network guard, writes a JSON result, and captures a full-page PNG.

Examples:
    python tools/download_benchmark_videos.py --only mumbai_street_cc0
    python tools/setup_ort_web.py
    python tools/run_portfolio_benchmark.py --video-id mumbai_street_cc0
    python tools/run_portfolio_benchmark.py --video-id shengli_night_market_cc0 --seconds 20

Outputs:
    benchmarks/results/<video-id>.json
    docs/assets/<video-id>-demo.png

The JSON output is machine/browser specific. Do not present it as a universal
YOLOX benchmark.
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "videos.json"
RESULTS_DIR = ROOT / "benchmarks" / "results"
ASSETS_DIR = ROOT / "docs" / "assets"
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


def mean(values):
    return round(statistics.mean(values), 2) if values else None


def maximum(values):
    return max(values) if values else None


def evaluate_metrics(page):
    return page.evaluate(
        """(() => {
          const text = (selector) => {
            const el = document.querySelector(selector);
            return el ? el.textContent.trim() : "";
          };
          const number = (value) => {
            const parsed = Number(value);
            return Number.isFinite(parsed) ? parsed : null;
          };
          const detector = privacyDemo.detector;
          const network = privacyDemo.getNetworkReport();
          return {
            at_ms: performance.now(),
            video_fps: number(text("#fps")),
            people: number(text("#people")),
            detector_fps: (() => {
              const m = text("#diag-detector").match(/^([0-9.]+) FPS/);
              return m ? Number(m[1]) : null;
            })(),
            inference_ms: Number.isFinite(detector.inferenceMs)
              ? detector.inferenceMs : null,
            backend: detector.backend || null,
            tracks: privacyDemo.tracker.tracks.length,
            privacy_regions: privacyDemo.pixelator.regions,
            render_ms: Number.isFinite(privacyDemo.state.lastRenderMs)
              ? privacyDemo.state.lastRenderMs : null,
            image_uploads: network.imageUploads,
            requests: network.requests,
            source_time_s: document.getElementById("camera").currentTime || 0,
            resolution: text("#resolution")
          };
        })()"""
    )


def capture_full_page(page, target):
    result = page.call(
        "Page.captureScreenshot",
        {"format": "png", "captureBeyondViewport": True, "fromSurface": True},
        timeout=30,
        primitive=False,
    )
    data = result.get("data")
    if not data:
        raise RuntimeError("Chrome returned no screenshot data")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(base64.b64decode(data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", default="mumbai_street_cc0")
    parser.add_argument("--seconds", type=float, default=12.0,
                        help="seconds of live playback to sample")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="sampling interval in seconds")
    parser.add_argument("--no-screenshot", action="store_true")
    args = parser.parse_args()

    item = load_manifest(args.video_id)
    video_path = ROOT / "benchmarks" / "videos" / item["local_filename"]
    model_path = ROOT / "web" / "models" / "yolox_s_mot17.onnx"
    ort_path = ROOT / "web" / "vendor" / "onnxruntime" / "ort.webgpu.min.js"

    if not video_path.is_file():
        raise SystemExit(
            f"benchmark video missing: {video_path}\n"
            f"Run: python tools/download_benchmark_videos.py --only {item['id']}"
        )
    if not model_path.is_file():
        raise SystemExit(f"model missing: {model_path}")
    if not ort_path.is_file():
        raise SystemExit(
            "ONNX Runtime Web assets are missing. "
            "Run: python tools/setup_ort_web.py"
        )

    smoke = load_smoke_helpers()
    chrome = smoke.find_chrome()
    if chrome is None:
        raise SystemExit("Chrome/Edge not found; set CHROME=<browser path>")

    web_port = smoke.free_port()
    debug_port = smoke.free_port()
    server = smoke.start_server(web_port)
    profile = smoke.tempfile.mkdtemp(prefix="privacy-benchmark-")
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
        page.wait_for(
            "document.querySelector('#people').textContent",
            lambda value: isinstance(value, str) and value.isdigit(),
            timeout=60,
        )

        # Give the tracker and smoothed rates a brief warm-up period.
        time.sleep(2.0)

        samples = []
        deadline = time.time() + max(args.seconds, 1.0)
        screenshot_done = False
        screenshot_path = ASSETS_DIR / f"{item['id']}-demo.png"
        while time.time() < deadline:
            sample = evaluate_metrics(page)
            samples.append(sample)

            # Prefer a screenshot that visibly contains privacy regions.
            if (not args.no_screenshot and not screenshot_done
                    and (sample.get("privacy_regions") or 0) > 0):
                capture_full_page(page, screenshot_path)
                screenshot_done = True

            time.sleep(max(args.interval, 0.1))

        if not samples:
            raise RuntimeError("no benchmark samples were collected")

        uploads = [s["image_uploads"] for s in samples if s["image_uploads"] is not None]
        if any(value != 0 for value in uploads):
            raise RuntimeError(f"privacy guard failed: image/frame uploads = {uploads}")

        if not args.no_screenshot and not screenshot_done:
            capture_full_page(page, screenshot_path)
            screenshot_done = True

        def values(key):
            return [s[key] for s in samples if isinstance(s.get(key), (int, float))]

        backends = [s["backend"] for s in samples if s.get("backend")]
        report = {
            "schema_version": 1,
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "video": {
                "id": item["id"],
                "title": item["title"],
                "source_page": item["source_page"],
                "license": item["license"],
                "resolution": item["resolution"],
                "duration_seconds": item["duration_seconds"],
            },
            "environment": {
                "browser_binary": chrome,
                "browser_product": version.get("Browser"),
                "backend": backends[-1] if backends else None,
            },
            "sampling": {
                "requested_seconds": args.seconds,
                "interval_seconds": args.interval,
                "sample_count": len(samples),
            },
            "metrics": {
                "average_video_fps": mean(values("video_fps")),
                "average_detector_fps": mean(values("detector_fps")),
                "average_inference_ms": mean(values("inference_ms")),
                "average_render_ms": mean(values("render_ms")),
                "peak_people": maximum(values("people")),
                "peak_tracks": maximum(values("tracks")),
                "peak_privacy_regions": maximum(values("privacy_regions")),
                "image_frame_uploads": maximum(values("image_uploads")),
                "network_requests_observed": maximum(values("requests")),
            },
            "evidence": {
                "screenshot": str(screenshot_path.relative_to(ROOT)).replace("\\", "/")
                if screenshot_done else None,
            },
            "samples": samples,
        }

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        result_path = RESULTS_DIR / f"{item['id']}.json"
        result_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print("PASS: portfolio benchmark complete")
        print(f"  video      : {item['title']}")
        print(f"  backend    : {report['environment']['backend']}")
        print(f"  video FPS  : {report['metrics']['average_video_fps']}")
        print(f"  detector FPS: {report['metrics']['average_detector_fps']}")
        print(f"  inference  : {report['metrics']['average_inference_ms']} ms")
        print(f"  peak people: {report['metrics']['peak_people']}")
        print(f"  uploads    : {report['metrics']['image_frame_uploads']}")
        print(f"  result     : {result_path}")
        if screenshot_done:
            print(f"  screenshot : {screenshot_path}")
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
