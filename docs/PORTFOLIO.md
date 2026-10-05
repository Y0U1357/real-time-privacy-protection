# Portfolio Presentation

## Project title

**Real-Time Privacy Protection System — Browser-Local Computer Vision**

## One-line summary

A browser-local privacy pipeline that detects and tracks people with
YOLOX-S MOT17 + ByteTrack, assigns short-term Stable IDs, and pixelates person
regions without uploading raw camera/video frames.

## What to show first

1. Open the live GitHub Pages demo.
2. Load a documented CC0 benchmark video.
3. Keep **Privacy ON**, **Stable ID ON**, and **Boxes & IDs ON**.
4. Show the Diagnostics panel with the inference backend and
   **0 image/frame uploads**.
5. Toggle Comparison briefly to show raw vs processed output.
6. Show Pause / seek / Play to demonstrate local-file handling.
7. Return to the processed view before taking the portfolio screenshot.

## Recommended screenshots

Generate them with the real browser pipeline:

```powershell
python tools/download_benchmark_videos.py --only mumbai_street_cc0
python tools/setup_ort_web.py
python tools/run_portfolio_benchmark.py --video-id mumbai_street_cc0
```

The script writes:

```text
docs/assets/mumbai_street_cc0-demo.png
benchmarks/results/mumbai_street_cc0.json
```

For the harder night/crowd case:

```powershell
python tools/download_benchmark_videos.py --only shengli_night_market_cc0
python tools/run_portfolio_benchmark.py --video-id shengli_night_market_cc0 --seconds 20
```

## Benchmark result format

Only publish values produced by an actual run. Record the browser/backend beside
performance numbers because WebGPU/WASM and hardware can change the result
substantially.

| Scene | Backend | Video FPS | Detector FPS | Inference | Peak people | Frame uploads |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mumbai street | WebGPU | 22.74 | 11.20 | 43.76 ms | 4 | 0 |
| Shengli night market | WebGPU | 25.61 | 12.27 | 45.34 ms | 17 | 0 |

These values come from actual local Windows browser runs. They are suitable as
portfolio evidence for this machine, but should not be presented as universal
model throughput. The night-market clip is the denser stress case and reached
17 simultaneously active people during the sampled run.

## Suggested portfolio description

### Traditional Chinese

這是一套完全在瀏覽器本機執行的即時人物隱私保護系統。系統使用
YOLOX-S MOT17 進行人物偵測、ByteTrack 進行多人追蹤，再以短期
Stable ID、Privacy Hold 與 5% 邊界擴張降低短暫漏偵測造成的隱私缺口。
推論透過 ONNX Runtime Web 執行，優先使用 WebGPU，並提供 WASM
fallback。Webcam 與本機影片影格皆在目前分頁內處理，診斷介面會監控
網路請求，正常驗收流程應維持 0 image/frame uploads。

### English

A browser-local real-time privacy system using YOLOX-S MOT17 for person
detection, ByteTrack for multi-object tracking, short-term Stable ID
reassociation, privacy hold, and configurable pixelation. ONNX Runtime Web uses
WebGPU when available with a WASM fallback. Camera and local-video frames are
processed inside the current tab, while a network guard exposes unexpected
image/frame uploads in the diagnostics UI.

## Claims to avoid

- Do not call Stable ID biometric identity recognition or person ReID.
- Do not claim guaranteed anonymity.
- Do not compare FPS across machines without naming the browser/backend/device.
- Do not describe CC0 as meaning that every possible non-copyright right is
  automatically cleared.
