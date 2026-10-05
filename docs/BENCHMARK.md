# Benchmark Video Suite

This benchmark set uses **CC0** videos from Wikimedia Commons so the portfolio
can document the exact source and license of its public test media.

The videos are not committed to Git. Download them locally with:

```powershell
python tools/download_benchmark_videos.py --list
python tools/download_benchmark_videos.py
```

To download only the smaller baseline clip:

```powershell
python tools/download_benchmark_videos.py --only mumbai_street_cc0
```

## Test set

| ID | Scene | License | Resolution / duration | Intended test |
| --- | --- | --- | --- | --- |
| `mumbai_street_cc0` | Street in Mumbai | CC0 1.0 | 960×540 / 22 s | Baseline pedestrian detection, tracking continuity, Stable ID and privacy masking |
| `shengli_night_market_cc0` | Shengli night market, Taichung | CC0 1.0 | 1920×1080 / 91.333 s | Dense-crowd/night stress test, occlusion, distant people and renderer load |

Exact source pages, author names and filenames are stored in
`benchmarks/videos.json`.

## Browser acceptance procedure

1. Deploy/open the browser demo.
2. Load one benchmark video with **Load MP4 / WebM**.
3. Keep Privacy ON and verify detected people are masked.
4. Verify ByteTrack/Stable ID remain active while people move and overlap.
5. Exercise Pause, Play and seek.
6. Record the diagnostics below.
7. Confirm `image/frame uploads = 0`.

For each run record:

```text
Browser / OS:
Backend: WebGPU or WASM
Video:
Resolution:
Average video FPS:
Average detector FPS:
Average inference latency:
Peak people count:
Peak privacy regions:
Image/frame uploads:
Notes:
```

## Portfolio reporting

Do not present a single machine's FPS as a universal model benchmark. Report the
browser, device/backend and test video beside any performance figure.

A useful portfolio claim is:

> Browser-local person detection and tracking with YOLOX-S MOT17 + ByteTrack;
> raw frames remain in the tab and the network diagnostic reports zero
> image/frame uploads during the acceptance run.

This is an engineering demo, not a formal anonymity guarantee. Detector misses,
occlusion and runtime differences can reduce privacy coverage.

## Licensing note

CC0 addresses the copyright license of the media. It does not automatically
resolve every non-copyright issue that could apply to depicted people or
locations. Keep the source/license record with portfolio screenshots and do not
imply that people appearing in the test footage endorse the project.
