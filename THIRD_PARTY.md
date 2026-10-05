# Third-party attribution

This repository contains or derives from third-party work. Keep the original copyright and license notices when redistributing the corresponding code.

## ByteTrack

The Python tracking code and parts of the project base derive from ByteTrack.
ByteTrack is distributed under the MIT License. The upstream license copy is retained at:

```text
third_party/ByteTrack-LICENSE.txt
```

Project/paper identifiers and upstream links are preserved throughout the derived source and documentation.

## YOLOX

Detector/training components under `yolox/`, experiment definitions under `exps/`, and the Python model export flow derive from YOLOX.

Upstream YOLOX is distributed under the Apache License 2.0. A copy is retained at:

```text
third_party/YOLOX-LICENSE.txt
```

Do not present the YOLOX detector or ByteTrack tracker as original algorithms of this project.

## Browser model checkpoint

The committed browser model `web/models/yolox_s_mot17.onnx` is an ONNX export of the ByteTrack-S MOT17 checkpoint used by this project. The upstream ByteTrack model zoo publishes `bytetrack_s_mot17`; the repository records the checkpoint and exported-model SHA-256 values in `web/models/yolox_s_mot17.json`.

Model source page:
`https://github.com/FoundationVision/ByteTrack#model-zoo`

The model is included so the GitHub Pages demo works without an external model host. Keep upstream attribution with redistributions.

## ONNX Runtime Web

The browser demo uses ONNX Runtime Web. Runtime binaries are **not committed** in this GitHub-ready package.

Run:

```powershell
python tools/setup_ort_web.py
```

The setup script downloads pinned ONNX Runtime Web assets and also retrieves upstream `LICENSE` and `ThirdPartyNotices.txt` into `web/vendor/onnxruntime/`.

## Image-to-Pixel

The Pixel Art privacy mode was inspired by / adapted from Image-to-Pixel behavior. The repository keeps the upstream attribution/license text at:

```text
third_party/Image-to-Pixel-LICENSE.txt
```

## Notes

This file summarizes attribution that can be confirmed from the repository. It is not legal advice and does not replace the license text shipped with each dependency.
