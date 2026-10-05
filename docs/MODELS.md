# Models and generated runtime files

## Browser model

The browser model is committed at:

```text
web/models/yolox_s_mot17.onnx
```

It is kept in the repository so the GitHub Pages demo can run without an
external model host. The adjacent metadata file
`web/models/yolox_s_mot17.json` records the checkpoint SHA-256, exported model
SHA-256, graph configuration, and validation metadata.

The model was exported from the ByteTrack-S MOT17 checkpoint used by this
project. The upstream ByteTrack model zoo publishes `bytetrack_s_mot17` at:

```text
https://github.com/FoundationVision/ByteTrack#model-zoo
```

To reproduce the export from a trusted local checkpoint:

```powershell
python tools/export_yolox_onnx.py
```

Default checkpoint path:

```text
pretrained/bytetrack_s_mot17.pth.tar
```

The export script:

1. loads `exps/example/mot/yolox_s_mix_det.py`;
2. loads and fuses the PyTorch model;
3. exports ONNX opset 17;
4. validates the graph with `onnx.checker`;
5. runs ONNX Runtime and checks numerical parity against PyTorch;
6. writes model metadata beside the ONNX file.

> `torch.load` deserializes Python objects. Only use checkpoints from a source you trust.

## ONNX Runtime Web

The browser expects runtime files under:

```text
web/vendor/onnxruntime/
```

They are downloaded during local setup and GitHub Pages builds with:

```powershell
python tools/setup_ort_web.py
```

The setup script pins ONNX Runtime Web version `1.23.2`, downloads the upstream
npm tarball, copies the required `.js` / `.mjs` / `.wasm` files, and also
retrieves upstream `LICENSE` and `ThirdPartyNotices.txt`.

These generated runtime files are ignored by Git because the Pages workflow
recreates them during deployment.
