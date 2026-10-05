# Browser model directory

The GitHub Pages demo uses the committed model:

```text
yolox_s_mot17.onnx
```

The adjacent `yolox_s_mot17.json` records the checkpoint and exported-model
SHA-256 values plus graph/validation metadata.

The upstream ByteTrack model zoo publishes the `bytetrack_s_mot17` checkpoint:

```text
https://github.com/FoundationVision/ByteTrack#model-zoo
```

To reproduce the ONNX export from a trusted local checkpoint:

```powershell
python tools/export_yolox_onnx.py
```

See `docs/MODELS.md` and `THIRD_PARTY.md` for details.
