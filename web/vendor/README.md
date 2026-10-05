# Browser runtime directory

`web/vendor/onnxruntime/` is generated locally and is not committed.

From the repository root run:

```powershell
python tools/setup_ort_web.py
```

The setup script downloads the pinned ONNX Runtime Web build together with its upstream `LICENSE` and `ThirdPartyNotices.txt`.
