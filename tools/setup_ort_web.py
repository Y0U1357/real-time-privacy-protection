"""Download a pinned ONNX Runtime Web distribution for offline page operation."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

VERSION = "1.23.2"
DEST = Path(__file__).resolve().parents[1] / "web/vendor/onnxruntime"


def main():
    url = f"https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-{VERSION}.tgz"
    with urllib.request.urlopen(url, timeout=120) as response:
        archive = response.read()
    DEST.mkdir(parents=True, exist_ok=True)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = member.name.removeprefix("package/")
            if name in ("LICENSE", "ThirdPartyNotices.txt") or (name.startswith("dist/") and
                (name.endswith(".wasm") or name.endswith(".mjs") or name == "dist/ort.webgpu.min.js")):
                filename = Path(name).name
                content = tar.extractfile(member).read()
                (DEST / filename).write_bytes(content)
                files[filename] = hashlib.sha256(content).hexdigest()
    (DEST / "manifest.json").write_text(json.dumps({"version": VERSION, "source": url, "sha256": files}, indent=2))
    print(f"Downloaded ONNX Runtime Web {VERSION}: {len(files)} files to {DEST}")


if __name__ == "__main__":
    main()
