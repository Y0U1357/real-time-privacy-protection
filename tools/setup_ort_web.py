"""Download and verify a pinned ONNX Runtime Web distribution for offline use."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

VERSION = "1.23.2"
# npm dist.integrity for onnxruntime-web@1.23.2. Update this together with
# VERSION only after independently verifying the new published package.
EXPECTED_ARCHIVE_INTEGRITY = (
    "sha512-T09JUtMn+CZLk3mFwqiH0lgQf+4S7+oYHHtk6uhaYAAJI95bTcKi5bOOZYwORXfS/"
    "RLZCjDDEXGWIuOCAFlEjg=="
)
DEST = Path(__file__).resolve().parents[1] / "web/vendor/onnxruntime"


def archive_integrity(data):
    digest = base64.b64encode(hashlib.sha512(data).digest()).decode("ascii")
    return "sha512-" + digest


def verify_archive(data):
    actual = archive_integrity(data)
    if actual != EXPECTED_ARCHIVE_INTEGRITY:
        raise RuntimeError(
            "ONNX Runtime Web archive integrity check failed. "
            f"Expected {EXPECTED_ARCHIVE_INTEGRITY}, got {actual}."
        )
    return actual


def main():
    url = f"https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-{VERSION}.tgz"
    with urllib.request.urlopen(url, timeout=120) as response:
        archive = response.read()

    verified_integrity = verify_archive(archive)

    DEST.mkdir(parents=True, exist_ok=True)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = member.name.removeprefix("package/")
            if name in ("LICENSE", "ThirdPartyNotices.txt") or (
                name.startswith("dist/")
                and (
                    name.endswith(".wasm")
                    or name.endswith(".mjs")
                    or name == "dist/ort.webgpu.min.js"
                )
            ):
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"Could not extract required archive member: {name}")
                filename = Path(name).name
                content = extracted.read()
                (DEST / filename).write_bytes(content)
                files[filename] = hashlib.sha256(content).hexdigest()

    manifest = {
        "version": VERSION,
        "source": url,
        "archive_integrity": verified_integrity,
        "sha256": files,
    }
    (DEST / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Downloaded and verified ONNX Runtime Web {VERSION}: "
        f"{len(files)} files to {DEST}"
    )


if __name__ == "__main__":
    main()
