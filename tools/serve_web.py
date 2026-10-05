"""Static server for the browser demo with the MIME types ONNX Runtime needs.

`python -m http.server` answers *.mjs with `text/plain`, and Chrome then refuses
to evaluate the ONNX Runtime WASM loader module ("Expected a JavaScript module
script"), so every session creation fails with `initWasm() failed`. This server
only changes the content types the demo requires and nothing else.

Run:  py tools/serve_web.py --port 8080
Then: http://localhost:8080/   (Ctrl+C to stop, camera released on exit)
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"

# Chromium treats only JavaScript MIME types as executable modules.
MIME_TYPES = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".wasm": "application/wasm",
    ".onnx": "application/octet-stream",
}


class DemoHandler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, **MIME_TYPES}

    def log_message(self, *args):  # keep the console readable
        pass


def main():
    parser = argparse.ArgumentParser(description="Serve the browser-local demo.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    handler = partial(DemoHandler, directory=str(WEB))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"web demo: http://{args.host}:{args.port}/   (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
