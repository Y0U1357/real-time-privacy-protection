"""Headless end-to-end smoke test for the browser demo in web/.

Run:  py tests/smoke_web.py

Starts a local static server for web/, launches Chrome/Edge headless with a
fake camera, and drives the page over the DevTools protocol in real time:

  1. the local YOLOX-S MOT17 ONNX model loads from ./models,
  2. Start Camera opens a MediaStream and the render loop runs,
  3. the diagnostics network counter stays at 0 image/frame uploads,
  4. Privacy / Stable ID / overlay / comparison / tuning controls respond,
  5. a local MP4 can be imported and is processed like a camera feed
     (people are detected and pixelated, still 0 uploads),
  6. Stop Camera releases the source.

Skips (exit 0) when no Chrome/Edge binary is found, so it is safe to wire into
any test run. Stdlib only: no websocket or webdriver dependency.
"""
import base64
import json
import os
import shutil
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"

CHROME_CANDIDATES = (
    os.environ.get("CHROME", ""),
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)


def find_chrome():
    for candidate in CHROME_CANDIDATES:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class QuietHandler(SimpleHTTPRequestHandler):
    # Chromium refuses a *.mjs served as text/plain, which would break the ORT
    # WASM loader; tools/serve_web.py sets the same types for manual runs.
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".mjs": "text/javascript",
                      ".wasm": "application/wasm"}

    def log_message(self, *args):  # keep test output clean
        pass


def start_server(port):
    handler = partial(QuietHandler, directory=str(WEB))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server

class WebSocket:
    """Minimal RFC 6455 client: text frames in, text frames out."""

    def __init__(self, url):
        assert url.startswith("ws://"), url
        hostport, _, path = url[5:].partition("/")
        host, _, port = hostport.partition(":")
        self.sock = socket.create_connection((host, int(port or 80)), timeout=30)
        self.buffer = b""
        key = base64.b64encode(os.urandom(16)).decode()
        request = (
            f"GET /{path} HTTP/1.1\r\nHost: {hostport}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(request.encode())
        while b"\r\n\r\n" not in self.buffer:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise RuntimeError("websocket handshake failed")
            self.buffer += chunk
        head, _, rest = self.buffer.partition(b"\r\n\r\n")
        if b"101" not in head.split(b"\r\n")[0]:
            raise RuntimeError(f"websocket handshake rejected: {head!r}")
        self.buffer = rest

    def _read(self, size):
        while len(self.buffer) < size:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("websocket closed")
            self.buffer += chunk
        data, self.buffer = self.buffer[:size], self.buffer[size:]
        return data

    def send_text(self, text):
        payload = text.encode()
        header = bytearray([0x81])  # FIN + text opcode
        mask = os.urandom(4)
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header += struct.pack(">H", length)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", length)
        header += mask
        masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        self.sock.sendall(bytes(header) + masked)

    def recv_text(self):
        while True:
            first, second = self._read(2)
            opcode = first & 0x0F
            length = second & 0x7F
            if length == 126:
                length = struct.unpack(">H", self._read(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", self._read(8))[0]
            payload = self._read(length) if length else b""
            if opcode == 0x9:  # ping -> pong
                mask = os.urandom(4)
                self.sock.sendall(bytes([0x8A, 0x80 | len(payload)]) + mask +
                                  bytes(byte ^ mask[i % 4]
                                        for i, byte in enumerate(payload)))
                continue
            if opcode == 0x8:
                raise ConnectionError("websocket closed by Chrome")
            if opcode in (0x1, 0x2):
                return payload.decode(errors="replace")

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class Page:
    """One CDP page target with a synchronous evaluate()."""

    def __init__(self, ws_url):
        self.ws = WebSocket(ws_url)
        self.next_id = 0
        self.errors = []
        self.console = []

    def call(self, method, params=None, timeout=60, primitive=True):
        self.next_id += 1
        wanted = self.next_id
        self.ws.send_text(json.dumps({"id": wanted, "method": method,
                                      "params": params or {}}))
        deadline = time.time() + timeout
        while time.time() < deadline:
            message = json.loads(self.ws.recv_text())
            if message.get("method") == "Runtime.exceptionThrown":
                details = message["params"]["exceptionDetails"]
                exception = details.get("exception", {})
                self.errors.append(
                    f"{details.get('text', '')} {exception.get('description', '')}".strip())
            elif message.get("method") == "Runtime.consoleAPICalled":
                entry = message["params"]
                if entry.get("type") in ("error", "warning"):
                    args = " ".join(str(a.get("value", a.get("description", "")))
                                    for a in entry.get("args", []))
                    self.console.append(f"{entry['type']}: {args}")
            elif message.get("id") == wanted:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                result = message.get("result", {})
                if "exceptionDetails" in result:
                    raise RuntimeError(f"{method} raised: {result['exceptionDetails']}")
                payload = result.get("result", {})
                # primitive=False returns the whole CDP result object, which is
                # what DOM.getDocument / DOM.querySelector reply with.
                return payload.get("value") if primitive else result
        raise TimeoutError(f"{method} timed out after {timeout}s")

    def evaluate(self, expression, timeout=60):
        return self.call("Runtime.evaluate",
                         {"expression": expression, "returnByValue": True,
                          "awaitPromise": True}, timeout=timeout)

    def wait_for(self, expression, predicate, timeout=60, interval=0.25):
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            try:
                last = self.evaluate(expression, timeout=10)
            except (TimeoutError, ConnectionError, RuntimeError):
                last = None
            if predicate(last):
                return last
            time.sleep(interval)
        raise AssertionError(
            f"condition not met in {timeout}s: {expression!r} -> last value {last!r}")

    def close(self):
        self.ws.close()


def text(page, selector):
    return page.evaluate(
        "(() => { const el = document.querySelector(%s); "
        "return el ? el.textContent.trim() : null; })()" % json.dumps(selector))


def click(page, selector):
    return page.evaluate(
        "(() => { const el = document.querySelector(%s); if (!el) return 'missing';"
        " el.click(); return 'clicked'; })()" % json.dumps(selector))


def json_get(url, method="GET"):
    request = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode())


def start_browser(chrome, profile, port):
    command = [
        chrome, "--headless=new", "--no-first-run", "--no-default-browser-check",
        "--remote-allow-origins=*", f"--user-data-dir={profile}",
        f"--remote-debugging-port={port}",
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        "--autoplay-policy=no-user-gesture-required",
        "about:blank",
    ]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            version = json_get(f"http://127.0.0.1:{port}/json/version")
            return process, version
        except Exception:
            if process.poll() is not None:
                raise RuntimeError(f"Chrome exited early with {process.returncode}")
            time.sleep(0.3)
    raise TimeoutError("Chrome DevTools endpoint never came up")


def new_page(port, url):
    target = None
    for method in ("PUT", "GET"):
        try:
            target = json_get(f"http://127.0.0.1:{port}/json/new?{url}", method=method)
            break
        except Exception:
            continue
    if target is None:
        raise RuntimeError("could not open a page target")
    return Page(target["webSocketDebuggerUrl"])


def run_checks(page):
    page.call("Runtime.enable")
    page.call("Page.enable")

    # 1. Local YOLOX model loads from ./models without any upload.
    model = page.wait_for("document.querySelector('#model').textContent",
                          lambda v: v == "YOLOX-S MOT17",
                          timeout=90)
    print(f"PASS: model loaded -> {model!r}")
    assert text(page, "#error") in ("", None), text(page, "#error")

    # 2. Start the camera and wait for the live pipeline.
    assert click(page, "#start-button") == "clicked"
    state = page.wait_for("document.querySelector('#camera-state').textContent",
                          lambda v: v is not None and "Live" in v, timeout=45)
    print(f"PASS: camera -> {state!r}")
    resolution = page.wait_for(
        "document.querySelector('#resolution').textContent",
        lambda v: isinstance(v, str) and v not in ("—", "0x0") and
        v.split("x")[0].isdigit() and int(v.split("x")[0]) > 0, timeout=30)
    print(f"PASS: resolution -> {resolution!r}")
    tracking = page.wait_for("document.querySelector('#tracking-state').textContent",
                             lambda v: v is not None and v.startswith("Active"), timeout=30)
    print(f"PASS: tracking -> {tracking!r}")

    # 2b. Start again while live: no second MediaStream and no second loop.
    click(page, "#start-button")
    time.sleep(1.5)
    assert page.evaluate("document.querySelector('#start-button').disabled") is True
    assert page.evaluate("privacyDemo.camera.hasLiveTrack") is True
    assert page.evaluate("privacyDemo.camera.stream.getVideoTracks().length") == 1
    print("PASS: repeated Start is a no-op (one stream, one loop)")

    # 3. Detections and render metrics are being produced.
    count = page.wait_for("window.privacyDemo ? privacyDemo.detector.inferenceCount : 0",
                          lambda v: isinstance(v, int) and v >= 2, timeout=60)
    print(f"PASS: detector ran {count} inferences")
    people = page.wait_for("document.querySelector('#people').textContent",
                           lambda v: v is not None and v.isdigit(), timeout=30)
    fps = page.wait_for("document.querySelector('#fps').textContent",
                        lambda v: v is not None and v not in ("—",) and
                        v.replace(".", "").isdigit(), timeout=30)
    print(f"PASS: metrics people={people} fps={fps}")

    # 4. Local-only guarantee: no image/frame upload since page load.
    network = page.wait_for("document.querySelector('#diag-network').textContent",
                            lambda v: v is not None and "image/frame uploads" in v)
    assert "0 image/frame uploads" in network, network
    uploads = page.evaluate("privacyDemo.getNetworkReport().imageUploads")
    assert uploads == 0, uploads
    print(f"PASS: network guard -> {network!r}")

    # 5. Toggles respond and update both button and metric readouts.
    click(page, "#privacy-button")
    page.wait_for("document.querySelector('#privacy-state').textContent",
                  lambda v: v == "OFF", timeout=15)
    click(page, "#privacy-button")
    page.wait_for("document.querySelector('#privacy-state').textContent",
                  lambda v: v == "ON", timeout=15)
    click(page, "#stable-button")
    page.wait_for("document.querySelector('#stable-state').textContent",
                  lambda v: v == "OFF", timeout=15)
    click(page, "#stable-button")
    page.wait_for("document.querySelector('#stable-state').textContent",
                  lambda v: v == "ON", timeout=15)
    click(page, "#comparison-button")
    page.wait_for("document.querySelector('#raw-wrap').hidden", lambda v: v is False,
                  timeout=15)
    click(page, "#comparison-button")
    page.wait_for("document.querySelector('#raw-wrap').hidden", lambda v: v is True,
                  timeout=15)
    print("PASS: privacy / stable-id / comparison toggles")


    # 6. Live tuning: range inputs write CONFIG and their output labels.
    padding = page.evaluate("""(() => {
      const el = document.querySelector('#padding-input');
      el.value = '0.2';
      el.dispatchEvent(new Event('input', {bubbles: true}));
      return {config: privacyDemo.CONFIG.privacy.padding,
              label: document.querySelector('#padding-value').textContent};
    })()""")
    assert abs(padding["config"] - 0.2) < 1e-9 and padding["label"] == "20%", padding
    hold = page.evaluate("""(() => {
      const el = document.querySelector('#hold-input');
      el.value = '12';
      el.dispatchEvent(new Event('input', {bubbles: true}));
      return privacyDemo.CONFIG.privacy.holdFrames;
    })()""")
    assert hold == 12, hold
    style = page.evaluate("""(() => {
      const el = document.querySelector('#style-select');
      el.value = 'retro';
      el.dispatchEvent(new Event('change', {bubbles: true}));
      return privacyDemo.CONFIG.privacy.style;
    })()""")
    assert style == "retro", style
    click(page, "#tuning-reset")
    defaults = page.evaluate("({padding: privacyDemo.CONFIG.privacy.padding, "
                             "hold: privacyDemo.CONFIG.privacy.holdFrames, "
                             "style: privacyDemo.CONFIG.privacy.style})")
    assert defaults == {"padding": 0.05, "hold": 5, "style": "block"}, defaults
    print(f"PASS: tuning panel writes and resets CONFIG {defaults}")

    # 7. Import a local video file: same pipeline, and the file never leaves
    #    the machine (blob: URL, zero uploads).
    video_path = ROOT / "videos" / "palace.mp4"
    if video_path.is_file():
        page.call("DOM.enable")
        document = page.call("DOM.getDocument", {"depth": 1}, primitive=False)
        node = page.call("DOM.querySelector",
                         {"nodeId": document["root"]["nodeId"],
                          "selector": "#video-input"}, primitive=False)
        page.call("DOM.setFileInputFiles",
                  {"files": [str(video_path)], "nodeId": node["nodeId"]})
        # Chrome fires the events itself; dispatch once more so the app reacts
        # even if a future build skips them.
        page.evaluate("document.querySelector('#video-input')"
                      ".dispatchEvent(new Event('change', {bubbles: true}))")
        source = page.wait_for("document.querySelector('#camera-state').textContent",
                               lambda v: v is not None and "Video" in v, timeout=45)
        print(f"PASS: imported video is the source -> {source!r}")
        label = text(page, "#video-name")
        assert "palace.mp4" in (label or ""), label
        people = page.wait_for("document.querySelector('#people').textContent",
                               lambda v: v is not None and v.isdigit() and int(v) > 0,
                               timeout=60)
        regions = page.wait_for("document.querySelector('#diag-render').textContent",
                                lambda v: v is not None and "0 regions" not in v,
                                timeout=30)
        uploads = page.evaluate("privacyDemo.getNetworkReport().imageUploads")
        assert uploads == 0, uploads
        assert page.evaluate("privacyDemo.camera.hasLiveTrack") is True
        print(f"PASS: {people} people detected + pixelated in the imported video -> "
              f"{regions!r}, {uploads} uploads")

        # 7a. Retro must still PAINT. It batches every region into one canvas
        #      read/write per frame, so a bug there would blank the pixelation
        #      instead of just slowing it down - check real canvas pixels, and
        #      that the block path stays fast again straight after switching back.
        def region_colours(style):
            page.evaluate(
                "(() => { const el = document.querySelector('#style-select');"
                f" el.value = '{style}';"
                " el.dispatchEvent(new Event('change', {bubbles: true})); })()")
            page.wait_for("privacyDemo.pixelator.regions",
                          lambda n: isinstance(n, (int, float)) and n > 3,
                          timeout=30)
            return page.evaluate("""(() => {
              const canvas = document.getElementById('canvas');
              const track = privacyDemo.tracker.tracks.find(
                (t) => t.confirmed && t.valid && t.missingTicks === 0);
              if (!track) return null;
              const box = track.box;
              // Sample a grid inside the padded person box.
              const seen = new Set();
              for (let y = box.y1 + 8; y < box.y2 - 8; y += 9) {
                for (let x = box.x1 + 8; x < box.x2 - 8; x += 9) {
                  const d = canvas.getContext('2d')
                    .getImageData(Math.round(x), Math.round(y), 1, 1).data;
                  seen.add(d[0] + ',' + d[1] + ',' + d[2]);
                }
              }
              return seen.size;
            })()""")

        block_colours = region_colours("block")
        retro_colours = region_colours("retro")
        assert block_colours and block_colours > 3, block_colours
        assert retro_colours and retro_colours > 1, retro_colours
        # Retro snaps every pixel to one of 8 palette colours, so it cannot have
        # more distinct colours than block; it must also not collapse to one.
        assert retro_colours <= block_colours, (retro_colours, block_colours)
        print(f"PASS: retro still paints ({retro_colours} colours inside the "
              f"person box vs {block_colours} for block)")

        page.evaluate("""(() => {
          window.__renderSamples = [];
          const collect = () => {
            window.__renderSamples.push(privacyDemo.state.lastRenderMs);
            if (window.__renderSamples.length < 40) requestAnimationFrame(collect);
          };
          requestAnimationFrame(collect);
        })()""")
        region_colours("block")
        page.wait_for("window.__renderSamples.length",
                      lambda n: isinstance(n, (int, float)) and n >= 40,
                      timeout=20)
        recovered = page.evaluate("window.__renderSamples.slice(-20)")
        render_ms = statistics.mean(recovered)
        # Block was ~1ms/frame with 15+ regions; retro used to leave it at 90ms+
        # for the rest of the session. This is the regression guard.
        assert render_ms < 20, render_ms
        print(f"PASS: switching retro -> block recovers immediately "
              f"({render_ms:.1f} ms/frame after the switch)")

        # The clip must actually advance in real time, and the reported FPS must
        # describe the clip rather than the render loop spinning on rAF.
        def playback():
            return page.evaluate(
                "({t: document.getElementById('camera').currentTime,"
                " frames: privacyDemo.camera.presentedFrames,"
                " fps: parseFloat(document.querySelector('#fps').textContent)})")

        first = playback()
        time.sleep(2.0)
        second = playback()
        advanced = second["t"] - first["t"]
        # Allow a wide band: headless timing is noisy, but the clip cannot stand
        # still and cannot run away from real time either.
        assert 0.8 < advanced < 3.5, advanced
        assert second["frames"] > first["frames"], (first, second)
        # A 30 FPS clip must never be reported as the 100+ FPS rAF rate.
        assert second["fps"] < 90, second["fps"]
        print(f"PASS: clip plays in real time ({advanced:.2f}s of video in 2.0s), "
              f"presented {second['frames'] - first['frames']} frames, "
              f"FPS {second['fps']} (not the rAF rate)")

        # A picked file plays with its own audio; the element must not be
        # force-muted. Headless has no audio device, so an autoplay refusal is
        # legitimate there - but then it must be reported, not silent.
        muted = page.evaluate("document.getElementById('camera').muted")
        blocked = page.evaluate("privacyDemo.camera.audioBlocked")
        viewer = text(page, "#viewer-message")
        if muted:
            assert blocked is True, (muted, blocked)
            assert "muted" in (viewer or ""), viewer
            print("PASS: autoplay policy muted the clip and the UI says so")
        else:
            assert blocked is False, blocked
            print("PASS: imported clip plays with its own audio")

        # 7b. Playback bar: pause freezes the clip, the timeline seeks it and
        #     play resumes - while the pipeline keeps running on that source.
        assert page.evaluate("document.getElementById('video-playback-controls').hidden") is False
        assert click(page, "#video-pause-button") == "clicked"
        page.wait_for("document.getElementById('camera').paused",
                      lambda v: v is True, timeout=10)
        page.wait_for("document.querySelector('#tracking-state').textContent",
                      lambda v: v == "Paused", timeout=10)
        paused_label = text(page, "#video-pause-button")
        assert paused_label and "Play" in paused_label, paused_label
        # A frozen clip must not advertise a render-loop FPS either.
        assert page.evaluate("document.querySelector('#fps').textContent") == "0.0"
        frozen = page.evaluate("document.getElementById('camera').currentTime")
        time.sleep(1.0)
        still = page.evaluate("document.getElementById('camera').currentTime")
        assert abs(still - frozen) < 0.2, (frozen, still)
        # Seek to the middle of the clip through the timeline input event.
        target = page.evaluate("""(() => {
          const tl = document.getElementById('video-timeline');
          const mid = Number(tl.max) / 2;
          tl.value = String(mid);
          tl.dispatchEvent(new Event('input', {bubbles: true}));
          tl.dispatchEvent(new Event('change', {bubbles: true}));
          return mid;
        })()""")
        page.wait_for("document.getElementById('camera').currentTime",
                      lambda v: v is not None and abs(v - target) < 1.0, timeout=10)
        assert page.evaluate("document.getElementById('camera').paused") is True
        clock = (text(page, "#video-current-time") or "") + " / " + (text(page, "#video-duration") or "")
        assert "/" in clock, clock
        # Resume: the playhead moves again and the controls flip back.
        page.wait_for("!privacyDemo.state.seeking && !document.getElementById('camera').seeking",
                      bool, timeout=10)
        assert click(page, "#video-pause-button") == "clicked"
        page.wait_for("document.getElementById('camera').paused",
                      lambda v: v is False, timeout=10)
        resumed = page.evaluate("document.getElementById('camera').currentTime")
        time.sleep(1.0)
        advanced = page.evaluate("document.getElementById('camera').currentTime")
        assert advanced > resumed, (resumed, advanced)
        print(f"PASS: pause / timeline seek / resume (clock {clock!r})")
    else:
        print(f"SKIP: {video_path} missing; video import not exercised")

    # 8. Stop releases the source and the UI says so.
    assert click(page, "#stop-button") == "clicked"
    stopped = page.wait_for("document.querySelector('#camera-state').textContent",
                            lambda v: v == "Camera off", timeout=15)
    message = page.wait_for("document.querySelector('#viewer-message').textContent",
                            lambda v: v is not None and "stopped" in v, timeout=15)
    live = page.evaluate("privacyDemo.camera.hasLiveTrack")
    assert live is False, live
    print(f"PASS: camera stopped -> {stopped!r}, {message!r}")

    # 8. No uncaught exceptions and no console errors during the whole run.
    assert not page.errors, page.errors
    # ORT logs its own internal warnings through console.error; only page-level
    # failures should fail the acceptance run.
    console_errors = [line for line in page.console
                      if line.startswith("error:") and "onnxruntime" not in line.lower()]
    assert not console_errors, console_errors
    print("PASS: no page exceptions, no console errors")


def main():
    chrome = find_chrome()
    if chrome is None:
        print("SKIP: no Chrome/Edge found; set CHROME=<path> to run this test")
        return 0
    web_port = free_port()
    debug_port = free_port()
    server = start_server(web_port)
    profile = tempfile.mkdtemp(prefix="privacy-web-")
    process = None
    page = None
    try:
        process, _ = start_browser(chrome, profile, debug_port)
        page = new_page(debug_port, f"http://127.0.0.1:{web_port}/index.html")
        run_checks(page)
        print("PASS: web smoke test complete")
        page.close()
        page = None
        model_load_failure_recovery(chrome)
        print("PASS: model load failure / recovery checks complete")
        return 0
    except Exception as error:  # report the failure and exit non-zero
        print(f"FAIL: {error}")
        traceback.print_exc()
        return 1
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
        shutil.rmtree(profile, ignore_errors=True)


def model_load_failure_recovery(chrome):
    """A model that never arrives must not leave the page loading forever.

    This is the regression guard for three separate bugs:
      * an unbounded await on fetch() left the UI on "loading" with no error;
      * the rejected load promise was cached, so no retry could ever succeed;
      * Start Camera stayed enabled and silently did nothing without a model.

    Both a stalled response and a hard failure are exercised, and each must end
    with a reported error plus a working "Retry Model" button.
    """
    for scenario, stalled in (("stalled", True), ("failed", False)):
        web_port = free_port()
        debug_port = free_port()
        server = start_server(web_port)
        profile = tempfile.mkdtemp(prefix="privacy-model-fail-")
        process = None
        page = None
        try:
            process, _ = start_browser(chrome, profile, debug_port)
            page = new_page(debug_port, "about:blank")
            page.call("Runtime.enable")
            page.call("Page.enable")
            if stalled:
                # 4s deadline keeps the test quick while still proving the
                # timeout path fires instead of waiting out the real 30s.
                page.call("Page.addScriptToEvaluateOnNewDocument", {"source":
                    "window.__PROBE_TIMEOUT = 4000;"})
            # Intercept only the model request so the page itself still loads.
            page.call("Fetch.enable", {"patterns": [
                {"urlPattern": "*yolox_s_mot17.onnx", "requestStage": "Request"}]})
            page.call("Page.navigate", {"url": f"http://127.0.0.1:{web_port}/index.html"})
            page.ws.sock.settimeout(30)
            deadline = time.time() + 30
            while time.time() < deadline:
                message = json.loads(page.ws.recv_text())
                if message.get("method") == "Fetch.requestPaused":
                    request_id = message["params"]["requestId"]
                    # stalled: never answer, reproducing a server that accepts
                    # the connection and then says nothing at all.
                    if not stalled:
                        page.call("Fetch.failRequest", {"requestId": request_id,
                                                        "errorReason": "Failed"})
                    break
            time.sleep(10 if stalled else 6)

            failed = page.wait_for(
                "document.querySelector('#model').textContent",
                lambda v: v == "Model failed to load", timeout=45)
            assert failed == "Model failed to load", failed
            error = text(page, "#error") or ""
            assert "could not be loaded" in error, error
            assert not page.evaluate("document.querySelector('#error').hidden")
            # Start must not pretend it can work without a detector.
            assert page.evaluate("document.querySelector('#start-button').disabled") is True
            # A failed attempt must not be cached, or retry could never recover.
            assert page.evaluate("privacyDemo.detector._loading") is None
            print(f"PASS: {scenario} model load reports an error instead of "
                  f"loading forever -> {error[:60]!r}")

            # Recovery must work through the UI, with no page reload.
            page.call("Fetch.disable")
            recovered = page.evaluate("""
(async () => {
  const button = document.getElementById('model-retry');
  const wasVisible = !button.hidden;
  button.click();
  const started = performance.now();
  while (performance.now() - started < 90000) {
    if (privacyDemo.detector.loaded) break;
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  return {wasVisible, loaded: privacyDemo.detector.loaded,
          retryHidden: button.hidden,
          startDisabled: document.getElementById('start-button').disabled};
})()""")
            assert recovered["wasVisible"] is True, recovered
            assert recovered["loaded"] is True, recovered
            assert recovered["retryHidden"] is True, recovered
            assert recovered["startDisabled"] is False, recovered
            assert text(page, "#model") == "YOLOX-S MOT17", text(page, "#model")
            assert not page.errors, page.errors
            print(f"PASS: {scenario} model load recovers via Retry Model")
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
            shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

