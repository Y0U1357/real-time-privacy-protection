"""Real Chromium + local MP4 regression for source/model loading races."""
import tempfile
from pathlib import Path
from smoke_web import (ROOT, find_chrome, free_port, start_server,
                       start_browser, new_page)


def main():
    video = ROOT / "videos/palace.mp4"
    assert video.is_file(), "A local test MP4 is required: videos/palace.mp4"
    browser = find_chrome()
    assert browser, "Chrome or Edge is required"
    port, debug = free_port(), free_port()
    server = start_server(port)
    process = page = None
    with tempfile.TemporaryDirectory(prefix="privacy-loading-", ignore_cleanup_errors=True) as profile:
        try:
            process, _ = start_browser(browser, profile, debug)
            page = new_page(debug, "about:blank")
            page.call("Runtime.enable")
            page.call("Page.enable")
            page.call("Page.addScriptToEvaluateOnNewDocument", {"source": """
              window.loadingLogs=[];
              const log=console.log;
              console.log=(...args)=>{ loadingLogs.push(args.map(String).join(' ')); log(...args); };
              const realFetch=window.fetch;
              window.fetch=async (...args)=>{
                if (String(args[0]).endsWith('.onnx')) {
                  await new Promise(resolve=>{window.releaseModel=resolve;});
                }
                return realFetch(...args);
              };
              const realPlay=HTMLMediaElement.prototype.play;
              HTMLMediaElement.prototype.play=function() {
                if (!this.muted) return Promise.reject(new DOMException('Test autoplay policy','NotAllowedError'));
                return realPlay.call(this);
              };
            """})
            page.call("Page.navigate", {"url": f"http://127.0.0.1:{port}/"})
            page.wait_for("!!window.privacyDemo && !!window.releaseModel", bool)
            print("Model download held; selecting local MP4", flush=True)
            page.call("DOM.enable")
            doc = page.call("DOM.getDocument", {"depth": 1}, primitive=False)
            node = page.call("DOM.querySelector", {"nodeId": doc["root"]["nodeId"],
                             "selector": "#video-input"}, primitive=False)
            page.call("DOM.setFileInputFiles", {"files": [str(video)], "nodeId": node["nodeId"]})
            page.evaluate("document.querySelector('#video-input').dispatchEvent(new Event('change', {bubbles:true}))")
            # Phase split (source first, detector second): the media source
            # becomes ready while the model download is still held, and the
            # loading lock is released without it - busy and ready are never
            # observable together because the finally block runs in the same
            # microtask chain as the source-ready transition.
            page.wait_for("privacyDemo.camera.ready", bool)
            assert page.evaluate("privacyDemo.state.cameraBusy") is False
            assert page.evaluate("document.querySelector('#video-input').disabled") is False
            print("Video ready independently of model", flush=True)
            assert page.evaluate("privacyDemo.detector.loaded") is False
            page.evaluate("releaseModel()")
            page.wait_for("privacyDemo.detector.inferenceCount", lambda n: n >= 2, timeout=90)
            assert page.evaluate("privacyDemo.state.cameraBusy") is False
            assert page.evaluate("privacyDemo.camera.audioBlocked") is True
            assert page.evaluate("document.querySelector('#viewer-message').hidden") is True
            logs = page.evaluate("loadingLogs")
            for stage in ("file selected", "blob created", "metadata loaded", "canplay", "play success", "processing started"):
                assert any(stage in line for line in logs), (stage, logs)
            print("PASS: pick MP4 before model ready -> muted fallback -> inference -> loading cleared")
            page.evaluate("privacyDemo.stopCamera()")
            # A pending play() is cancelled immediately by Stop; it must not restart later.
            page.evaluate("""(() => {
              const source=privacyDemo.camera;
              window.originalPlay=source.video.play;
              source.video.play=()=>new Promise(()=>{});
            })()""")
            page.call("DOM.setFileInputFiles", {"files": [str(video)], "nodeId": node["nodeId"]})
            page.evaluate("document.querySelector('#video-input').dispatchEvent(new Event('change', {bubbles:true}))")
            page.wait_for("privacyDemo.camera.state === 'starting'", bool)
            page.evaluate("privacyDemo.stopCamera()")
            page.wait_for("privacyDemo.camera._starting === null", bool)
            assert page.evaluate("privacyDemo.state.cameraBusy") is False
            assert page.evaluate("privacyDemo.state.rafId") is None
            page.evaluate("privacyDemo.camera.video.play=originalPlay")
            # Invalid media must show an error and enable another selection.
            page.evaluate("privacyDemo.startVideo(new File(['invalid'], 'bad.mp4', {type:'video/mp4'}))")
            assert page.evaluate("privacyDemo.state.cameraBusy") is False
            assert page.evaluate("document.querySelector('#video-input').disabled") is False
            assert "Unable to start" in page.evaluate("document.querySelector('#error').textContent")
            assert not page.errors, page.errors
            print("PASS: Stop cancels pending playback; invalid file clears loading and permits retry")
        finally:
            if page:
                page.close()
            if process:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except Exception:
                    process.kill()
                    process.wait(timeout=10)
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
