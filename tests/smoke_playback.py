"""Real browser/model playback checks; uses local palace.mp4 and a fake webcam."""
import tempfile
import time
from smoke_web import ROOT, find_chrome, free_port, start_server, start_browser, new_page, click


def checks(page):
    page.call('Runtime.enable')
    page.wait_for('!!window.privacyDemo && privacyDemo.detector.loaded', bool, timeout=90)
    page.call('DOM.enable')
    doc = page.call('DOM.getDocument', {'depth': 1}, primitive=False)
    node = page.call('DOM.querySelector', {'nodeId': doc['root']['nodeId'], 'selector': '#video-input'}, primitive=False)

    def load():
        page.call('DOM.setFileInputFiles', {'files': [str(ROOT / 'videos/palace.mp4')], 'nodeId': node['nodeId']})
        page.evaluate("document.querySelector('#video-input').dispatchEvent(new Event('change',{bubbles:true}))")
        page.wait_for('privacyDemo.camera.ready && !privacyDemo.state.cameraBusy', bool)

    load()
    page.wait_for('privacyDemo.tracker.tick', lambda n: n > 1)
    assert page.evaluate("document.querySelector('#video-playback-controls').hidden") is False
    assert page.evaluate("Number(document.querySelector('#video-timeline').max) === privacyDemo.camera.duration")
    for width in (1280, 360):
        page.call('Emulation.setDeviceMetricsOverride', {'width': width, 'height': 900,
                  'deviceScaleFactor': 1, 'mobile': False})
        assert page.evaluate("""(() => {
          const bar=document.querySelector('#video-playback-controls');
          return bar.scrollWidth <= bar.clientWidth &&
            document.querySelector('#video-timeline').getBoundingClientRect().width >= 60;
        })()"""), f'Playback controls overflow at {width}px'
    page.call('Emulation.clearDeviceMetricsOverride')
    assert page.evaluate("[65,135,3735,NaN,Infinity].map(privacyDemo.formatTime)") == ['01:05','02:15','01:02:15','00:00','00:00']
    click(page, '#video-pause-button')
    page.wait_for('privacyDemo.camera.paused && !privacyDemo.state.detecting', bool)
    snap = "({time:privacyDemo.camera.currentTime,tick:privacyDemo.tracker.tick,count:privacyDemo.detector.inferenceCount,resets:privacyDemo.state.resetCount,ids:[...privacyDemo.stableIds.rawToStable]})"
    first = page.evaluate(snap)
    time.sleep(1)
    assert page.evaluate(snap) == first, 'Pause must freeze time, IDs, ticks and inference'
    assert page.evaluate("document.querySelector('#tracking-state').textContent") == 'Paused'
    assert page.evaluate("document.querySelector('#diag-detector').textContent.startsWith('0.0 FPS')")
    page.evaluate('window.statesBeforeResume = privacyDemo.stableIds.states')
    click(page, '#video-pause-button')
    page.wait_for('privacyDemo.detector.inferenceCount', lambda n: n > first['count'])
    assert page.evaluate('privacyDemo.state.resetCount') == first['resets']
    assert page.evaluate('statesBeforeResume === privacyDemo.stableIds.states')
    assert page.evaluate('privacyDemo.camera.currentTime') > first['time']
    assert page.evaluate("Math.abs(Number(document.querySelector('#video-timeline').value)-privacyDemo.camera.currentTime)<1")
    print('PASS: load, clock/duration, timeline progress, pause freezes AI, resume preserves session', flush=True)

    click(page, '#video-pause-button')
    page.wait_for('!privacyDemo.state.detecting', bool)
    before = page.evaluate('privacyDemo.state.resetCount')
    session = page.evaluate('privacyDemo.detector.sessionCount')
    # Multiple drag previews do not reset; only the committed change seeks.
    target = page.evaluate("""(() => {
      const t=document.querySelector('#video-timeline');
      for (const fraction of [.2,.3,.5]) {
        t.value=String(privacyDemo.camera.duration*fraction);
        t.dispatchEvent(new Event('input',{bubbles:true}));
      }
      return Number(t.value);
    })()""")
    assert page.evaluate('privacyDemo.state.resetCount') == before
    page.evaluate("document.querySelector('#video-timeline').dispatchEvent(new Event('change',{bubbles:true}))")
    page.wait_for('privacyDemo.state.resetCount', lambda n: n == before + 1)
    assert abs(page.evaluate('privacyDemo.camera.currentTime')-target) < .1
    assert page.evaluate('privacyDemo.tracker.tick') == 0
    assert page.evaluate('privacyDemo.stableIds.states.size') == 0
    time.sleep(.5)
    assert page.evaluate('privacyDemo.state.resetCount') == before + 1
    assert page.evaluate('privacyDemo.detector.sessionCount') == session
    print('PASS: drag commits once, seek clears tracker/Stable ID, no model reload', flush=True)

    # Delay precisely one detector result to verify epoch rejection across seek.
    page.evaluate("""window.realDetect=privacyDemo.detector.detect;
      privacyDemo.detector.detect=()=>new Promise(resolve=>{window.releaseDetection=resolve;});""")
    click(page, '#video-pause-button')
    page.wait_for('typeof window.releaseDetection === "function"', bool)
    page.evaluate('privacyDemo.camera.pauseVideo(); privacyDemo.camera.seekVideo(1);')
    page.wait_for('!privacyDemo.state.seeking && !privacyDemo.camera.video.seeking', bool)
    page.evaluate('releaseDetection([{x1:10,y1:10,x2:80,y2:200,score:.99}]); privacyDemo.detector.detect=realDetect;')
    page.wait_for('!privacyDemo.state.detecting', bool)
    assert page.evaluate('privacyDemo.tracker.tick') == 0
    assert page.evaluate('privacyDemo.stableIds.states.size') == 0
    print('PASS: result captured before seek cannot repopulate the new timeline', flush=True)

    page.evaluate('privacyDemo.camera.seekVideo(privacyDemo.camera.duration - 0.15)')
    page.wait_for('!privacyDemo.state.seeking && !privacyDemo.camera.video.seeking', bool)
    before = page.evaluate('privacyDemo.state.resetCount')
    click(page, '#video-pause-button')
    page.wait_for('privacyDemo.camera.currentTime', lambda n: n < 2, timeout=30)
    page.wait_for('privacyDemo.state.resetCount', lambda n: n > before)
    page.evaluate('privacyDemo.camera.pauseVideo()')
    assert page.evaluate('privacyDemo.state.resetCount') == before + 1
    print('PASS: automatic loop resets tracking once', flush=True)

    page.evaluate('privacyDemo.stopCamera()')
    assert page.evaluate("document.querySelector('#video-playback-controls').hidden")
    assert page.evaluate("document.querySelector('#video-timeline').value") == '0'
    load()
    assert page.evaluate("Number(document.querySelector('#video-timeline').max) === privacyDemo.camera.duration")
    page.evaluate('privacyDemo.startCamera()')
    page.wait_for('privacyDemo.camera.kind === "camera" && privacyDemo.camera.ready', bool)
    assert page.evaluate("document.querySelector('#video-playback-controls').hidden")
    assert page.evaluate('privacyDemo.getNetworkReport().imageUploads') == 0
    assert page.evaluate('privacyDemo.detector.maxInFlight') == 1
    page.evaluate('privacyDemo.stopCamera()')
    assert not page.errors, page.errors
    print('PASS: Stop/reload, webcam switch, one inference, zero frame uploads', flush=True)


def main():
    port, debug = free_port(), free_port()
    server = start_server(port)
    process = page = None
    with tempfile.TemporaryDirectory(prefix='privacy-playback-', ignore_cleanup_errors=True) as profile:
        try:
            browser = find_chrome()
            assert browser, 'Chrome/Edge required'
            process, _ = start_browser(browser, profile, debug)
            page = new_page(debug, f'http://127.0.0.1:{port}/')
            checks(page)
        finally:
            if page: page.close()
            if process:
                process.terminate()
                try: process.wait(timeout=10)
                except Exception:
                    process.kill()
                    process.wait(timeout=10)
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    main()
