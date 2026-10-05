"""Check pixels, not just button labels; run against a real model/local MP4."""
import time
import smoke_playback as harness
from smoke_web import ROOT, click


def checks(page):
    page.call('Runtime.enable')
    page.wait_for('!!window.privacyDemo && privacyDemo.detector.loaded', bool, timeout=90)
    page.call('DOM.enable')
    doc = page.call('DOM.getDocument', {'depth': 1}, primitive=False)
    node = page.call('DOM.querySelector', {'nodeId': doc['root']['nodeId'],
                     'selector': '#video-input'}, primitive=False)
    page.call('DOM.setFileInputFiles', {'files': [str(ROOT / 'videos/palace.mp4')], 'nodeId': node['nodeId']})
    page.evaluate("document.querySelector('#video-input').dispatchEvent(new Event('change',{bubbles:true}))")
    page.wait_for('privacyDemo.metrics.active', lambda n: n > 0)
    fps = page.wait_for("Number(document.querySelector('#fps').textContent)", lambda n: n and n > 0)
    print(f'PASS: live video FPS updates ({fps})', flush=True)
    # Simulate a browser without requestVideoFrameCallback; decoder statistics
    # must still supply video FPS, never rAF callbacks.
    page.evaluate('privacyDemo.camera.video.cancelVideoFrameCallback(privacyDemo.camera._frameCbId); privacyDemo.camera.video.requestVideoFrameCallback = undefined')
    page.wait_for("Number(document.querySelector('#fps').textContent)", lambda n: n and n > 0)
    click(page, '#video-pause-button')
    page.wait_for('privacyDemo.camera.paused && !privacyDemo.state.detecting', bool)
    if page.evaluate('privacyDemo.state.showOverlay'):
        click(page, '#overlay-button')
    before = page.evaluate('({tick:privacyDemo.tracker.tick,count:privacyDemo.detector.inferenceCount,time:privacyDemo.camera.currentTime})')
    page.evaluate("""window.pixelHash = () => {
      const c=document.querySelector('#canvas');
      const pixels=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
      let hash=2166136261;
      for (let i=0;i<pixels.length;i++) hash=Math.imul(hash ^ pixels[i],16777619);
      return hash>>>0;
    };""")
    on = page.evaluate('pixelHash()')
    click(page, '#privacy-button')
    off = page.evaluate('pixelHash()')
    assert on != off, 'Privacy OFF must actually change the paused pixels'
    assert page.evaluate('privacyDemo.pixelator.regions') == 0
    click(page, '#privacy-button')
    assert page.evaluate('pixelHash()') == on, 'Privacy ON must restore the processed frozen frame'
    assert page.evaluate('privacyDemo.pixelator.regions') > 0
    page.evaluate("const style=document.querySelector('#style-select'); style.value='retro'; style.dispatchEvent(new Event('change'));")
    retro = page.evaluate('pixelHash()')
    assert retro != on
    click(page, '#privacy-button')
    assert page.evaluate('pixelHash()') == off, 'Retro OFF must restore the same raw frame'
    click(page, '#privacy-button')
    assert page.evaluate('pixelHash()') == retro
    time.sleep(.6)
    assert page.evaluate('({tick:privacyDemo.tracker.tick,count:privacyDemo.detector.inferenceCount,time:privacyDemo.camera.currentTime})') == before
    assert page.evaluate("document.querySelector('#fps').textContent") == '0.0'
    assert page.evaluate("document.querySelector('#diag-detector').textContent.startsWith('0.0 FPS')")
    print('PASS: block/retro Privacy changes actual paused pixels without inference or track updates', flush=True)
    click(page, '#video-pause-button')
    page.wait_for('privacyDemo.detector.inferenceCount', lambda n: n > before['count'])
    page.wait_for("Number(document.querySelector('#fps').textContent)", lambda n: n and n > 0)
    assert page.evaluate('privacyDemo.getNetworkReport().imageUploads') == 0
    page.evaluate('privacyDemo.stopCamera()')
    assert not page.errors, page.errors
    print('PASS: FPS resumes, decoder fallback works, zero uploads and no page errors', flush=True)


if __name__ == '__main__':
    harness.checks = checks
    harness.main()
