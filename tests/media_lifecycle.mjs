// No model/tracker mocks needed: exercise the media owner's async lifecycle.
import assert from 'node:assert/strict';
import fs from 'node:fs';
const source = fs.readFileSync(new URL('../web/js/camera.js', import.meta.url), 'utf8')
  .replace("import { CONFIG } from './config.js';", 'const CONFIG = {camera:{}};')
  .replace('const METADATA_TIMEOUT_MS = 8000;', 'const METADATA_TIMEOUT_MS = 100;');
const {CameraSource} = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
let revoked = [];
URL.createObjectURL = () => 'blob:test';
URL.revokeObjectURL = url => revoked.push(url);
class Video extends EventTarget {
  constructor() { super(); this.readyState=0; this.videoWidth=0; this.videoHeight=0; this.src=''; this.calls=0; }
  getAttribute() { return this.src; }
  removeAttribute() { this.src=''; }
  pause() {}
  load() {
    if (this.hang) return;
    // Synchronous readiness catches missed-event bugs as well as normal events.
    this.readyState=3; this.videoWidth=640; this.videoHeight=360;
    this.dispatchEvent(new Event('loadedmetadata'));
    this.dispatchEvent(new Event('canplay'));
  }
  play() {
    this.calls++;
    if (this.rejectAudio && !this.muted) return Promise.reject(new DOMException('Blocked', 'NotAllowedError'));
    if (this.hangPlay) return new Promise(()=>{});
    return Promise.resolve();
  }
}
const file = {name:'clip.mp4', type:'video/mp4'};
{
  const video=new Video(); video.rejectAudio=true;
  const camera=new CameraSource(video);
  assert.equal(await camera.startFile(file),true);
  assert.equal(camera.ready,true);
  assert.equal(video.calls,2);
  assert.equal(camera.audioBlocked,true);
  assert.equal(camera._starting,null);
  assert.equal(revoked.length,0,'URL must remain valid while playing');
  camera.stop(); assert.equal(revoked.length,1);
}
for (const pending of ['hang','hangPlay']) {
  const video=new Video(); video[pending]=true;
  const camera=new CameraSource(video);
  const load=camera.startFile(file);
  await new Promise(resolve=>setTimeout(resolve,0));
  camera.stop();
  await assert.rejects(load,{name:'AbortError'});
  assert.equal(camera._starting,null);
  assert.equal(camera.state,'idle');
  video[pending]=false;
  assert.equal(await camera.startFile(file),true,'retry after cancellation');
  camera.stop();
}
{
  const video=new Video(); video.play=()=>Promise.reject(new Error('decode failed'));
  const camera=new CameraSource(video);
  await assert.rejects(camera.startFile(file),/decode failed/);
  assert.equal(camera._starting,null);
  assert.equal(camera.state,'error');
  assert.equal(camera.objectUrl,null);
}
console.log('PASS: readiness, audio fallback, cancel during metadata/play, retry, error cleanup');
for (const pending of ['hang','hangPlay']) {
  const video=new Video(); video[pending]=true;
  const camera=new CameraSource(video);
  await assert.rejects(camera.startFile(file), /Timed out/);
  assert.equal(camera.state,'error');
  assert.equal(camera._starting,null);
}
console.log('PASS: stalled metadata/play time out and release loading state');
