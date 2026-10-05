/**
 * Application wiring for the browser-local privacy pipeline.
 *
 *   Webcam -> getUserMedia -> detector -> tracker -> Stable ID -> 5% padding
 *          -> block pixelation -> privacy hold -> canvas
 *
 * There is exactly one animation-frame loop and one in-flight inference. If the
 * detector is slower than the camera the current inference frame is skipped
 * instead of queued, so latency stays low and no backlog builds up.
 */

import { CONFIG, LIMITS } from './config.js';
import { CameraSource, SOURCE } from './camera.js';
import { PersonDetector, MODEL_LABEL } from './detector.js';
import { Tracker, predictBox, advanceBox } from './tracker.js';
import { StableIdManager } from './stable-id.js';
import { Pixelator, paddingBox } from './privacy.js';
import { Renderer, OVERLAY_ACTIVE, OVERLAY_HOLD } from './renderer.js';
import { installNetworkGuard, getNetworkReport } from './network-guard.js';
import { isUsableBox } from './geometry.js';
import { bindPixelArtControls } from './pixel-art-controls.js';

const els = {
  camera: document.getElementById('camera'),
  canvas: document.getElementById('canvas'),
  rawCanvas: document.getElementById('raw-canvas'),
  rawWrap: document.getElementById('raw-wrap'),
  viewerMessage: document.getElementById('viewer-message'),
  error: document.getElementById('error'),
  cameraState: document.getElementById('camera-state'),
  people: document.getElementById('people'),
  fps: document.getElementById('fps'),
  privacyState: document.getElementById('privacy-state'),
  stableState: document.getElementById('stable-state'),
  trackingState: document.getElementById('tracking-state'),
  model: document.getElementById('model'),
  modelRetry: document.getElementById('model-retry'),
  resolution: document.getElementById('resolution'),
  diagDetector: document.getElementById('diag-detector'),
  diagTracks: document.getElementById('diag-tracks'),
  diagRender: document.getElementById('diag-render'),
  diagDevice: document.getElementById('diag-device'),
  diagNetwork: document.getElementById('diag-network'),
  diagIds: document.getElementById('diag-ids'),
  start: document.getElementById('start-button'),
  stop: document.getElementById('stop-button'),
  video: document.getElementById('video-input'),
  videoName: document.getElementById('video-name'),
  playback: document.getElementById('video-playback-controls'),
  play: document.getElementById('video-pause-button'),
  timeline: document.getElementById('video-timeline'),
  currentTime: document.getElementById('video-current-time'),
  duration: document.getElementById('video-duration'),
  privacy: document.getElementById('privacy-button'),
  stable: document.getElementById('stable-button'),
  overlay: document.getElementById('overlay-button'),
  comparison: document.getElementById('comparison-button'),
  padding: document.getElementById('padding-input'),
  paddingValue: document.getElementById('padding-value'),
  scale: document.getElementById('scale-input'),
  scaleValue: document.getElementById('scale-value'),
  hold: document.getElementById('hold-input'),
  holdValue: document.getElementById('hold-value'),
  style: document.getElementById('style-select'),
  tuningReset: document.getElementById('tuning-reset'),
};

const camera = new CameraSource(els.camera, CONFIG.camera);
const detector = new PersonDetector(CONFIG.detector);
const tracker = new Tracker(CONFIG.tracker);
const stableIds = new StableIdManager(CONFIG.stableId);
const pixelator = new Pixelator(CONFIG.privacy);
const renderer = new Renderer(els.canvas, CONFIG.renderer);
const rawRenderer = new Renderer(els.rawCanvas, CONFIG.renderer);

const state = {
  rafId: null,
  detecting: false,
  epoch: 0,
  context: '—',
  lastDetectStartMs: -Infinity,
  lastResultMs: 0,
  showOverlay: true,
  comparison: false,
  lastRenderMs: 0,
  cameraBusy: false,
  sourceRequest: 0,
  scrubbing: false,     // the timeline pointer owns the thumb; don't overwrite it
  seeking: false,
  awaitingFrame: false,
  previousVideoTime: null,
  playbackBusy: false,
  resetCount: 0,
  displayFrameValid: false,
};

const frame = { width: 0, height: 0 };
const regions = [];
const holdStates = [];

let smoothedFps = null;
let smoothedDetFps = null;
let loopFrames = 0;
let loopDetections = 0;
let sampleStartMs = 0;
let frameBase = 0;
let lastUiMs = 0;

installNetworkGuard();

async function startup() {
  describeDevice();
  els.model.textContent = 'YOLOX-S MOT17 — loading';
  renderViewerMessage('Loading the YOLOX-S MOT17 model in this browser…');
  // Opt-in (?autostart=1): open the camera without a click. Used by the headless
  // smoke test and handy for demos; camera permission still applies. It runs
  // before the model finishes loading because the preview stays hidden until
  // the detector is ready, so no unprotected frame is ever shown.
  try {
    if (new URLSearchParams(window.location.search).get('autostart') === '1') {
      void startCamera();
    }
  } catch (error) {
    showError('Autostart failed: ' + error.message);
  }
  try {
    await detector.load();
    els.model.textContent = MODEL_LABEL;
    // Keep any 'Starting the camera…' message from an early autostart.
    if (!state.cameraBusy && (camera.state === 'idle' || camera.state === 'error')) {
      renderViewerMessage('Model ready. Press Start Camera — frames are processed on this device.');
    }
  } catch (error) {
    els.model.textContent = 'Model failed to load';
    // Say it here too: the model tile is small and easy to miss, and this is
    // the line that explains why Start Camera will not do anything.
    renderViewerMessage(
      'The detector model could not be loaded, so nothing can be processed yet. ' +
      'Fix the cause and press Retry Model.');
    showError('Detector model could not be loaded: ' + error.message);
  }
  finally {
    refreshControls();
  }
}

/**
 * Load the model on demand, after a failure or a manual retry.
 *
 * PersonDetector no longer caches a failed attempt, so this really starts a
 * new one instead of re-throwing the previous error. Kept separate from
 * startup() because startup() also seeds the messages for a first load.
 */
async function retryModel() {
  clearError();
  els.modelRetry.disabled = true;
  els.model.textContent = `${MODEL_LABEL} — loading`;
  renderViewerMessage('Reloading the YOLOX-S MOT17 model in this browser…');
  try {
    await detector.load();
    els.model.textContent = MODEL_LABEL;
    if (camera.state === 'idle' || camera.state === 'error') {
      renderViewerMessage('Model ready. Press Start Camera — frames are processed on this device.');
    }
  } catch (error) {
    els.model.textContent = 'Model failed to load';
    renderViewerMessage(
      'The detector model could not be loaded, so nothing can be processed yet. ' +
      'Fix the cause and press Retry Model.');
    showError('Detector model could not be loaded: ' + error.message);
  } finally {
    els.modelRetry.disabled = false;
    refreshControls();
    updateUi(performance.now());
  }
}

/** Counters shown in the diagnostics panel. */
const metrics = { active: 0, held: 0, holdStates: 0, tracks: 0 };

const clamp01 = (value) => (value < 0 ? 0 : (value > 1 ? 1 : value));

/**
 * Start the one and only processing loop. The guard is what makes repeated
 * Start/Stop cycles safe: a second loop can never be created.
 */
function startLoop() {
  if (state.rafId !== null) return;
  sampleStartMs = 0;
  loopFrames = 0;
  loopDetections = 0;
  frameBase = camera.presentedFrameCount;
  state.rafId = requestAnimationFrame(loop);
}

function stopLoop() {
  if (state.rafId === null) return;
  cancelAnimationFrame(state.rafId);
  state.rafId = null;
}

function loop(nowMs) {
  state.rafId = requestAnimationFrame(loop);
  updateTimeline();
  if (camera.kind === SOURCE.VIDEO && !state.seeking && !camera.video.seeking) {
    const time = camera.currentTime;
    if (state.previousVideoTime !== null && time < state.previousVideoTime - 0.1) {
      beginSeek();
      finishSeek(); // Fallback for browsers that omit loop seeking events.
    }
    state.previousVideoTime = time;
  }
  if (!camera.ready || state.seeking || camera.video.seeking) return;
  if (videoHalted()) {
    if (nowMs - lastUiMs >= CONFIG.metrics.uiUpdateMs) updateUi(nowMs);
    return; // Preserve the last processed canvas; do not tick any AI state.
  }
  frame.width = camera.width;
  frame.height = camera.height;
  if (renderer.resize(frame.width, frame.height)) rawRenderer.resize(frame.width, frame.height);
  const interval = Math.max(16, CONFIG.detector.intervalMs);
  const fraction = clamp01((nowMs - state.lastResultMs) / interval);
  if (!state.awaitingFrame) renderFrame(fraction);
  sampleFps(nowMs);
  // Latest-frame scheduling: skip this tick while the previous inference runs.
  if (!state.detecting && detector.loaded && nowMs - state.lastDetectStartMs >= interval) {
    void detectTick(nowMs);
  }
}

/** Reused offscreen canvas: webcam frame -> ImageData for one inference. */
const grabCanvas = document.createElement('canvas');
const grabCtx = grabCanvas.getContext('2d', { willReadFrequently: true });
// Raw source of the LAST DISPLAYED frame, retained only in browser memory.
// UI changes can repaint it while paused without inference or tracker updates.
const displaySource = document.createElement('canvas');
const displayCtx = displaySource.getContext('2d', { alpha: false });

/**
 * Copy the current video frame into an ImageData for the detector.
 * The copy stays inside this tab - it is never handed to fetch, XHR or a
 * WebSocket, which is what keeps raw webcam frames off the network.
 */
function grabFrame(video) {
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) return null;
  if (grabCanvas.width !== width || grabCanvas.height !== height) {
    grabCanvas.width = width;
    grabCanvas.height = height;
  }
  grabCtx.drawImage(video, 0, 0, width, height);
  return grabCtx.getImageData(0, 0, width, height);
}

async function detectTick(nowMs) {
  state.detecting = true;
  state.lastDetectStartMs = nowMs;
  const epoch = state.epoch;
  try {
    const imageData = grabFrame(camera.video);
    if (!imageData) return;
    const boxes = await detector.detect(imageData);
    // Discard results from a session that was stopped while inferring.
    if (!camera.ready || epoch !== state.epoch || state.seeking || camera.video.seeking || videoHalted()) return;
    tracker.update(boxes, performance.now(), frame);
    stableIds.removeExpired(tracker.tick);
    assignStableIds();
    state.lastResultMs = performance.now();
    loopDetections += 1;
    state.awaitingFrame = false;
  } catch (error) {
    showError('Detection failed: ' + error.message);
  } finally {
    state.detecting = false;
  }
}

/**
 * Give every tracker (raw) ID a stable display ID and collect the held IDs.
 * Port of the per-frame loop in PrivacyEngine.process_frame().
 */
function assignStableIds() {
  // Port of the per-frame loop in PrivacyEngine.process_frame(): only valid,
  // confirmed, currently matched tracks feed Stable ID + People Count, and the
  // aspect-ratio / min-area filter runs on the Kalman track box.
  const liveStableIds = new Set();
  const pending = [];
  for (const track of tracker.tracks) {
    track.valid = isUsableBox(track.box, CONFIG.tracker);
    const mapped = stableIds.rawToStable.get(track.id);
    if (mapped !== undefined) liveStableIds.add(mapped);
    if (track.missingTicks === 0 && track.confirmed && track.valid) {
      pending.push(track);
    }
  }
  // Reserve existing IDs first, so association order cannot matter.
  const activeStableIds = new Set();
  for (const track of pending) {
    const mapped = stableIds.rawToStable.get(track.id);
    if (mapped !== undefined) activeStableIds.add(mapped);
  }
  let active = 0;
  let held = 0;
  for (const track of pending) {
    const stableId = stableIds.update(track.id, track.box, tracker.tick,
      activeStableIds, frame);
    track.stableId = stableId;
    activeStableIds.add(stableId);
    active += 1;
  }
  for (const track of tracker.tracks) {
    if (track.missingTicks > 0 && track.confirmed && track.valid) held += 1;
  }
  // Safety net: IDs whose raw track already expired but are still inside the
  // privacy hold window (mirrors the second loop in the Python prototype).
  holdStates.length = 0;
  const holdFrames = CONFIG.privacy.holdFrames;
  for (const [stableId, stableState] of stableIds.states) {
    // Live raw tracks already draw their own HOLD region from the tracker's
    // Kalman box; only stable IDs without a raw track need this prediction.
    if (liveStableIds.has(stableId)) continue;
    const missingTicks = tracker.tick - stableState.lastSeen;
    if (missingTicks < 1 || missingTicks > holdFrames) continue;
    holdStates.push({
      stableId,
      missingTicks,
      score: 0,
      velocityX: stableState.velocity.x,
      velocityY: stableState.velocity.y,
      box: stableIds.getHoldBox(stableId, tracker.tick, frame),
    });
  }
  metrics.active = active;
  metrics.held = held;
  metrics.holdStates = holdStates.length;
  metrics.tracks = tracker.tracks.length;
}

/**
 * Draw one output frame: raw video -> pixelated regions -> overlays.
 * Called from the animation loop, never from the detector.
 */
function renderFrame(fraction) {
  const renderStart = performance.now();
  const privacyOn = CONFIG.privacy.enabled;
  pixelator.regions = 0;  // diagnostics show regions per frame, not a lifetime total
  if (privacyOn && !detector.loaded) {
    // Never show an unprotected frame while privacy mode claims to be on.
    // A failed detector leaves its own message behind, so only paint the
    // loading note while the load can still settle: otherwise this per-frame
    // write would overwrite the real error with "Loading..." forever and the
    // page would look stuck on loading after a load failure.
    renderer.clear();
    if (!detector.loadError) {
      renderViewerMessage('Loading the detector — the preview stays hidden until ' +
        'privacy processing is ready.');
    }
    state.lastRenderMs = performance.now() - renderStart;
    return;
  }
  hideViewerMessage();
  if (displaySource.width !== frame.width || displaySource.height !== frame.height) {
    displaySource.width = frame.width;
    displaySource.height = frame.height;
  }
  displayCtx.drawImage(camera.video, 0, 0, frame.width, frame.height);
  state.displayFrameValid = true;
  renderer.drawVideo(displaySource);
  if (state.comparison) rawRenderer.drawVideo(displaySource);
  // Retro quantises on the CPU: give it one snapshot of the drawn frame instead
  // of a per-region GPU readback. No-op for the block fast path.
  pixelator.beginFrame(renderer.ctx);
  const holdFrames = CONFIG.privacy.holdFrames;
  let count = 0;
  for (const track of tracker.tracks) {
    if (track.missingTicks > holdFrames || !track.confirmed || !track.valid) continue;
    count = drawRegion(track, predictBox(track, fraction, frame), count, privacyOn);
  }
  for (const hold of holdStates) {
    const box = advanceBox(hold.box, hold.velocityX, hold.velocityY, fraction, frame);
    count = drawRegion(hold, box, count, privacyOn);
  }
  regions.length = count;
  pixelator.flushFrame(renderer.ctx);
  if (state.showOverlay) renderer.drawOverlays(regions);
  state.lastRenderMs = performance.now() - renderStart;
}

/**
 * Pad, clip, pixelate and (optionally) label one privacy region.
 * @returns {number} next free index in the reusable region pool
 */
function drawRegion(source, box, count, privacyOn) {
  const padded = paddingBox(box, frame);
  if (padded.x2 - padded.x1 < 2 || padded.y2 - padded.y1 < 2) return count;
  if (privacyOn) pixelator.apply(renderer.ctx, renderer.toCanvasBox(padded));
  const region = regions[count] || (regions[count] = { box: null, label: '', color: '' });
  region.sourceBox = box;
  region.box = padded;
  region.label = labelFor(source);
  region.color = source.missingTicks === 0 ? OVERLAY_ACTIVE : OVERLAY_HOLD;
  return count + 1;
}

function labelFor(source) {
  const id = source.stableId === undefined ? '?' : source.stableId;
  if (source.missingTicks === 0) return `SID:${id} ${(source.score || 0).toFixed(2)}`;
  return `SID:${id} HOLD ${source.missingTicks}`;
}

/**
 * Exponential moving average FPS, refreshed on a fixed sampling window.
 *
 * For a video file the reported rate is the number of frames the decoder
 * actually presented, not the render-loop rate: the loop spins on every
 * animation frame (60-120 Hz on a fast machine) while the clip may only be
 * 30 FPS, so counting rAF callbacks would advertise a frame rate the source
 * does not have. The webcam path keeps the loop rate, because there the loop
 * is what is actually pulling new frames.
 */
function sampleFps(nowMs) {
  if (sampleStartMs === 0) {
    sampleStartMs = nowMs;
    loopFrames = 0;
    loopDetections = 0;
    frameBase = camera.presentedFrameCount;
    return;
  }
  loopFrames += 1;
  if (nowMs - sampleStartMs < CONFIG.metrics.sampleMs) return;
  const elapsed = nowMs - sampleStartMs;
  const presented = camera.presentedFrameCount - frameBase;
  const countsSourceFrames = camera.kind === SOURCE.VIDEO;
  const currentFps = countsSourceFrames
    ? (presented * 1000) / elapsed
    : (loopFrames * 1000) / elapsed;
  const currentDetFps = (loopDetections * 1000) / elapsed;
  const keep = 1 - CONFIG.metrics.fpsSmoothing;
  // A halted clip presents no frames, so this window only counts rAF ticks:
  // freeze the clip-FPS estimate instead of feeding it the render-loop rate.
  if (!videoHalted()) {
    smoothedFps = smoothedFps === null
      ? currentFps : CONFIG.metrics.fpsSmoothing * currentFps + keep * smoothedFps;
  }
  smoothedDetFps = smoothedDetFps === null
    ? currentDetFps
    : CONFIG.metrics.fpsSmoothing * currentDetFps + keep * smoothedDetFps;
  loopFrames = 0;
  loopDetections = 0;
  frameBase = camera.presentedFrameCount;
  sampleStartMs = nowMs;
  if (nowMs - lastUiMs >= CONFIG.metrics.uiUpdateMs) updateUi(nowMs);
}

function updateUi(nowMs) {
  lastUiMs = nowMs;
  const live = camera.state === 'live';
  const halted = videoHalted();
  const fpsShown = live && !halted && smoothedFps !== null ? smoothedFps : (live ? 0 : null);
  els.people.textContent = live ? String(metrics.active) : '—';
  els.fps.textContent = fpsShown === null ? '—' : fpsShown.toFixed(1);
  els.privacyState.textContent = CONFIG.privacy.enabled ? 'ON' : 'OFF';
  els.stableState.textContent = stableIds.enabled ? 'ON' : 'OFF';
  els.trackingState.textContent = live
    ? (halted ? 'Paused' : (state.detecting ? 'Active (inference)' : 'Active'))
    : camera.state;
  els.cameraState.textContent = live
    ? (camera.kind === SOURCE.VIDEO ? (halted ? 'Video - Paused' : '● Video') : '● Live')
    : (camera.state === 'error' ? 'Error' : 'Camera off');
  els.resolution.textContent = live ? `${camera.width}x${camera.height}` : '—';
  const detectionFps = halted || smoothedDetFps === null ? 0 : smoothedDetFps;
  els.diagDetector.textContent = `${detectionFps.toFixed(1)} FPS · ` +
    `${detector.inferenceMs.toFixed(0)} ms · backend ${detector.backend}`;
  els.diagTracks.textContent = `${metrics.tracks} tracks · ${metrics.active} active · ` +
    `${metrics.held} hold + ${metrics.holdStates} expired-hold`;
  els.diagRender.textContent = `${state.lastRenderMs.toFixed(1)} ms · ` +
    `${pixelator.regions} regions · small ${pixelator.lastSmall} · ` +
    `block ${pixelator.blockPixels}px`;
  els.diagIds.textContent = `${stableIds.rawToStable.size} raw ids · ` +
    `${stableIds.states.size} stable ids · next ${stableIds.nextStableId} · ` +
    `gap ${stableIds.config.maxGap}`;
  els.diagDevice.textContent = `${camera.trackLabel} · ${state.context} · ` +
    `inferences ${detector.inferenceCount} · session ${detector.sessionCount}`;
  const network = getNetworkReport();
  els.diagNetwork.textContent = `${network.requests} requests · ` +
    `${network.uploads} with body · ${network.imageUploads} image/frame uploads` +
    (network.websockets ? ` · ${network.websockets} websockets` : '');
  updateTimeline();
}

function renderViewerMessage(text) {
  els.viewerMessage.hidden = false;
  els.viewerMessage.textContent = text;
}

function hideViewerMessage() {
  els.viewerMessage.hidden = true;
}

function showError(message) {
  els.error.hidden = false;
  els.error.textContent = message;
}

function clearError() {
  els.error.hidden = true;
  els.error.textContent = '';
}

/** Fresh session: forget tracks, IDs, metrics and pending privacy holds. */
function resetSession() {
  state.displayFrameValid = false;
  state.resetCount++;
  tracker.reset();
  stableIds.reset();
  pixelator.reset();
  holdStates.length = 0;
  regions.length = 0;
  metrics.active = 0;
  metrics.held = 0;
  metrics.holdStates = 0;
  metrics.tracks = 0;
  smoothedFps = null;
  smoothedDetFps = null;
  loopFrames = 0;
  loopDetections = 0;
  sampleStartMs = 0;
  frameBase = 0;
  lastUiMs = 0;
  state.scrubbing = false;
  state.seeking = false;
  state.awaitingFrame = false;
  state.previousVideoTime = null;
  state.epoch += 1;
  state.lastResultMs = performance.now();
  state.lastDetectStartMs = -Infinity;
}

/** Open the camera and start the single processing loop. */
async function startCamera() {
  await startSource(null);
}

async function startVideo(file) {
  console.log('file selected');
  try {
    await startSource(file);
  } catch (error) {
    console.log('startVideo threw', error);
  } finally {
    console.log('startVideo settled, cameraBusy =', state.cameraBusy);
  }
}

/**
 * Source first, detector second — never Promise.all.
 *
 * The previous Promise.all([source, detector.load()]) gated the processing
 * loop on the ~36MB model download: picking an MP4 before the model finished
 * left the page on "Loading video..." even though the video itself was ready
 * in <1s. Phase 1 now awaits only the media source and starts the loop;
 * phase 2 awaits the detector without holding the loading lock, so a slow
 * model can never look like a stuck video. renderFrame() keeps the canvas
 * hidden under a detector message while privacy is ON, so no unprotected
 * frame is ever shown. A detector failure keeps the playing video and
 * reports the error instead of tearing a valid source down.
 */
async function startSource(file) {
  if (state.cameraBusy) {
    console.log('[load] startSource ignored: cameraBusy already true');
    return;
  }
  if (!file && camera.state === 'live' && camera.kind === SOURCE.CAMERA) return;
  const request = ++state.sourceRequest;
  const label = file ? (file.name || 'video') : 'camera';
  clearError();
  stopLoop();
  camera.stop();
  resetSession(); // Invalidate results BEFORE awaiting a different source.
  renderer.clear();
  rawRenderer.clear();
  state.cameraBusy = true;
  renderViewerMessage(file ? 'Loading video and preparing detector...' : 'Opening camera and preparing detector...');
  refreshControls();
  try {
    console.log('[load] source load started:', label);
    await (file ? camera.startFile(file) : camera.start());
    if (request !== state.sourceRequest) {
      console.log('[load] source superseded after await:', label);
      return;
    }
    console.log('[load] source ready:', label,
      `${camera.width}x${camera.height}`,
      `muted=${camera.video.muted}`, `readyState=${camera.video.readyState}`);
    startLoop();
    console.log('[load] processing started');
  } catch (error) {
    console.log('[load] startSource failed:', error && error.message ? error.message : error);
    if (request !== state.sourceRequest) return;
    camera.stop();
    stopLoop();
    const message = 'Unable to start processing: ' + (error.message || String(error));
    showError(message);
    renderViewerMessage(message);
    return;
  } finally {
    // Release the loading lock once the media source settles, even though the
    // detector may still be pending: otherwise a slow model keeps the file
    // picker disabled and the page looks hung on "Loading video...".
    if (request === state.sourceRequest) {
      state.cameraBusy = false;
      els.video.value = ''; // Selecting the same file again must fire change.
      refreshControls();
      updateUi(performance.now());
    }
  }
  // Phase 2: detector without holding the source hostage.
  if (detector.loaded) {
    if (request === state.sourceRequest) hideViewerMessage();
    return;
  }
  if (request === state.sourceRequest) {
    renderViewerMessage('Loading the detector — the preview stays hidden until ' +
      'privacy processing is ready.');
    refreshControls();
  }
  try {
    console.log('[load] detector initialization started (source ready, not blocking)');
    await detector.load();
    if (request !== state.sourceRequest) {
      console.log('[load] detector superseded:', label);
      return;
    }
    console.log('[load] detector ready', detector.backend);
    hideViewerMessage();
  } catch (error) {
    if (request !== state.sourceRequest) return;
    console.log('[load] detector failed:', error && error.message ? error.message : error);
    showError('Detector model could not be loaded: ' + (error.message || String(error)));
    renderViewerMessage(
      'The detector model could not be loaded, so nothing can be processed yet. ' +
      'Fix the cause and press Retry Model.');
  } finally {
    if (request === state.sourceRequest) {
      refreshControls();
      updateUi(performance.now());
    }
  }
}



function stopCamera() {
  state.sourceRequest++;
  state.cameraBusy = false;
  const wasVideo = camera.kind === SOURCE.VIDEO;
  camera.stop();
  stopLoop();
  resetSession();
  renderer.clear();
  rawRenderer.clear();
  renderViewerMessage((wasVideo ? 'Video' : 'Camera') + ' stopped - no frames are being captured.');
  updateUi(performance.now());
  refreshControls();
}
function repaintDisplayedFrame() {
  if (!state.displayFrameValid || state.seeking || state.awaitingFrame) return;
  const start = performance.now();
  renderer.drawVideo(displaySource);
  pixelator.regions = 0;
  if (CONFIG.privacy.enabled) {
    pixelator.beginFrame(renderer.ctx);
    for (const region of regions) {
      region.box = paddingBox(region.sourceBox, frame);
      pixelator.apply(renderer.ctx, renderer.toCanvasBox(region.box));
    }
    pixelator.flushFrame(renderer.ctx);
  }
  if (state.showOverlay) renderer.drawOverlays(regions);
  if (state.comparison) rawRenderer.drawVideo(displaySource);
  state.lastRenderMs = performance.now() - start;
}
function togglePrivacy() {
  CONFIG.privacy.enabled = !CONFIG.privacy.enabled;
  repaintDisplayedFrame();
  // Keep the button state and the actual pixelation pipeline in lockstep, so a
  // toggle is visible immediately without waiting for the next track update.
  els.privacy.setAttribute('aria-pressed', String(CONFIG.privacy.enabled));
  refreshControls();
  updateUi(performance.now());
}
function toggleStable() {
  stableIds.enabled = !stableIds.enabled;
  refreshControls();
  updateUi(performance.now());
}
function toggleOverlay() {
  state.showOverlay = !state.showOverlay;
  repaintDisplayedFrame();
  refreshControls();
  updateUi(performance.now());
}
function toggleComparison() {
  state.comparison = !state.comparison;
  els.rawWrap.hidden = !state.comparison;
  if (!state.comparison) rawRenderer.clear();
  else repaintDisplayedFrame();
  refreshControls();
  updateUi(performance.now());
}
/** Stable clock labels, including hour-long clips and unknown metadata. */
function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '00:00';
  const total = Math.floor(seconds);
  const h = Math.floor(total / 3600), m = Math.floor(total / 60) % 60, sec = total % 60;
  const pad = value => String(value).padStart(2, '0');
  return (h ? pad(h) + ':' : '') + pad(m) + ':' + pad(sec);
}
function videoHalted() {
  return camera.kind === SOURCE.VIDEO && (camera.paused || camera.video.ended);
}
function updateTimeline() {
  const visible = camera.kind === SOURCE.VIDEO && camera.state === 'live';
  els.playback.hidden = !visible;
  if (!visible) {
    els.timeline.value = '0';
    els.timeline.max = '0';
    els.currentTime.textContent = els.duration.textContent = '00:00';
    return;
  }
  els.timeline.max = String(camera.duration);
  els.timeline.disabled = camera.duration <= 0;
  els.play.disabled = state.playbackBusy || state.seeking;
  els.play.textContent = videoHalted() ? 'Play' : 'Pause';
  els.play.setAttribute('aria-label', videoHalted() ? 'Resume the video' : 'Pause the video');
  els.duration.textContent = formatTime(camera.duration);
  if (!state.scrubbing) {
    els.timeline.value = String(camera.currentTime);
    els.currentTime.textContent = formatTime(camera.currentTime);
  }
}
/** Invalidate old work at seek START, reset exactly once at seek completion. */
function beginSeek() {
  if (camera.kind !== SOURCE.VIDEO || camera.state !== 'live' || state.seeking) return;
  state.seeking = true;
  state.displayFrameValid = false;
  state.epoch++;
  renderer.clear();
  rawRenderer.clear();
}
function finishSeek() {
  if (!state.seeking || camera.video.seeking || camera.kind !== SOURCE.VIDEO) return;
  resetSession();
  state.previousVideoTime = camera.currentTime;
  state.awaitingFrame = true;
  renderer.clear();
  rawRenderer.clear();
  if (camera.paused) renderViewerMessage('Position selected. Press Play to process this frame.');
  updateTimeline();
  updateUi(performance.now());
}
function commitSeek() {
  state.scrubbing = false;
  if (camera.kind !== SOURCE.VIDEO) return;
  const target = Number(els.timeline.value);
  if (!Number.isFinite(target) || !camera.duration || Math.abs(target - camera.currentTime) < 0.001) {
    updateTimeline(); return;
  }
  beginSeek();
  try {
    if (!camera.seekVideo(target)) finishSeek();
  } catch (error) {
    finishSeek();
    showError('Could not seek: ' + error.message);
  }
  updateTimeline();
}
async function togglePlayback() {
  if (camera.kind !== SOURCE.VIDEO || camera.state !== 'live' || state.playbackBusy || state.seeking) return;
  const request = state.sourceRequest;
  state.playbackBusy = true;
  try {
    if (videoHalted()) await camera.resumeVideo();
    else camera.pauseVideo();
  } catch (error) {
    if (request === state.sourceRequest) showError('Could not resume the video: ' + error.message);
  } finally {
    state.playbackBusy = false;
    updateTimeline();
    updateUi(performance.now());
  }
}
function resetPlaybackSampling() {
  sampleStartMs = 0;
  loopFrames = loopDetections = 0;
  smoothedFps = smoothedDetFps = null;
  frameBase = camera.presentedFrameCount;
  state.lastDetectStartMs = -Infinity;
}
function refreshControls() {
  const starting = state.cameraBusy || camera.state === 'starting';
  const live = camera.state === 'live';
  const videoSource = camera.kind === SOURCE.VIDEO;
  const modelPending = !detector.loaded && !detector.loadError;
  els.privacy.setAttribute('aria-pressed', String(CONFIG.privacy.enabled));
  els.stable.setAttribute('aria-pressed', String(stableIds.enabled));
  els.overlay.setAttribute('aria-pressed', String(state.showOverlay));
  // Start must not pretend it can work without a detector: gated while the
  // model is still loading AND after a failed load (Retry Model is the way
  // out; startup()'s error copy says so). The file input stays ungated by the
  // model on purpose - the phase split lets a clip play before the model is
  // ready, with the preview hidden until privacy processing can run.
  els.start.disabled = starting || live || !detector.loaded;
  els.video.disabled = starting;
  els.modelRetry.hidden = detector.loaded || modelPending;
  els.stop.disabled = !starting && camera.state === 'idle';
  els.start.textContent = starting ? 'Starting...' : (live ? (videoSource ? 'Video On' : 'Camera On') : 'Start Camera');
  els.videoName.textContent = videoSource ? camera.fileName : 'Load MP4 / WebM';
  els.privacy.setAttribute('aria-pressed', String(CONFIG.privacy.enabled));
  els.privacy.textContent = 'Privacy ' + (CONFIG.privacy.enabled ? 'ON' : 'OFF');
  els.stable.setAttribute('aria-pressed', String(stableIds.enabled));
  els.stable.textContent = 'Stable ID ' + (stableIds.enabled ? 'ON' : 'OFF');
  els.overlay.setAttribute('aria-pressed', String(state.showOverlay));
  els.overlay.textContent = 'Boxes & IDs ' + (state.showOverlay ? 'ON' : 'OFF');
  els.comparison.textContent = 'Comparison ' + (state.comparison ? 'ON' : 'OFF');
  els.cameraState.textContent = live ? (videoSource ? 'Video' : 'Live') : (camera.state === 'error' ? 'Error' : 'Camera off');
  updateTimeline();
}
const PRIVACY_DEFAULTS = {
  padding: CONFIG.privacy.padding, pixelScale: CONFIG.privacy.pixelScale,
  holdFrames: CONFIG.privacy.holdFrames, style: CONFIG.privacy.style,
};
const percent = value => Math.round(Number(value) * 100) + '%';
function syncTuning() {
  els.padding.value = String(CONFIG.privacy.padding);
  els.paddingValue.textContent = percent(CONFIG.privacy.padding);
  els.scale.value = String(CONFIG.privacy.pixelScale);
  els.scaleValue.textContent = percent(CONFIG.privacy.pixelScale);
  els.hold.value = String(CONFIG.privacy.holdFrames);
  els.holdValue.textContent = String(CONFIG.privacy.holdFrames);
  els.style.value = CONFIG.privacy.style;
  if (videoHalted()) repaintDisplayedFrame();
}
function bindControls() {
  els.start.addEventListener('click', () => { void startCamera(); });
  els.stop.addEventListener('click', stopCamera);
  els.video.addEventListener('change', () => {
    const file = els.video.files && els.video.files[0];
    if (file) void startVideo(file);
  });
  els.privacy.addEventListener('click', togglePrivacy);
  els.modelRetry.addEventListener('click', () => { void retryModel(); });
  els.stable.addEventListener('click', toggleStable);
  els.overlay.addEventListener('click', toggleOverlay);
  els.comparison.addEventListener('click', toggleComparison);
  els.play.addEventListener('click', () => { void togglePlayback(); });
  els.timeline.addEventListener('pointerdown', () => { state.scrubbing = true; });
  els.timeline.addEventListener('input', () => {
    state.scrubbing = true; // Also protects keyboard previews from rAF updates.
    els.currentTime.textContent = formatTime(Number(els.timeline.value));
  });
  els.timeline.addEventListener('change', commitSeek);
  window.addEventListener('pointerup', () => { if (state.scrubbing) commitSeek(); });
  window.addEventListener('pointercancel', () => { state.scrubbing = false; updateTimeline(); });
  els.timeline.addEventListener('blur', () => { if (state.scrubbing) commitSeek(); });
  els.camera.addEventListener('seeking', beginSeek);
  els.camera.addEventListener('seeked', finishSeek);
  for (const event of ['timeupdate', 'durationchange', 'loadedmetadata']) {
    els.camera.addEventListener(event, updateTimeline);
  }
  for (const event of ['play', 'pause', 'ended']) {
    els.camera.addEventListener(event, () => {
      resetPlaybackSampling();
      updateTimeline();
      updateUi(performance.now());
    });
  }
  els.padding.addEventListener('input', () => { CONFIG.privacy.padding = Number(els.padding.value); syncTuning(); });
  els.scale.addEventListener('input', () => { CONFIG.privacy.pixelScale = Number(els.scale.value); syncTuning(); });
  els.hold.addEventListener('input', () => { CONFIG.privacy.holdFrames = Math.round(Number(els.hold.value)); syncTuning(); });
  els.style.addEventListener('change', () => {
    const previous = CONFIG.privacy.style;
    CONFIG.privacy.style = els.style.value;
    // Retro -> block: drop the CPU frame snapshot and scratch tile so the block
    // path stops carrying them, and the next block frame is already fast.
    if (previous === 'retro' && CONFIG.privacy.style !== 'retro') {
      pixelator.releaseRetroBuffers();
    }
    repaintDisplayedFrame();
  });
  els.tuningReset.addEventListener('click', () => { Object.assign(CONFIG.privacy, PRIVACY_DEFAULTS); syncTuning(); });
  window.addEventListener('pagehide', stopCamera);
}
function describeDevice() {
  state.context = window.isSecureContext ? 'secure context' : 'not a secure context (getUserMedia blocked)';
  if (!window.isSecureContext) showError('Open through https:// or http://localhost for camera access.');
}
bindControls();
bindPixelArtControls(CONFIG.privacy, () => { repaintDisplayedFrame(); updateUi(performance.now()); });
syncTuning();
refreshControls();
describeDevice();
updateUi(performance.now());
window.privacyDemo = Object.freeze({
  CONFIG, LIMITS, camera, detector, tracker, stableIds, pixelator, renderer, metrics,
  getNetworkReport, stopCamera, startCamera, startVideo, state, formatTime,
  stepWithDetections(boxes) {
    tracker.update(boxes, performance.now(), frame);
    stableIds.removeExpired(tracker.tick);
    assignStableIds();
    state.lastResultMs = performance.now();
    return {people: metrics.active, tracks: tracker.tracks.length};
  },
});
void startup();
