/**
 * Webcam ownership only. This module knows nothing about the detector, the
 * tracker or privacy: it opens exactly one MediaStream and always releases it.
 *
 * navigator.mediaDevices.getUserMedia() keeps every raw frame inside the page;
 * nothing here can upload a frame because nothing here has network access.
 */

import { CONFIG } from './config.js';

export const CAMERA_STATE = {
  IDLE: 'idle',
  STARTING: 'starting',
  LIVE: 'live',
  ERROR: 'error',
};

/** Which source currently feeds the pipeline. */
export const SOURCE = {
  NONE: 'none',
  CAMERA: 'camera',
  VIDEO: 'video',
};

const METADATA_TIMEOUT_MS = 8000;

export class CameraSource {
  /**
   * @param {HTMLVideoElement} video the (visually hidden) source element
   * @param {object} [config] camera constraints, defaults to CONFIG.camera
   */
  constructor(video, config = CONFIG.camera) {
    this.video = video;
    this.config = config;
    this.state = CAMERA_STATE.IDLE;
    this.error = null;
    this.stream = null;
    this.kind = SOURCE.NONE;
    this.fileName = null;
    this.objectUrl = null;
    this.audioBlocked = false;
    this.presentedFrames = 0;
    this._frameCbId = null;
    this._starting = null;
    this._generation = 0;
    this._fileAbort = null;
  }

  get ready() {
    return this.state === CAMERA_STATE.LIVE && this.video.readyState >= 2 &&
      this.video.videoWidth > 0 && this.video.videoHeight > 0;
  }

  get width() {
    return this.video.videoWidth;
  }

  get height() {
    return this.video.videoHeight;
  }

  get currentTime() {
    return Number.isFinite(this.video.currentTime) ? this.video.currentTime : 0;
  }

  get duration() {
    return Number.isFinite(this.video.duration) ? Math.max(0, this.video.duration) : 0;
  }

  get paused() { return this.video.paused; }

  get presentedFrameCount() {
    if (typeof this.video.requestVideoFrameCallback === 'function') return this.presentedFrames;
    // Older browsers expose decoder counters instead of frame callbacks.
    const quality = this.video.getVideoPlaybackQuality?.();
    if (quality) return Math.max(0, quality.totalVideoFrames - quality.droppedVideoFrames);
    return Number(this.video.webkitDecodedFrameCount) || 0;
  }

  pauseVideo() {
    if (this.kind === SOURCE.VIDEO) this.video.pause();
  }

  async resumeVideo() {
    if (this.kind !== SOURCE.VIDEO || this.state !== CAMERA_STATE.LIVE) return;
    const generation = this._generation;
    const controller = new AbortController();
    this._fileAbort = controller;
    try {
      try {
        await boundedPlay(this.video, controller.signal);
      } catch (error) {
        if (controller.signal.aborted || !isAutoplayRefusal(error)) throw error;
        this.video.muted = true;
        this.audioBlocked = true;
        await boundedPlay(this.video, controller.signal);
      }
      return generation === this._generation;
    } finally {
      if (this._fileAbort === controller) this._fileAbort = null;
    }
  }

  seekVideo(seconds) {
    if (this.kind !== SOURCE.VIDEO || !this.duration || !Number.isFinite(seconds)) return false;
    const target = Math.max(0, Math.min(seconds, this.duration));
    if (Math.abs(target - this.currentTime) < 0.001) return false;
    this.video.currentTime = target;
    return true;
  }

  get trackLabel() {
    if (this.kind === SOURCE.VIDEO) return this.fileName || 'video file';
    if (!this.stream) return '—';
    const [track] = this.stream.getVideoTracks();
    return track ? (track.label || 'camera') : '—';
  }

  /**
   * Start the camera. Repeated calls while live or starting are ignored, so a
   * double click can never open a second stream or a second processing loop.
   * @returns {Promise<boolean>} true when this call opened the stream
   */
  async start() {
    if (this.state === CAMERA_STATE.STARTING) return this._starting;
    if (this.state === CAMERA_STATE.LIVE && this.kind === SOURCE.CAMERA) return false;
    this._starting = this._open();
    const pending = this._starting;
    try { return await pending; }
    finally { if (this._starting === pending) this._starting = null; }
  }

  async _open() {
    const generation = ++this._generation;
    const controller = new AbortController();
    this._fileAbort = controller;
    this.state = CAMERA_STATE.STARTING;
    this.error = null;
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('getUserMedia is unavailable. Use HTTPS or localhost.');
      const {width,height,frameRate} = this.config;
      const stream = await navigator.mediaDevices.getUserMedia({
        audio:false, video:{width:{ideal:width},height:{ideal:height},frameRate:{ideal:frameRate}},
      });
      // Permission dialogs cannot be cancelled, but their late streams can.
      if (controller.signal.aborted || generation !== this._generation) {
        for (const track of stream.getTracks()) track.stop();
        return false;
      }
      this._releaseStream();
      this.stream = stream;
      this.kind = SOURCE.CAMERA;
      this.video.srcObject = stream;
      this.video.muted = true;
      await waitForMedia(this.video, 1, 'loadedmetadata', controller.signal);
      await boundedPlay(this.video, controller.signal);
      const [track] = stream.getVideoTracks();
      track?.addEventListener('ended', () => {
        if (this.stream === stream) {
          this.state = CAMERA_STATE.ERROR;
          this.error = 'Camera stream ended. Press Start Camera to reconnect.';
          this._releaseStream();
        }
      });
      this.state = CAMERA_STATE.LIVE;
      return true;
    } catch (error) {
      if (generation === this._generation) {
        this._releaseStream();
        this.state = CAMERA_STATE.ERROR;
        this.error = describeCameraError(error);
      }
      throw error;
    } finally {
      if (this._fileAbort === controller) this._fileAbort = null;
    }
  }

  /**
   * Play a video file the user picked instead of the webcam. The file is read
   * through a blob: URL, so it stays inside this tab: no upload, no request,
   * exactly like the camera path.
   * @param {File} file a local video file (mp4 / webm / mov ...)
   * @returns {Promise<boolean>} true when the file is playing
   */
  async startFile(file) {
    if (this.state === CAMERA_STATE.STARTING) return this._starting;
    if (!file) throw new Error('No video file was selected.');
    // Some platforms hand over an empty MIME type, so also accept the extension.
    const looksLikeVideo = file.type.startsWith('video/') ||
      /\.(mp4|m4v|webm|mov|ogv|avi|mkv)$/i.test(file.name || '');
    if (!looksLikeVideo) {
      throw new Error('Please choose a video file (MP4 / WebM / MOV).');
    }
    this._starting = this._openFile(file);
    const pending = this._starting;
    try { return await pending; }
    finally { if (this._starting === pending) this._starting = null; }
  }

  async _openFile(file) {
    const generation = ++this._generation;
    const controller = new AbortController();
    this._fileAbort = controller;
    const signal = controller.signal;
    this.state = CAMERA_STATE.STARTING;
    this.error = null;
    const video = this.video;
    const fileName = file.name || 'video';
    try {
      this._releaseStream();
      this.kind = SOURCE.VIDEO;
      this.fileName = fileName;
      this.objectUrl = URL.createObjectURL(file);
      console.log('[load] blob created', this.objectUrl);
      video.loop = true;
      video.muted = false;
      video.preload = 'auto';
      video.src = this.objectUrl;
      video.load();
      await waitForMedia(video, 1, 'loadedmetadata', signal);
      console.log('[load] metadata loaded');
      await waitForMedia(video, 3, 'canplay', signal);
      console.log('[load] canplay');
      try {
        await boundedPlay(video, signal);
      } catch (error) {
        if (signal.aborted || !isAutoplayRefusal(error)) throw error;
        video.muted = true;
        this.audioBlocked = true;
        await boundedPlay(video, signal);
      }
      console.log('[load] play success');
      if (signal.aborted || generation !== this._generation) return false;
      this.state = CAMERA_STATE.LIVE;
      this._trackPresentedFrames();
      return true;
    } catch (error) {
      if (generation === this._generation) {
        // The blob: URL created above must die here too: _releaseStream() at the
        // top only clears the *previous* source. Without this call a failed
        // startFile() leaves objectUrl set (leak) and — worse — because state
        // falls through to IDLE while camera.kind is still VIDEO, later webcam
        // starts misdetect the active kind. Stop paths stay untouched: stop()
        // bumps _generation first, so a stop()-caused abort lands here with a
        // stale generation and revokes nothing twice.
        this._releaseStream();
        this.state = CAMERA_STATE.ERROR;
        this.error = 'Could not play ' + fileName + ': ' + error.message;
      }
      throw error;
    } finally {
      if (this._fileAbort === controller) this._fileAbort = null;
      if (generation === this._generation && this.state === CAMERA_STATE.STARTING) {
        this.state = CAMERA_STATE.IDLE;
      }
    }
  }

  /**
   * Count frames the decoder actually presented. requestVideoFrameCallback
   * fires once per painted video frame, i.e. the clip's real frame rate. The
   * render loop spins far faster than that (rAF), so it must never be reported
   * as the source FPS.
   */
  _trackPresentedFrames() {
    const video = this.video;
    if (typeof video.requestVideoFrameCallback !== 'function') return;
    let previous = null;
    const onFrame = (_, metadata) => {
      const current = metadata?.presentedFrames;
      this.presentedFrames += previous !== null && Number.isFinite(current)
        ? Math.max(0, current - previous) : 1;
      previous = Number.isFinite(current) ? current : null;
      this._frameCbId = video.requestVideoFrameCallback(onFrame);
    };
    this._frameCbId = video.requestVideoFrameCallback(onFrame);
  }

  /** Stop every MediaStreamTrack and detach the element. Safe to call twice. */
  stop() {
    this._generation++;
    this._fileAbort?.abort();
    this._fileAbort = null;
    this._releaseStream();
    if (this.state !== CAMERA_STATE.ERROR) {
      this.state = CAMERA_STATE.IDLE;
      this.error = null;
    }
  }

  /** True while the active source (camera or video file) still delivers frames. */
  get hasLiveTrack() {
    if (this.state !== CAMERA_STATE.LIVE) return false;
    if (this.kind === SOURCE.VIDEO) return this.video.readyState >= 2;
    return Boolean(this.stream) &&
      this.stream.getVideoTracks().some((track) => track.readyState === 'live');
  }

  _releaseStream() {
    const video = this.video;
    if (this._frameCbId !== null && typeof video.cancelVideoFrameCallback === 'function') {
      video.cancelVideoFrameCallback(this._frameCbId);
    }
    this._frameCbId = null;
    this.presentedFrames = 0;
    this.audioBlocked = false;
    if (video.srcObject) {
      video.pause();
      video.srcObject = null;
    }
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
      this.stream = null;
    }
    if (video.getAttribute('src')) {
      video.pause();
      video.removeAttribute('src');
    }
    if (this.objectUrl) {
      URL.revokeObjectURL(this.objectUrl);
      this.objectUrl = null;
    }
    this.kind = SOURCE.NONE;
    this.fileName = null;
  }
}

/**
 * True when play() was refused because of the autoplay policy rather than
 * because the file itself is broken. Only these cases are worth retrying muted.
 */
function isAutoplayRefusal(error) {
  const name = error && error.name ? error.name : '';
  return name === 'NotAllowedError';
}

export function describeCameraError(error) {
  const name = error && error.name ? error.name : '';
  switch (name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Camera permission was denied. Allow camera access for this ' +
        'site in the browser, then press Start Camera again.';
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return 'No camera was found. Connect a webcam and press Start Camera again.';
    case 'NotReadableError':
    case 'TrackStartError':
      return 'The camera is busy or blocked by the operating system. Close ' +
        'other apps that use the camera and try again.';
    case 'OverconstrainedError':
      return 'This camera cannot provide the requested resolution ' +
        `(${CONFIG.camera.width}x${CONFIG.camera.height}).`;
    default:
      return 'Could not start the camera: ' +
        (error && error.message ? error.message : String(error));
  }
}

function waitForMedia(video, level, event, signal) {
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      clearTimeout(timer);
      video.removeEventListener(event, check);
      video.removeEventListener('loadeddata', check);
      video.removeEventListener('error', failed);
      signal.removeEventListener('abort', aborted);
    };
    const finish = error => { cleanup(); error ? reject(error) : resolve(); };
    const check = () => {
      if (video.readyState >= level && video.videoWidth > 0) finish();
    };
    const failed = () => finish(new Error('Video decode failed (media error ' +
      (video.error?.code || 'unknown') + '). Check the file codec.'));
    const aborted = () => finish(new DOMException('Source loading cancelled', 'AbortError'));
    const timer = setTimeout(() => finish(new Error('Timed out waiting for ' + event)), METADATA_TIMEOUT_MS);
    video.addEventListener(event, check);
    video.addEventListener('loadeddata', check);
    video.addEventListener('error', failed);
    signal.addEventListener('abort', aborted, {once:true});
    if (signal.aborted) aborted();
    else if (video.error) failed();
    else check(); // Also handles events that fired before listener registration.
  });
}

function boundedPlay(video, signal) {
  return new Promise((resolve, reject) => {
    const cleanup = () => { clearTimeout(timer); signal.removeEventListener('abort', aborted); };
    const finish = error => { cleanup(); error ? reject(error) : resolve(); };
    const aborted = () => finish(new DOMException('Playback cancelled', 'AbortError'));
    const timer = setTimeout(() => finish(new Error('Timed out waiting for video.play()')), METADATA_TIMEOUT_MS);
    signal.addEventListener('abort', aborted, {once:true});
    if (signal.aborted) { aborted(); return; }
    try { Promise.resolve(video.play()).then(() => finish(), finish); }
    catch (error) { finish(error); }
  });
}
