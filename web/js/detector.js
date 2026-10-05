import { CONFIG } from './config.js';
import { preprocess, decodeDetections } from './yolox.js';
export const MODEL_LABEL = 'YOLOX-S MOT17';

/** A wrong *.mjs content type looks like a runtime crash; name the real cause. */
const MIME_HINT = ' The ONNX Runtime files must be served as text/javascript ' +
  '(*.mjs) and application/wasm (*.wasm) - try "py tools/serve_web.py".';
/**
 * Reject if `promise` has not settled within `timeoutMs`.
 *
 * Without this a stalled model download (or a WebGPU adapter that never
 * resolves) leaves the page claiming to load forever: no error, no timeout,
 * and the Start button silently useless because there is no detector.
 * `onTimeout` aborts whatever is still in flight.
 */
function withDeadline(promise, timeoutMs, onTimeout) {
  const limit = Number(timeoutMs);
  if (!(limit > 0)) return promise;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      if (onTimeout) onTimeout();
      reject(new Error(
        `Loading the YOLOX model timed out after ${Math.round(limit / 1000)}s. ` +
        'Check that the server is still running and that ' +
        './models/yolox_s_mot17.onnx is reachable, then press Retry Model.'));
    }, limit);
    promise.then(
      (value) => { clearTimeout(timer); resolve(value); },
      (error) => { clearTimeout(timer); reject(error); });
  });
}

export class PersonDetector {
  constructor(config = CONFIG.detector) {
    Object.assign(this, {config, session:null, backend:'loading', loadError:null,
      inferenceCount:0, inferenceMs:0, sessionCount:0, inFlight:0, maxInFlight:0,
      _loading:null, _abortLoad:null});
    this.buffer = new Float32Array(3 * config.height * config.width);
  }
  get loaded() { return Boolean(this.session); }
  load() {
    if (this.session) return Promise.resolve(this);
    if (this._loading) return this._loading;
    // The in-flight attempt is bounded: fetch() and a WebGPU adapter request
    // can both stall indefinitely, and an unbounded await here is exactly what
    // leaves the dashboard sitting on "Loading the model" forever.
    // __PROBE_TIMEOUT is a test seam: tests/smoke_web.py shortens the deadline
    // so the timeout path can be exercised without waiting the full 30s.
    const deadline = globalThis.__PROBE_TIMEOUT || this.config.loadTimeoutMs;
    this.loadError = null;
    const controller = new AbortController();
    // TEMP-DIAG-2026-10-26: model-loading vs page-ready ordering probe.
    // console.log('detector initialization started');
    const attempt = withDeadline(this._load(controller.signal), deadline,
      () => controller.abort());
    // A failed attempt is deliberately not cached. Keeping the rejected promise
    // in this._loading made the first hiccup permanent for the whole tab: every
    // later call re-threw the same stale error and a retry could never succeed.
    this._loading = attempt.then(() => { console.log('detector ready', this.backend); return this; }, (error) => {
      this._loading = null;
      const message = error && error.message ? error.message : String(error);
      this.loadError = /initWasm|no available backend/i.test(message)
        ? message + MIME_HINT
        : message;
      throw error;
    });
    return this._loading;
  }
  async _load(signal) {
    const ort = globalThis.ort;
    if (!ort) throw new Error('ONNX Runtime missing. Run python tools/setup_ort_web.py.');
    // Keep the page console clean: ORT routes its own warnings to console.error.
    ort.env.logLevel = 'error';
    ort.env.wasm.wasmPaths = new URL('./vendor/onnxruntime/', document.baseURI).href;
    ort.env.wasm.numThreads = 1;
    // Abort the transfer if the deadline hits, so a stalled download does not
    // keep a socket (and the browser's network panel) open for nothing.

    let model;
    try {
      const response = await fetch(this.config.modelUrl, {signal});
      if (!response.ok) throw new Error('YOLOX model missing. Run python tools/export_yolox_onnx.py.');
      model = new Uint8Array(await response.arrayBuffer());
    } finally {
      this._abortLoad = null;
    }
    let session = null;
    let backend;
    if (signal.aborted) throw new Error('Model initialization cancelled');
    if (navigator.gpu && this.config.preferWebGPU) {
      try {
        session = await ort.InferenceSession.create(model, { executionProviders: ['webgpu'] });
        backend = 'webgpu';
      } catch (error) { this.fallbackReason = String(error); }
    }
    if (!session && !signal.aborted) {
      session = await ort.InferenceSession.create(model, { executionProviders: ['wasm'] });
      backend = 'wasm';
    }
    if (signal.aborted) {
      await session?.release();
      throw new Error('Model initialization cancelled');
    }
    this.session = session;
    this.backend = backend;
    this.sessionCount++;
    return this;
  }
  async detect(imageData) {
    if (this.inFlight) throw new Error('Concurrent inference is not allowed');
    this.inFlight++;
    this.maxInFlight = Math.max(this.maxInFlight, this.inFlight);
    const start = performance.now();
    let input, result;
    try {
      await this.load();
      const { data, ratio } = preprocess(imageData, this.config, this.buffer);
      input = new ort.Tensor('float32', data, [1, 3, this.config.height, this.config.width]);
      result = await this.session.run({ images: input });
      const output = result.output;
      if (output.dims.join(',') !== '1,10710,6') throw new Error('Unexpected YOLOX output: ' + output.dims);
      const boxes = decodeDetections(output.data, ratio, this.config);
      this.inferenceCount++;
      this.inferenceMs = performance.now() - start;
      return boxes;
    } finally {
      input?.dispose();
      if (result) for (const tensor of Object.values(result)) tensor.dispose();
      this.inFlight--;
    }
  }
}

