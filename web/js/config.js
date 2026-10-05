/**
 * Central tunables for the browser-local privacy pipeline.
 *
 * Every value that used to be a Python constant (see app/privacy_engine.py)
 * lives here instead of being hard-coded inside a module. CONFIG.privacy is the
 * live settings object: the dashboard tuning panel mutates it, no other
 * behaviour depends on where the values came from.
 *
 * Boxes are always passed around as {x1, y1, x2, y2, score} in source-frame
 * pixels. That is the only box format used by this application.
 */

export const CONFIG = {
  /** getUserMedia constraints. Frames stay inside the browser. */
  camera: {
    width: 1280,
    height: 720,
    frameRate: 30,
  },

  /** Original YOLOX-S MOT17, fixed FP32 graph shared by WebGPU and WASM. */
  detector: {
    modelUrl: './models/yolox_s_mot17.onnx',
    width: 960, height: 544,
    mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225],
    strides: [8, 16, 32], minScore: 0.05, nmsThreshold: 0.7,
    preferWebGPU: true,
    // Hard deadline for one model-load attempt (fetch + session creation). A
    // stalled download or a WebGPU adapter that never answers would otherwise
    // leave the dashboard on "Loading the model" forever with no error.
    loadTimeoutMs: 30000,
    // Minimum gap between inference starts (~15 FPS cap). Combined with the
    // latest-frame skip in the render loop, a slow detector never queues frames.
    intervalMs: 66,
  },
  /** ByteTrack thresholds match yolox/tracker/byte_tracker.py. */
  tracker: {
    trackThresh: 0.50, trackBuffer: 30, matchThresh: 0.80,
    lowThresh: 0.1, secondMatchThresh: 0.5, unconfirmedMatchThresh: 0.7,
    newTrackMargin: 0.1, duplicateDistance: 0.15,
    maxAspectRatio: 1.6, minBoxArea: 10,
  },
  /** Stable ID: short-term geometric re-association, NOT person re-identification. */
  stableId: {
    enabled: true,
    maxGap: 30,
    minIou: 0.15,
    maxCenterDistance: 120,
  },

  /** Privacy stage: 5% padding, block pixelation, motion-aware hold. */
  privacy: {
    enabled: true,
    padding: 0.05,     // Python prototype default; validated on WildTrack
    pixelScale: 0.08,  // Python prototype default
    useMotionHold: true,
    holdFrames: 5,     // Pixelate up to this many detection ticks after loss
    style: 'block',    // 'block' (fast path) or 'retro' (palette + Bayer dither)
    palette: [
      '#1b1b1e', '#f4f1de', '#e07a5f', '#3d405b',
      '#81b29a', '#f2cc8f', '#8d5a97', '#ef3054',
    ],
    dither: 'bayer4',
    ditherStrength: 20,
  },

  /** Dashboard metrics. */
  metrics: {
    // smoothedFPS = 0.1 * currentFPS + 0.9 * smoothedFPS
    fpsSmoothing: 0.1,
    sampleMs: 500,
    uiUpdateMs: 500,
  },

  /** Rendering. */
  renderer: {
    mirror: false,        // Mirrored preview; labels stay readable either way
    overlayLineWidth: 2,
    labelFontSize: 15,
  },
};

/** Tuning limits used by both the dashboard panel and the modules. */
export const LIMITS = {
  padding: { min: 0, max: 0.30, step: 0.01 },
  pixelScale: { min: 0.02, max: 0.50, step: 0.01 },
  holdFrames: { min: 0, max: 30, step: 1 },
};
