/**
 * Privacy stage: 5% padding, block pixelation and motion-aware hold support.
 *
 * Block pixelation, matching the Python prototype
 * (cv2.resize down with INTER_LINEAR -> cv2.resize up with INTER_NEAREST):
 *
 *   source region -> small offscreen canvas -> back into the frame
 *
 * The offscreen canvas is created once and grown when needed, so no canvas
 * object is allocated per frame or per person. No getImageData/putImageData is
 * used on the fast path - the optional 'retro' style only reads the small ROI.
 */

import { CONFIG, LIMITS } from './config.js';
import { expandBox } from './geometry.js';

const BAYER_BASE = [[0, 2], [3, 1]];
const DITHER_SIZES = { none: 2, bayer2: 2, bayer4: 4, bayer8: 8, ordered: 8, clustered: 4 };

/**
 * Expand a person box by PRIVACY_PADDING on all four sides and clamp it to the
 * frame, so a padded region can never fall outside the canvas.
 */
export function paddingBox(box, frame, settings = CONFIG.privacy) {
  return expandBox(box, frame.width, frame.height, settings.padding);
}

/** '#rrggbb' -> [r, g, b]; invalid entries fall back to black. */
function parseColor(value) {
  const match = /^#([0-9a-fA-F]{6})$/.exec(value || '');
  if (!match) return [0, 0, 0];
  const number = parseInt(match[1], 16);
  return [(number >> 16) & 255, (number >> 8) & 255, number & 255];
}

function buildDitherMatrix(size) {
  let matrix = BAYER_BASE.map((row) => row.slice());
  while (matrix.length < size) {
    const current = matrix.length;
    const next = [];
    for (let y = 0; y < current * 2; y += 1) {
      next.push(new Array(current * 2).fill(0));
    }
    for (let y = 0; y < current; y += 1) {
      for (let x = 0; x < current; x += 1) {
        const value = matrix[y][x];
        next[y][x] = 4 * value;
        next[y][x + current] = 4 * value + 2;
        next[y + current][x] = 4 * value + 3;
        next[y + current][x + current] = 4 * value + 1;
      }
    }
    matrix = next;
  }
  return matrix;
}

/** Palette + ordered dithering arrays, ported from prepare_pixel_style(). */
function buildStyle(palette, dither, strength) {
  const colors = new Float32Array(Math.max(2, palette.length) * 3);
  let count = 0;
  for (const value of palette.slice(0, 16)) {
    const [r, g, b] = parseColor(value);
    colors[count * 3] = r;
    colors[count * 3 + 1] = g;
    colors[count * 3 + 2] = b;
    count += 1;
  }
  const size = DITHER_SIZES[dither] || DITHER_SIZES.bayer4;
  const matrix = dither === 'clustered'
    ? [[12,5,6,13],[4,0,1,7],[11,3,2,8],[15,10,9,14]] : buildDitherMatrix(size);
  const offsets = new Float32Array(size * size);
  const amount = (strength / 100) * 255;
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const threshold = ((matrix[y][x] + 0.5) / (size * size)) - 0.5;
      offsets[y * size + x] = ['none', 'floyd', 'atkinson'].includes(dither) ? 0 : threshold * amount;
    }
  }
  return { colors, count, offsets, size, lut: buildLookup(colors, count) };
}

/**
 * 15-bit RGB (5 bits per channel) -> palette byte offset.
 *
 * 32768 entries, built once per palette change, so retro quantization costs one
 * array read per pixel instead of scanning the palette. 5-bit channels are the
 * classic retro depth: at this scale the extra banding is invisible in the
 * output, and it keeps the table small enough to stay cache-friendly.
 */
function buildLookup(colors, count) {
  const lut = new Uint16Array(32768);
  for (let r = 0; r < 32; r++) {
    // Centre of each 5-bit bucket: the 8-bit value it stands for.
    const cr = r * 8 + 4;
    for (let g = 0; g < 32; g++) {
      const cg = g * 8 + 4;
      for (let b = 0; b < 32; b++) {
        const cb = b * 8 + 4;
        let best = 0;
        let bestDistance = Infinity;
        for (let color = 0; color < count; color += 1) {
          const base = color * 3;
          const dr = cr - colors[base];
          const dg = cg - colors[base + 1];
          const db = cb - colors[base + 2];
          const distance = dr * dr + dg * dg + db * db;
          if (distance < bestDistance) {
            bestDistance = distance;
            best = base;
          }
        }
        lut[(r << 10) | (g << 5) | b] = best;
      }
    }
  }
  return lut;
}

export class Pixelator {
  /**
   * @param {object} [settings] live privacy settings (CONFIG.privacy by default)
   */
  constructor(settings = CONFIG.privacy) {
    this.settings = settings;
    this.work = null;
    this.workCtx = null;
    this.tile = null;
    this.tileCtx = null;
    this.frame = null;
    this.frameCtx = null;
    this.pending = null;
    this.style = null;
    this.styleKey = '';
    this.lastSmall = '—';
    this.regions = 0;
  }

  /** Effective pixel scale, clamped to the documented tuning range. */
  get scale() {
    const value = Number(this.settings.pixelScale);
    if (!Number.isFinite(value)) return LIMITS.pixelScale.min;
    return Math.min(LIMITS.pixelScale.max, Math.max(LIMITS.pixelScale.min, value));
  }

  /** Block size hint for the dashboard (1 / scale pixels per block edge). */
  get blockPixels() {
    return Math.round(1 / this.scale);
  }

  reset() {
    this.diffusionErrors = null;
    this.work = null;
    this.workCtx = null;
    this.tile = null;
    this.tileCtx = null;
    this.frame = null;
    this.frameCtx = null;
    this.pending = null;
    this.style = null;
    this.styleKey = '';
    this.regions = 0;
  }

  /**
   * Release the retro-only CPU buffers (frame snapshot + scratch tile).
   *
   * Called when leaving the retro style. The GPU `work` canvas is untouched on
   * purpose: retro never reads it, so the block path keeps its acceleration.
   */
  releaseRetroBuffers() {
    this.diffusionErrors = null;
    this.frame = null;
    this.frameCtx = null;
    this.tile = null;
    this.tileCtx = null;
    this.pending = null;
  }

  /**
   * Begin one output frame.
   *
   * Retro needs CPU-side pixels, so it snapshots the drawn frame ONCE here
   * instead of reading back every person box: one GPU->CPU copy per frame
   * rather than one per region, which is what made retro stutter badly.
   * Called from renderFrame() before any apply().
   * @param {CanvasRenderingContext2D} ctx the output context holding the frame
   */
  beginFrame(ctx) {
    if (this.settings.style !== 'retro') return;
    this.pending = [];
    const canvas = ctx.canvas;
    const frameCtx = this._ensureFrame(canvas.width, canvas.height);
    if (!frameCtx) return;
    frameCtx.imageSmoothingEnabled = false;
    frameCtx.drawImage(canvas, 0, 0);
  }

  /**
   * Pixelate one padded region of `ctx.canvas` in place.
   * @param {CanvasRenderingContext2D} ctx destination (and source) context
   * @param {object} box already padded and clipped box
   * @returns {boolean} true when pixels were written
   */
  apply(ctx, box) {
    const width = box.x2 - box.x1;
    const height = box.y2 - box.y1;
    if (width <= 1 || height <= 1) return false;
    const scale = this.scale;
    const smallWidth = Math.max(1, Math.round(width * scale));
    const smallHeight = Math.max(1, Math.round(height * scale));
    if (this.settings.style === 'retro') {
      if (!this._applyRetro(ctx, box, width, height, smallWidth, smallHeight)) return false;
    } else {
      const workCtx = this._ensureWork(smallWidth, smallHeight);
      if (!workCtx) return false;
      // Downscale with smoothing, upscale with hard blocks: the pixel-art look of
      // resize(INTER_LINEAR) + resize(INTER_NEAREST) in the Python prototype.
      workCtx.imageSmoothingEnabled = true;
      workCtx.drawImage(ctx.canvas, box.x1, box.y1, width, height,
        0, 0, smallWidth, smallHeight);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(this.work, 0, 0, smallWidth, smallHeight,
        box.x1, box.y1, width, height);
      ctx.imageSmoothingEnabled = true;
    }
    this.lastSmall = `${smallWidth}x${smallHeight}`;
    this.regions += 1;
    return true;
  }

  /**
   * Retro step: record the region for this frame.
   *
   * Nothing is read or written here. Every getImageData/putImageData pair costs
   * a fixed synchronisation on the canvas, so doing one pair per person (15+ per
   * frame) dominated retro's frame time. Regions are collected and rendered in
   * one batch by flushFrame(), which needs a single read + write per frame.
   *
   * Called without beginFrame() it still behaves like the old single-region
   * apply(), so the self-test and any one-off caller keep working.
   */
  _applyRetro(ctx, box, width, height, smallWidth, smallHeight) {
    const immediate = !this.pending;
    if (immediate) this.pending = [];
    this.pending.push({box, width, height, w: smallWidth, h: smallHeight});
    this.lastSmall = `${smallWidth}x${smallHeight}`;
    if (immediate) this._flushRetro(ctx);
    return true;
  }

  /**
   * Render every retro region collected for this frame in one batch.
   *
   * The tiles are packed side by side into one small canvas, read once,
   * quantised in place and written back once. That turns 2N canvas
   * synchronisations into 2, whatever the number of people in frame.
   */
  flushFrame(ctx) {
    if (this.settings.style === 'retro') this._flushRetro(ctx);
  }

  _flushRetro(ctx) {
    const tiles = this.pending;
    if (!tiles || !tiles.length) return;
    this.pending = null;
    // No snapshot (self-test or direct use): fall back to the live canvas.
    const source = this.frameCtx ? this.frame : ctx.canvas;
    // Pack tiles into rows as wide as the frame allows.
    let x = 0;
    let y = 0;
    let rowHeight = 0;
    let usedWidth = 0;
    for (const tile of tiles) {
      if (x + tile.w > source.width && x > 0) {
        x = 0;
        y += rowHeight;
        rowHeight = 0;
      }
      tile.ax = x;
      tile.ay = y;
      x += tile.w;
      usedWidth = Math.max(usedWidth, x);
      rowHeight = Math.max(rowHeight, tile.h);
    }
    const atlasWidth = usedWidth;
    const atlasHeight = y + rowHeight;
    const tileCtx = this._ensureTile(atlasWidth, atlasHeight);
    if (!tileCtx) return;
    tileCtx.imageSmoothingEnabled = true;
    for (const tile of tiles) {
      tileCtx.drawImage(source, tile.box.x1, tile.box.y1, tile.width, tile.height,
        tile.ax, tile.ay, tile.w, tile.h);
    }
    const image = tileCtx.getImageData(0, 0, atlasWidth, atlasHeight);
    this._quantize(image.data, atlasWidth, tiles);
    tileCtx.putImageData(image, 0, 0);
    ctx.imageSmoothingEnabled = false;
    for (const tile of tiles) {
      ctx.drawImage(this.tile, tile.ax, tile.ay, tile.w, tile.h,
        tile.box.x1, tile.box.y1, tile.width, tile.height);
    }
    ctx.imageSmoothingEnabled = true;
  }

  _ensureWork(width, height) {
    if (!this.work) {
      this.work = document.createElement('canvas');
      this.workCtx = this.work.getContext('2d', { alpha: false });
      if (!this.workCtx) return null;
    }
    // Grow only: resizing reallocates the backing store, so never do it per frame.
    if (this.work.width < width || this.work.height < height) {
      this.work.width = Math.max(this.work.width, width);
      this.work.height = Math.max(this.work.height, height);
    }
    return this.workCtx;
  }

  /**
   * Small CPU-backed scratch tile for retro quantisation.
   *
   * willReadFrequently is mandatory here, not an optimisation hint: it tells
   * Chrome up front to keep this canvas in system memory. A canvas that is GPU
   * accelerated and only read later via getImageData() gets its acceleration
   * dropped - permanently - the first time it is read, which silently slowed
   * down every later frame of the block path too.
   */
  _ensureTile(width, height) {
    if (!this.tile) {
      this.tile = document.createElement('canvas');
      this.tileCtx = this.tile.getContext('2d',
        { alpha: false, willReadFrequently: true });
      if (!this.tileCtx) return null;
    }
    if (this.tile.width < width || this.tile.height < height) {
      this.tile.width = Math.max(this.tile.width, width);
      this.tile.height = Math.max(this.tile.height, height);
    }
    return this.tileCtx;
  }

  /**
   * One CPU-side copy of the current frame, so retro reads pixels once per
   * frame instead of once per region. Same willReadFrequently reasoning as
   * _ensureTile(); sized to the frame, not to a person box.
   */
  _ensureFrame(width, height) {
    if (!this.frame) {
      this.frame = document.createElement('canvas');
      this.frameCtx = this.frame.getContext('2d',
        { alpha: false, willReadFrequently: true });
      if (!this.frameCtx) return null;
    }
    if (this.frame.width !== width || this.frame.height !== height) {
      this.frame.width = width;
      this.frame.height = height;
    }
    return this.frameCtx;
  }

  _style() {
    const { dither, ditherStrength, palette } = this.settings;
    const key = `${dither}|${ditherStrength}|${palette.join(',')}`;
    if (key !== this.styleKey) {
      this.style = buildStyle(palette, dither, ditherStrength);
      this.styleKey = key;
    }
    return this.style;
  }

  /**
   * Palette quantization + ordered dithering over the batched tile atlas.
   *
   * Nearest-colour used to scan the whole palette per pixel, which dominated
   * retro's cost. The lookup table maps a 15-bit RGB (5 bits per channel, the
   * usual retro depth) straight to a palette entry, so each pixel costs one
   * array read instead of N squared-distance tests. The table is built once per
   * palette/dither change, never per frame.
   * @param {Uint8ClampedArray} data the atlas pixels, edited in place
   * @param {number} stride atlas width in pixels
   * @param {Array<object>} tiles packed tile rects (ax/ay/w/h) to quantise
   */
  _quantize(data, stride, tiles) {
    const { colors, offsets, size, lut } = this._style();
    const method = this.settings.dither;
    const diffusion = method === 'floyd' || method === 'atkinson';
    const amount = Math.max(0, Math.min(1, this.settings.ditherStrength / 100));
    const taps = method === 'atkinson'
      ? [[1,0,1/8],[2,0,1/8],[-1,1,1/8],[0,1,1/8],[1,1,1/8],[0,2,1/8]]
      : [[1,0,7/16],[-1,1,3/16],[0,1,5/16],[1,1,1/16]];
    const clamp = value => Math.max(0, Math.min(255, value));
    for (const tile of tiles) {
      const length = tile.w * tile.h * 3;
      if (diffusion) {
        if (!this.diffusionErrors || this.diffusionErrors.length < length) this.diffusionErrors = new Float32Array(length);
        this.diffusionErrors.fill(0, 0, length); // Never diffuse errors between people.
      }
      for (let ty = 0; ty < tile.h; ty += 1) {
        const rowOffset = (ty % size) * size;
        let index = ((tile.ay + ty) * stride + tile.ax) * 4;
        for (let tx = 0; tx < tile.w; tx += 1) {
          const offset = offsets[rowOffset + (tx % size)];
          // Dithering perturbs the colour, so clamp before it feeds the LUT.
          const ei = (ty * tile.w + tx) * 3;
          const r = clamp(data[index] + offset + (diffusion ? this.diffusionErrors[ei] : 0));
          const g = clamp(data[index + 1] + offset + (diffusion ? this.diffusionErrors[ei + 1] : 0));
          const b = clamp(data[index + 2] + offset + (diffusion ? this.diffusionErrors[ei + 2] : 0));
          const key = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3);
          const best = lut[key];
          if (diffusion) {
            for (const [dx,dy,weight] of taps) {
              const nx = tx + dx, ny = ty + dy;
              if (nx < 0 || nx >= tile.w || ny >= tile.h) continue;
              const next = (ny * tile.w + nx) * 3;
              this.diffusionErrors[next] += (r - colors[best]) * weight * amount;
              this.diffusionErrors[next + 1] += (g - colors[best + 1]) * weight * amount;
              this.diffusionErrors[next + 2] += (b - colors[best + 2]) * weight * amount;
            }
          }
          data[index] = colors[best];
          data[index + 1] = colors[best + 1];
          data[index + 2] = colors[best + 2];
          index += 4;
        }
      }
    }
  }
}
