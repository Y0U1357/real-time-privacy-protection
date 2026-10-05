/**
 * Canvas output. The <video> element is only the camera source; everything the
 * user looks at is drawn here. When privacy is ON the main canvas only ever
 * shows pixelated people.
 */

import { CONFIG } from './config.js';

export const OVERLAY_ACTIVE = '#00ff00';
export const OVERLAY_HOLD = '#ffa500';

export class Renderer {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {object} [options] rendering options (CONFIG.renderer by default)
   */
  constructor(canvas, options = CONFIG.renderer) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { alpha: false });
    this.options = options;
    this.mirror = Boolean(options.mirror);
    this.width = canvas.width;
    this.height = canvas.height;
    this._labelWidths = new Map();
  }

  /** Match the canvas backing store to the frame. Only writes when it changes. */
  resize(width, height) {
    if (width <= 0 || height <= 0) return false;
    if (this.canvas.width === width && this.canvas.height === height) return false;
    this.canvas.width = width;
    this.canvas.height = height;
    this.width = width;
    this.height = height;
    this._labelWidths.clear();
    return true;
  }

  clear(color = '#05080e') {
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.fillStyle = color;
    this.ctx.fillRect(0, 0, this.width, this.height);
  }

  /** Draw one raw camera/video frame. Used for the labelled comparison canvas. */
  drawVideo(video) {
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.imageSmoothingEnabled = true;
    if (this.mirror) {
      ctx.translate(this.width, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(video, 0, 0, this.width, this.height);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  /** Source-frame box -> canvas box (mirroring flips x). */
  toCanvasBox(box) {
    if (!this.mirror) return box;
    return {
      x1: this.width - box.x2,
      y1: box.y1,
      x2: this.width - box.x1,
      y2: box.y2,
      score: box.score,
    };
  }

  /**
   * Draw ID boxes and labels. Labels are always drawn unmirrored so they stay
   * readable in mirrored mode.
   * @param {Array<{box:object,label:string,color:string}>} regions
   */
  drawOverlays(regions) {
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.lineWidth = this.options.overlayLineWidth;
    ctx.font = `${this.options.labelFontSize}px "Segoe UI", system-ui, sans-serif`;
    ctx.textBaseline = 'alphabetic';
    for (const region of regions) {
      const box = this.toCanvasBox(region.box);
      const width = box.x2 - box.x1;
      const height = box.y2 - box.y1;
      if (width < 2 || height < 2) continue;
      ctx.strokeStyle = region.color;
      ctx.strokeRect(box.x1, box.y1, width, height);
      if (!region.label) continue;
      const textWidth = this._measure(region.label);
      const textHeight = this.options.labelFontSize + 6;
      const labelX = Math.max(0, Math.min(box.x1, this.width - textWidth - 10));
      const labelY = box.y1 - textHeight >= 0 ? box.y1 - textHeight : box.y1;
      ctx.fillStyle = 'rgba(5, 8, 14, 0.72)';
      ctx.fillRect(labelX, labelY, textWidth + 10, textHeight);
      ctx.fillStyle = region.color;
      ctx.fillText(region.label, labelX + 5, labelY + textHeight - 5);
    }
  }

  _measure(text) {
    let width = this._labelWidths.get(text);
    if (width === undefined) {
      width = Math.ceil(this.ctx.measureText(text).width);
      if (this._labelWidths.size > 512) this._labelWidths.clear();
      this._labelWidths.set(text, width);
    }
    return width;
  }
}
