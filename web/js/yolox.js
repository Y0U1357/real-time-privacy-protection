// Pure YOLOX preprocessing/postprocessing, shared by browser and parity tests.
import { boxIou } from './geometry.js';
export function preprocess(image, config, output = new Float32Array(3 * config.height * config.width)) {
  const { width: iw, height: ih, data } = image;
  const { width: w, height: h, mean, std } = config;
  const ratio = Math.min(h / ih, w / iw);
  const rw = Math.floor(iw * ratio), rh = Math.floor(ih * ratio), plane = w * h;
  for (let c = 0; c < 3; c++) output.fill((114 / 255 - mean[c]) / std[c], c * plane, (c + 1) * plane);
  // OpenCV pixel-center bilinear sampling; uint8 rounding can differ by one.
  for (let y = 0; y < rh; y++) {
    const sy = Math.max(0, Math.min(ih - 1, (y + 0.5) * ih / rh - 0.5));
    const y0 = Math.floor(sy), y1 = Math.min(ih - 1, y0 + 1), fy = sy - y0;
    for (let x = 0; x < rw; x++) {
      const sx = Math.max(0, Math.min(iw - 1, (x + 0.5) * iw / rw - 0.5));
      const x0 = Math.floor(sx), x1 = Math.min(iw - 1, x0 + 1), fx = sx - x0;
      for (let c = 0; c < 3; c++) {
        const top = data[(y0 * iw + x0) * 4 + c] * (1 - fx) + data[(y0 * iw + x1) * 4 + c] * fx;
        const bottom = data[(y1 * iw + x0) * 4 + c] * (1 - fx) + data[(y1 * iw + x1) * 4 + c] * fx;
        output[c * plane + y * w + x] = (Math.round(top * (1 - fy) + bottom * fy) / 255 - mean[c]) / std[c];
      }
    }
  }
  return { data: output, ratio };
}
export function decodeDetections(raw, ratio, config) {
  // The ONNX graph decodes the head (decode_in_inference stays on), so a row is
  // [cx, cy, w, h, obj, person] in network pixels - no stride grid, no exp().
  const candidates = [];
  const rows = raw.length / 6;
  for (let row = 0; row < rows; row += 1) {
    const i = row * 6;
    const score = raw[i + 4] * raw[i + 5];
    if (score < config.minScore) continue;
    const w = raw[i + 2] / ratio;
    const h = raw[i + 3] / ratio;
    if (!(w > 0) || !(h > 0)) continue;
    const cx = raw[i] / ratio;
    const cy = raw[i + 1] / ratio;
    if (![cx, cy, score].every(Number.isFinite)) continue;
    candidates.push({ x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, score });
  }
  candidates.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const box of candidates) if (!kept.some(other => boxIou(box, other) > config.nmsThreshold)) kept.push(box);
  return kept;
}
