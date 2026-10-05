/**
 * Box maths shared by the tracker, the stable ID manager and the privacy stage.
 *
 * Ported from app/privacy_engine.py so the browser pipeline keeps the exact
 * clipping / padding / association behaviour of the Python prototype.
 * Every box is {x1, y1, x2, y2, score} in source-frame pixels.
 */

const round = Math.round;

export function clampInt(value, low, high) {
  const rounded = round(value);
  if (!Number.isFinite(rounded)) return low;
  return rounded < low ? low : (rounded > high ? high : rounded);
}

/** Python clip_box: x1/y1 inside the frame, x2/y2 allowed on the last edge. */
export function clipBox(box, width, height) {
  return {
    x1: clampInt(box.x1, 0, width - 1),
    y1: clampInt(box.y1, 0, height - 1),
    x2: clampInt(box.x2, 0, width),
    y2: clampInt(box.y2, 0, height),
    score: box.score || 0,
  };
}

/** Python expand_box: symetric padding based on the box size, then clipped. */
export function expandBox(box, width, height, padding) {
  const boxWidth = box.x2 - box.x1;
  const boxHeight = box.y2 - box.y1;
  const padX = boxWidth * padding;
  const padY = boxHeight * padding;
  return clipBox({
    x1: box.x1 - padX,
    y1: box.y1 - padY,
    x2: box.x2 + padX,
    y2: box.y2 + padY,
    score: box.score || 0,
  }, width, height);
}

export function shiftBox(box, dx, dy, width, height) {
  return clipBox({
    x1: box.x1 + dx,
    y1: box.y1 + dy,
    x2: box.x2 + dx,
    y2: box.y2 + dy,
    score: box.score || 0,
  }, width, height);
}

export function boxWidth(box) {
  return box.x2 - box.x1;
}

export function boxHeight(box) {
  return box.y2 - box.y1;
}

export function boxArea(box) {
  return Math.max(0, box.x2 - box.x1) * Math.max(0, box.y2 - box.y1);
}

export function boxIou(boxA, boxB) {
  const interX1 = Math.max(boxA.x1, boxB.x1);
  const interY1 = Math.max(boxA.y1, boxB.y1);
  const interX2 = Math.min(boxA.x2, boxB.x2);
  const interY2 = Math.min(boxA.y2, boxB.y2);
  const inter = Math.max(0, interX2 - interX1) * Math.max(0, interY2 - interY1);
  const union = boxArea(boxA) + boxArea(boxB) - inter;
  return union <= 0 ? 0 : inter / union;
}

export function boxCenter(box) {
  return { x: (box.x1 + box.x2) / 2, y: (box.y1 + box.y2) / 2 };
}

export function centerDistance(boxA, boxB) {
  const a = boxCenter(boxA);
  const b = boxCenter(boxB);
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/** Longest box side, used to scale the center-distance gate. */
export function boxScale(box) {
  return Math.max(boxWidth(box), boxHeight(box), 1);
}

/**
 * Detector post-filter, mirrors the prototype's validity check:
 * positive size, minimum area and the person aspect ratio.
 */
export function isUsableBox(box, config) {
  const width = boxWidth(box);
  const height = boxHeight(box);
  if (width <= 1 || height <= 1) return false;
  if (width * height <= config.minBoxArea) return false;
  return width / height <= config.maxAspectRatio;
}
