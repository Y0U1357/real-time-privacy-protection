/**
 * Deterministic self-test for the privacy pipeline. It imports the same
 * modules app.js uses and drives them with synthetic detections, so the core
 * behaviour (padding, clipping, block pixelation, tracking, stable IDs,
 * overlays) can be verified without a camera, a model or a network.
 *
 * The page renders PASS/FAIL rows and sets <body data-status="passed|failed">,
 * which tests/test_web.py greps when it runs this page headlessly.
 */

import { CONFIG } from './config.js';
import { expandBox, clipBox, isUsableBox, boxIou } from './geometry.js';
import { Tracker, predictBox, advanceBox } from './tracker.js';
import { StableIdManager } from './stable-id.js';
import { Pixelator, paddingBox } from './privacy.js';
import { Renderer, OVERLAY_ACTIVE } from './renderer.js';

const frame = { width: 640, height: 480 };
const results = [];

function test(name, fn) {
  try {
    fn();
    results.push({ name, ok: true });
  } catch (error) {
    results.push({ name, ok: false, error: String(error && error.message || error) });
  }
}

function expect(condition, message) {
  if (!condition) throw new Error(message);
}

function equal(actual, expected, message) {
  if (actual !== expected) {
    throw new Error(`${message} — expected ${JSON.stringify(expected)}, ` +
      `got ${JSON.stringify(actual)}`);
  }
}

function box(x1, y1, x2, y2, score = 0.9) {
  return { x1, y1, x2, y2, score };
}

function move(b, dx) {
  return box(b.x1 + dx, b.y1, b.x2 + dx, b.y2, b.score);
}

/** Deterministic noise canvas: every pixel a different colour. */
function noiseCanvas(width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(width, height);
  let seed = 12345;
  for (let i = 0; i < image.data.length; i += 4) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    image.data[i] = seed & 255;
    image.data[i + 1] = (seed >> 8) & 255;
    image.data[i + 2] = (seed >> 16) & 255;
    image.data[i + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  return canvas;
}

function distinctColors(ctx, region) {
  const width = region.x2 - region.x1;
  const height = region.y2 - region.y1;
  const data = ctx.getImageData(region.x1, region.y1, width, height).data;
  const seen = new Set();
  for (let i = 0; i < data.length; i += 4) {
    seen.add((data[i] << 16) | (data[i + 1] << 8) | data[i + 2]);
  }
  return seen.size;
}

function pixelAt(ctx, x, y) {
  const data = ctx.getImageData(x, y, 1, 1).data;
  return [data[0], data[1], data[2]];
}

/* ------------------------------------------------------------------ */
test('padding expands by 5% of the box size', () => {
  const padded = paddingBox(box(100, 100, 200, 300), frame);
  // padX = 100 * .05 = 5, padY = 200 * .05 = 10
  equal(JSON.stringify(padded), JSON.stringify(box(95, 90, 205, 310)),
    'paddingBox result');
});

test('padding is clipped to the frame', () => {
  const nearEdge = paddingBox(box(2, 4, 60, 80), frame);
  expect(nearEdge.x1 >= 0 && nearEdge.y1 >= 0, 'clipped origin stays inside');
  expect(nearEdge.x2 <= frame.width && nearEdge.y2 <= frame.height,
    'padded end never exceeds the frame');
  const far = expandBox(box(-50, -50, 10, 10), frame.width, frame.height, 0.1);
  equal(far.x1, 0, 'x1 clamped to 0');
  equal(far.y1, 0, 'y1 clamped to 0');
  equal(clipBox(box(600, 400, 9999, 9999), 640, 480).x2, 640, 'x2 to frame width');
});

test('detector validity filter matches the prototype', () => {
  expect(isUsableBox(box(100, 50, 180, 300), CONFIG.tracker), 'person box accepted');
  expect(!isUsableBox(box(100, 50, 400, 200), CONFIG.tracker),
    'aspect ratio 2.5 rejected');
  expect(!isUsableBox(box(100, 50, 102, 52), CONFIG.tracker), 'tiny box rejected');
  expect(boxIou(box(0, 0, 10, 10), box(0, 0, 10, 10)) === 1, 'identical IoU = 1');
});

test('block pixelation crushes noise and keeps exact blocks', () => {
  const canvas = noiseCanvas(320, 320);
  const ctx = canvas.getContext('2d');
  const region = box(40, 40, 190, 200); // 150x160 -> at scale .1 => 15x16 blocks
  const before = distinctColors(ctx, region);
  const settings = { ...CONFIG.privacy, pixelScale: 0.1, style: 'block' };
  const pixelator = new Pixelator(settings);
  expect(pixelator.apply(ctx, region), 'apply() reports pixels written');
  equal(pixelator.regions, 1, 'region counter');
  expect(/^\d+x\d+$/.test(pixelator.lastSmall), 'small size reported');
  const after = distinctColors(ctx, region);
  expect(before > 500, `noise has many colours (got ${before})`);
  expect(after <= 15 * 16, `blocky region has at most one colour per block (got ${after})`);

  // Every 10x10 block must be a single flat colour.
  for (let by = 0; by < 16; by += 1) {
    for (let bx = 0; bx < 15; bx += 1) {
      const origin = pixelAt(ctx, region.x1 + bx * 10, region.y1 + by * 10);
      for (let dy = 0; dy < 10; dy += 1) {
        const row = pixelAt(ctx, region.x1 + bx * 10, region.y1 + by * 10 + dy);
        for (let dx = 0; dx < 10; dx += 1) {
          const probe = pixelAt(ctx, region.x1 + bx * 10 + dx,
            region.y1 + by * 10 + dy);
          equal(probe.join(','), origin.join(','),
            `block ${bx},${by} pixel ${dx},${dy}`);
          equal(row.join(','), origin.join(','), `block ${bx},${by} row ${dy}`);
        }
      }
    }
  }
  // The work canvas is reused instead of being recreated per region.
  const work = pixelator.work;
  pixelator.apply(ctx, box(210, 40, 280, 120));
  expect(pixelator.work === work, 'work canvas reused across regions');
  equal(pixelator.regions, 2, 'regions accumulate until reset()');
  pixelator.reset();
  equal(pixelator.regions, 0, 'reset() clears the counter');
});

test('ByteTrack keeps IDs, coasts lost tracks and drops them after the buffer', () => {
  const tracker = new Tracker();
  const person = box(100, 100, 160, 260);
  tracker.update([person], 1000, frame);
  equal(tracker.tracks.length, 1, 'one track created');
  const track = tracker.tracks[0];
  equal(track.id, 1, 'raw id');
  equal(track.hits, 1, 'hits after first detection');
  equal(track.confirmed, true, 'first frame activates the track (frame_id == 1)');

  tracker.update([move(person, 10)], 1066, frame);
  equal(track.hits, 2, 'hits after the matched detection');
  equal(track.confirmed, true, 'stays confirmed');
  expect(track.velocityX > 0, `Kalman velocity moves right (got ${track.velocityX})`);
  expect(track.box.x1 > person.x1 && track.box.x1 <= person.x1 + 10,
    `box pulled towards the detection (got ${track.box.x1})`);

  const beforeCoast = track.box.x1;
  tracker.update([], 1133, frame);
  equal(track.missingTicks, 1, 'missing tick counted');
  equal(track.state, 'lost', 'unmatched confirmed track turns lost');
  equal(tracker.tracks.length, 1, 'lost track is kept while coasting');
  expect(track.displayBox.x1 > beforeCoast, 'coasting advances with velocity');

  for (let tick = 4; tick <= 33; tick += 1) {
    tracker.update([], 1000 + tick * 66, frame);
  }
  equal(tracker.tracks.length, 0, 'track removed after trackBuffer ticks');
});

test('ByteTrack low batch keeps a track alive but never spawns tracks', () => {
  const tracker = new Tracker();
  const person = box(200, 100, 260, 260);
  tracker.update([person], 1000, frame);
  equal(tracker.tracks.length, 1, 'track created on the first frame');

  // Weak detection (0.3) on the same person matches via stage 2 (low batch).
  tracker.update([box(200, 100, 260, 260, 0.3)], 1066, frame);
  equal(tracker.tracks.length, 1, 'low-score detection reuses the track');
  equal(tracker.tracks[0].missingTicks, 0, 'track updated from the low batch');
  equal(tracker.tracks[0].state, 'tracked', 'still tracked');

  // A weak detection elsewhere must not create a new track.
  tracker.update([box(400, 100, 460, 260, 0.3)], 1133, frame);
  equal(tracker.tracks.length, 1, 'low-score leftovers never spawn tracks');
  equal(tracker.tracks[0].missingTicks, 1, 'person now coasting');
});

test('tracker starts a new ID for a distant person', () => {
  const tracker = new Tracker();
  const near = box(100, 100, 160, 260);
  tracker.update([near], 1000, frame);
  tracker.update([move(near, 10)], 1066, frame);
  tracker.update([move(near, 25)], 1133, frame);
  equal(tracker.tracks.length, 1, 'same person keeps one track');
  equal(tracker.tracks[0].hits, 3, 'hits keep growing');

  tracker.update([move(near, 25), box(400, 100, 460, 260)], 1199, frame);
  equal(tracker.tracks.length, 2, 'distant person gets its own track');
  equal(tracker.tracks[1].id, 2, 'new raw id');
});

test('render prediction advances the coasted box between detections', () => {
  const track = { displayBox: box(100, 100, 160, 260), velocityX: 10, velocityY: 4 };
  equal(predictBox(track, 0, frame).x1, 100, 'fraction 0 stays put');
  equal(predictBox(track, 0.5, frame).x1, 105, 'half interval');
  equal(predictBox(track, 2, frame).x1, 110, 'clamped to one detection tick');
  equal(advanceBox(track.displayBox, 10, 4, -1, frame).x1, 100, 'negative ignored');
  equal(predictBox(track, 1, frame).y1, 104, 'vertical velocity applied');
});

test('stable IDs survive a raw ID change when the position matches', () => {
  const manager = new StableIdManager();
  const active = new Set();
  const first = box(200, 100, 280, 300);
  equal(manager.update(1, first, 1, active, frame), 1, 'first stable id');
  equal(manager.update(1, move(first, 5), 2, active, frame), 1, 'same raw id');
  // Raw track dies (occlusion) and a new raw id appears one tick later nearby.
  equal(manager.update(2, move(first, 12), 3, active, frame), 1, 're-associated');
  equal(manager.rawToStable.get(2), 1, 'new raw id maps to the old stable id');
  expect(!manager.rawToStable.has(1), 'dead raw id is forgotten');

  const held = manager.getHoldBox(1, 4, frame);
  expect(held !== null, 'hold box exists while the gap is inside maxGap');
  expect(held.x1 > first.x1, `hold box follows the velocity (x1=${held.x1})`);

  equal(manager.removeExpired(3 + 31), 1, 'expired after maxGap ticks');
  equal(manager.size, 0, 'expired stable id forgotten');

  const disabled = new StableIdManager();
  disabled.enabled = false;
  equal(disabled.update(7, first, 1, active, frame), 1, 'disabled still numbers ids');
  equal(disabled.update(8, first, 2, active, frame), 2, 'disabled never re-associates');
});

test('renderer paints overlays and mirrors coordinates', () => {
  const canvas = document.createElement('canvas');
  canvas.width = 320;
  canvas.height = 180;
  const renderer = new Renderer(canvas, { ...CONFIG.renderer });
  renderer.clear('#112233');
  equal(pixelAt(renderer.ctx, 5, 5).join(','), '17,34,51', 'clear colour');
  renderer.drawOverlays([{
    box: box(50, 100, 150, 170),
    label: 'ID 7 0.90',
    color: OVERLAY_ACTIVE,
  }]);
  const edge = pixelAt(renderer.ctx, 100, 100);
  expect(edge[1] > 150 && edge[0] < 120, `top edge is green, got [${edge}]`);

  const mirrored = new Renderer(canvas, { ...CONFIG.renderer, mirror: true });
  const flipped = mirrored.toCanvasBox(box(10, 20, 30, 60));
  equal(JSON.stringify(flipped),
    JSON.stringify({ x1: 290, y1: 20, x2: 310, y2: 60, score: 0.9 }),
    'mirrored box');
});

test('privacy stage pixelates only the padded person region', () => {
  const canvas = noiseCanvas(320, 320);
  const renderer = new Renderer(canvas, { ...CONFIG.renderer });
  const padded = paddingBox(box(60, 40, 140, 240), { width: 320, height: 320 });
  const insideBefore = pixelAt(renderer.ctx, padded.x1 + 1, padded.y1 + 1);
  const outsideBefore = pixelAt(renderer.ctx, padded.x2 + 6, padded.y1 + 1);
  const pixelator = new Pixelator(CONFIG.privacy);
  expect(pixelator.apply(renderer.ctx, renderer.toCanvasBox(padded)),
    'region pixelated');
  const insideAfter = pixelAt(renderer.ctx, padded.x1 + 1, padded.y1 + 1);
  expect(insideAfter.join(',') !== insideBefore.join(','),
    'pixels inside the region changed');
  equal(pixelAt(renderer.ctx, padded.x2 + 6, padded.y1 + 1).join(','),
    outsideBefore.join(','), 'pixels outside the region untouched');
});

test('retro palette style runs on the small ROI only', () => {
  const canvas = noiseCanvas(200, 200);
  const ctx = canvas.getContext('2d');
  const region = box(20, 20, 140, 180);
  const before = distinctColors(ctx, region);
  const settings = { ...CONFIG.privacy, style: 'retro', pixelScale: 0.1 };
  const pixelator = new Pixelator(settings);
  pixelator.beginFrame(ctx);
  expect(pixelator.apply(ctx, region), 'retro apply()');
  pixelator.flushFrame(ctx);
  const after = distinctColors(ctx, region);
  expect(after < before / 2, `palette reduces colours ${after} < ${before / 2}`);
  expect(distinctColors(ctx, box(150, 20, 180, 60)) > 100,
    'region outside the ROI untouched');
});

test('retro batches many regions into one read/write per frame', () => {
  const settings = { ...CONFIG.privacy, style: 'retro', pixelScale: 0.1 };
  const pixelator = new Pixelator(settings);
  const canvas = noiseCanvas(400, 200);
  const ctx = canvas.getContext('2d');
  const regions = [box(10, 10, 90, 190), box(120, 10, 200, 190),
                   box(230, 10, 310, 190), box(330, 10, 395, 190)];
  pixelator.beginFrame(ctx);
  for (const region of regions) expect(pixelator.apply(ctx, region), 'region queued');
  equal(pixelator.pending.length, regions.length, 'all regions queued before flush');
  pixelator.flushFrame(ctx);
  equal(pixelator.pending, null, 'flush clears the queue');
  equal(pixelator.regions, regions.length, 'every queued region was counted');
  // Each region must be quantised to the palette, at its own downscale size.
  for (const region of regions) {
    const cols = Math.ceil((region.x2 - region.x1) * settings.pixelScale);
    const rows = Math.ceil((region.y2 - region.y1) * settings.pixelScale);
    expect(distinctColors(ctx, region) <= cols * rows,
      `region quantised to ${cols}x${rows} palette cells`);
  }
});

test('retro never reads the GPU work canvas, so block keeps its acceleration', () => {
  const settings = { ...CONFIG.privacy };
  const pixelator = new Pixelator(settings);
  // Block first: this is the GPU canvas that must stay GPU-backed for good.
  settings.style = 'block';
  const canvas = noiseCanvas(200, 200);
  const ctx = canvas.getContext('2d');
  expect(pixelator.apply(ctx, box(20, 20, 180, 180)), 'block apply()');
  const work = pixelator.work;
  expect(work instanceof HTMLCanvasElement, 'block allocates the work canvas');

  // Retro must run on its own CPU-backed buffers, never on `work`.
  settings.style = 'retro';
  pixelator.beginFrame(ctx);
  expect(pixelator.apply(ctx, box(20, 20, 180, 180)), 'retro apply()');
  expect(pixelator.work === work, 'retro does not replace the GPU work canvas');
  expect(pixelator.tile !== work, 'retro uses a separate scratch tile');
  expect(pixelator.frame !== work, 'retro keeps its own frame snapshot');

  // Back to block: the same GPU canvas is reused and the retro buffers are gone.
  settings.style = 'block';
  pixelator.releaseRetroBuffers();
  expect(pixelator.apply(ctx, box(20, 20, 180, 180)), 'block apply() again');
  expect(pixelator.work === work, 'block reuses the original work canvas');
  expect(pixelator.tile === null && pixelator.frame === null,
    'retro buffers released when leaving retro');
});

function renderResults() {
  const list = document.getElementById('results');
  for (const entry of results) {
    const item = document.createElement('li');
    item.className = entry.ok ? 'pass' : 'fail';
    item.textContent = `${entry.ok ? 'PASS' : 'FAIL'} · ${entry.name}` +
      (entry.ok ? '' : ` — ${entry.error}`);
    list.appendChild(item);
  }
  const failed = results.filter((entry) => !entry.ok).length;
  document.getElementById('summary').textContent =
    `${results.length - failed} passed · ${failed} failed`;
  document.body.dataset.status = failed === 0 ? 'passed' : 'failed';
}

renderResults();
