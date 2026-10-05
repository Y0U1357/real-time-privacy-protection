/**
 * Stable ID layer - a direct port of StableIDManager in app/privacy_engine.py.
 *
 * ByteTrack raw IDs change after occlusion or track recreation; a browser
 * tracker has the same problem. This layer keeps a stable display ID while a
 * person is only briefly lost:
 *
 *   1. known raw id          -> reuse its stable id
 *   2. unknown raw id        -> compare with recently lost stable tracks
 *   3. predicted position close enough -> reuse the old stable id
 *   4. otherwise             -> create a new stable id
 *
 * This is short-term geometric re-association. It is NOT person
 * re-identification: no face, no embedding, no ReID model, no history beyond
 * `maxGap` detection ticks.
 */

import { CONFIG } from './config.js';
import { boxIou, boxCenter, centerDistance, shiftBox } from './geometry.js';

export class StableIdManager {
  constructor(config = CONFIG.stableId) {
    this.config = config;
    this.enabled = config.enabled;
    this.nextStableId = 1;
    this.rawToStable = new Map();
    this.states = new Map();
  }

  /** Number of stable IDs currently remembered (held or active). */
  get size() {
    return this.states.size;
  }

  /** Forget all IDs; called together with Tracker.reset(). */
  reset() {
    this.nextStableId = 1;
    this.rawToStable.clear();
    this.states.clear();
  }

  /**
   * @param {number} rawId tracker id
   * @param {object} box observed box of that raw track
   * @param {number} tick current detection tick
   * @param {Set<number>} activeStableIds stable IDs already used this tick
   * @param {{width:number, height:number}} frame
   * @returns {number} stable id
   */
  update(rawId, box, tick, activeStableIds, frame) {
    if (this.rawToStable.has(rawId)) {
      const stableId = this.rawToStable.get(rawId);
      if (this.states.has(stableId)) {
        this._updateState(stableId, rawId, box, tick);
        return stableId;
      }
    }
    let stableId = null;
    if (this.enabled) {
      stableId = this._findReassociation(box, tick, activeStableIds, frame);
    }
    if (stableId === null) {
      return this._createStableId(rawId, box, tick);
    }
    // Reuse the old stable id for a newly created raw id.
    const oldRawId = this.states.get(stableId).rawId;
    if (this.rawToStable.get(oldRawId) === stableId) this.rawToStable.delete(oldRawId);
    this.rawToStable.set(rawId, stableId);
    this._updateState(stableId, rawId, box, tick);
    return stableId;
  }

  /**
   * Predicted box of a stable ID: last observed box shifted by its velocity
   * over the ticks it has been missing.
   */
  getHoldBox(stableId, tick, frame) {
    const state = this.states.get(stableId);
    if (!state) return null;
    return shiftBox(state.box, state.velocity.x * (tick - state.lastSeen),
      state.velocity.y * (tick - state.lastSeen), frame.width, frame.height);
  }

  /** Drop stable IDs that have been missing for more than maxGap ticks. */
  removeExpired(tick) {
    const expired = [];
    for (const [stableId, state] of this.states) {
      if (tick - state.lastSeen > this.config.maxGap) expired.push(stableId);
    }
    for (const stableId of expired) {
      const rawId = this.states.get(stableId).rawId;
      if (this.rawToStable.get(rawId) === stableId) this.rawToStable.delete(rawId);
      this.states.delete(stableId);
    }
    return expired.length;
  }

  _createStableId(rawId, box, tick) {
    const stableId = this.nextStableId;
    this.nextStableId += 1;
    this.rawToStable.set(rawId, stableId);
    this.states.set(stableId, {
      rawId,
      box: { x1: box.x1, y1: box.y1, x2: box.x2, y2: box.y2, score: box.score },
      previousBox: null,
      lastSeen: tick,
      velocity: { x: 0, y: 0 },
    });
    return stableId;
  }

  _findReassociation(box, tick, activeStableIds, frame) {
    let bestStableId = null;
    let bestScore = null;
    for (const [stableId, state] of this.states) {
      if (activeStableIds.has(stableId)) continue;
      const gap = tick - state.lastSeen;
      if (gap <= 0 || gap > this.config.maxGap) continue;
      const predicted = shiftBox(state.box, state.velocity.x * gap,
        state.velocity.y * gap, frame.width, frame.height);
      const iou = boxIou(box, predicted);
      const distance = centerDistance(box, predicted);
      // Need either some overlap or very close centers.
      if (iou < this.config.minIou && distance > this.config.maxCenterDistance) continue;
      // Higher IoU is better, lower distance is better.
      const score = iou - 0.002 * distance;
      if (bestScore === null || score > bestScore) {
        bestScore = score;
        bestStableId = stableId;
      }
    }
    return bestStableId;
  }

  _updateState(stableId, rawId, box, tick) {
    const state = this.states.get(stableId);
    const oldCenter = boxCenter(state.box);
    const newCenter = boxCenter(box);
    const gap = Math.max(1, tick - state.lastSeen);
    state.previousBox = state.box;
    state.box = { x1: box.x1, y1: box.y1, x2: box.x2, y2: box.y2, score: box.score };
    state.lastSeen = tick;
    state.rawId = rawId;
    state.velocity = {
      x: (newCenter.x - oldCenter.x) / gap,
      y: (newCenter.y - oldCenter.y) / gap,
    };
  }
}
