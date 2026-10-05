/**
 * ByteTrack for the browser - a direct port of yolox/tracker/byte_tracker.py
 * (MIT, Copyright 2021 Yifu Zhang) built on kalman.js and assignment.js.
 *
 * Two-stage association every detection tick, exactly like BYTETracker.update():
 *
 *   stage 0  predict every confirmed + lost track with the Kalman filter
 *   stage 1  confirmed & lost  <-> high-score dets (score > trackThresh)
 *            fused cost = 1 - IoU * det_score, global assignment, matchThresh
 *   stage 2  unmatched confirmed <-> low-score dets (lowThresh < score <= trackThresh)
 *            plain IoU cost (no score fusion), secondMatchThresh
 *   stage 3  unconfirmed tracks <-> leftover high dets, unconfirmedMatchThresh
 *   stage 4  new tracks from confident leftovers (score >= trackThresh + newTrackMargin)
 *   stage 5  lost tracks expire after trackBuffer ticks
 *   stage 6  suppress duplicate tracked/lost pairs (IoU > 1 - duplicateDistance)
 *
 * Public API kept from the previous geometric tracker, so app.js and the
 * self-test keep working unchanged:
 *
 *   update(detections, timeMs, frame)  - one detection tick
 *   reset()                            - forget everything (camera restart)
 *   tracks                             - live tracks (active + coasting + fresh)
 *   tick                               - detection tick counter
 *   predictBox(track, fraction, frame) / advanceBox(box, vx, vy, fraction, frame)
 *
 * Track fields consumed elsewhere: id, box, displayBox, previousBox, velocityX,
 * velocityY, missingTicks, confirmed, hits, score, stableId (added by app.js).
 * `box` is derived from the Kalman state every tick, so a coasting track is
 * predicted forward with the same motion model the Python prototype outputs.
 */

import { CONFIG } from './config.js';
import { KalmanFilter } from './kalman.js';
import { linearAssignment } from './assignment.js';
import { boxIou, shiftBox } from './geometry.js';

/** xyah measurement [cx, cy, aspect, height] of a {x1, y1, x2, y2} box. */
function xyah(box) {
  const width = box.x2 - box.x1;
  const height = box.y2 - box.y1;
  return [(box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2, width / height, height];
}

/** tlbr box implied by the Kalman mean [cx, cy, a, h, vx, vy, va, vh]. */
function boxFromMean(mean) {
  const width = mean[2] * mean[3];
  return {
    x1: mean[0] - width / 2,
    y1: mean[1] - mean[3] / 2,
    x2: mean[0] + width / 2,
    y2: mean[1] + mean[3] / 2,
    score: 0,
  };
}

export class Tracker {
  constructor(config = CONFIG.tracker) {
    this.config = config;
    this.kalman = new KalmanFilter();
    this.tracks = [];
    this.nextId = 1;
    this.tick = 0;
  }

  /** Forget every track; called when the camera (re)starts. */
  reset() {
    this.tracks.length = 0;
    this.nextId = 1;
    this.tick = 0;
  }

  /**
   * One ByteTrack update.
   * @param {Array<object>} detections detector boxes {x1,y1,x2,y2,score} in frame pixels
   * @param {number} timeMs performance.now() of this tick
   * @param {{width:number, height:number}} frame source frame size (kept for the
   *   API; boxes are clamped at the privacy stage, like Python's clip_box)
   */
  update(detections, timeMs, frame) {
    void frame;
    this.tick += 1;
    const cfg = this.config;
    const stats = {
      matched: 0, secondMatched: 0, reactivated: 0,
      created: 0, removed: 0, active: 0, coasting: 0,
    };

    // Score split identical to byte_tracker.py: the high batch feeds stages
    // 1/3/4, the low batch (lowThresh < score <= trackThresh) only stage 2.
    const high = [];
    const low = [];
    for (const detection of detections) {
      if (detection.score > cfg.trackThresh) high.push(detection);
      else if (detection.score > cfg.lowThresh) low.push(detection);
    }

    // Split live tracks exactly like the Python tracked/unconfirmed/lost lists.
    const pool = [];
    const unconfirmed = [];
    for (const track of this.tracks) {
      if (track.state === 'tracked' && !track.confirmed) unconfirmed.push(track);
      else pool.push(track); // confirmed tracked + lost
    }

    // Stage 0: Kalman prediction for the association pool.
    for (const track of pool) {
      this.kalman.predict(track);
      this._syncBox(track);
    }

    // Stage 1: first association with high-score detections.
    const stage1 = linearAssignment(
      this._costs(pool, high, true), high.length, cfg.matchThresh);
    for (const [row, column] of stage1.matches) {
      const track = pool[row];
      const reactivated = track.state !== 'tracked';
      this._match(track, high[column], timeMs);
      stats[reactivated ? 'reactivated' : 'matched'] += 1;
    }

    // Stage 2: second association - unmatched confirmed tracks with the
    // low-score batch. Plain IoU cost, no score fusion (byte_tracker.py).
    const rTracked = stage1.unmatchedRows
      .map((row) => pool[row])
      .filter((track) => track.state === 'tracked');
    const stage2 = linearAssignment(
      this._costs(rTracked, low, false), low.length, cfg.secondMatchThresh);
    for (const [row, column] of stage2.matches) {
      this._match(rTracked[row], low[column], timeMs);
      stats.secondMatched += 1;
    }
    for (const row of stage2.unmatchedRows) rTracked[row].state = 'lost';

    // Stage 3: unconfirmed tracks with the high detections left from stage 1.
    const remainingHigh = stage1.unmatchedCols.map((column) => high[column]);
    const stage3 = linearAssignment(
      this._costs(unconfirmed, remainingHigh, true),
      remainingHigh.length, cfg.unconfirmedMatchThresh);
    for (const [row, column] of stage3.matches) {
      this._match(unconfirmed[row], remainingHigh[column], timeMs);
      stats.matched += 1;
    }
    for (const row of stage3.unmatchedRows) {
      unconfirmed[row].state = 'removed';
      stats.removed += 1;
    }

    // Stage 4: brand-new tracks, only from confident leftovers.
    const newTrackThresh = cfg.trackThresh + cfg.newTrackMargin;
    for (const column of stage3.unmatchedCols) {
      const detection = remainingHigh[column];
      if (detection.score < newTrackThresh) continue;
      this.tracks.push(this._create(detection, timeMs));
      stats.created += 1;
    }

    // Stage 5: lost tracks are forgotten after track_buffer ticks.
    for (const track of this.tracks) {
      if (track.state === 'lost' && this.tick - track.lastSeenTick > cfg.trackBuffer) {
        track.state = 'removed';
        stats.removed += 1;
      }
    }

    // Stage 6: suppress near-equal tracked/lost duplicates.
    this._removeDuplicates();

    // Public bookkeeping: missingTicks + the final track list.
    for (const track of this.tracks) {
      track.missingTicks = this.tick - track.lastSeenTick;
      if (track.state === 'removed') continue;
      if (track.missingTicks === 0 && track.confirmed) stats.active += 1;
      else if (track.missingTicks > 0) stats.coasting += 1;
    }
    for (let index = this.tracks.length - 1; index >= 0; index -= 1) {
      if (this.tracks[index].state === 'removed') this.tracks.splice(index, 1);
    }
    stats.tracks = this.tracks;
    return stats;
  }

  /** Cost matrix rows: fused 1 - IoU * score, or plain 1 - IoU for stage 2. */
  _costs(tracks, detections, fuseScore) {
    return tracks.map((track) => detections.map((detection) => {
      const iou = boxIou(detection, track.box);
      return 1 - (fuseScore ? iou * detection.score : iou);
    }));
  }

  /** Matched track: Kalman correction + public fields (STrack.update / re_activate). */
  _match(track, detection, timeMs) {
    this.kalman.update(track, xyah(detection));
    track.state = 'tracked';
    track.confirmed = true; // is_activated
    track.score = detection.score;
    track.hits += 1;
    track.lastSeenTick = this.tick;
    track.lastTickMs = timeMs;
    track.missingTicks = 0;
    this._syncBox(track);
  }

  /** New track from a confident detection (STrack.activate). */
  _create(detection, timeMs) {
    const track = {
      id: this.nextId,
      state: 'tracked',
      // Python: a new track only activates immediately on frame_id == 1;
      // later it must survive one association round first (is_activated).
      confirmed: this.tick === 1,
      score: detection.score,
      hits: 1,
      startTick: this.tick,
      lastSeenTick: this.tick,
      lastTickMs: timeMs,
      missingTicks: 0,
      valid: true,
      box: null,
      displayBox: null,
      previousBox: null,
      velocityX: 0,
      velocityY: 0,
    };
    const state = this.kalman.initiate(xyah(detection));
    track.mean = state.mean;
    track.covariance = state.covariance;
    this.nextId += 1;
    this._syncBox(track);
    return track;
  }

  /** Publish the Kalman state as the public box + velocity fields. */
  _syncBox(track) {
    const box = boxFromMean(track.mean);
    box.score = track.score;
    track.previousBox = track.box;
    track.box = box;
    track.displayBox = box;
    track.velocityX = track.mean[4];
    track.velocityY = track.mean[5];
  }

  /**
   * Port of remove_duplicate_stracks(): when a tracked and a lost track sit
   * on top of each other (IoU > 1 - duplicateDistance) keep the longer-lived.
   */
  _removeDuplicates() {
    const threshold = 1 - this.config.duplicateDistance;
    const tracked = this.tracks.filter((track) => track.state === 'tracked');
    const lost = this.tracks.filter((track) => track.state === 'lost');
    const dropTracked = new Set();
    const dropLost = new Set();
    for (const active of tracked) {
      for (const coasting of lost) {
        if (boxIou(active.box, coasting.box) <= threshold) continue;
        const activeAge = active.lastSeenTick - active.startTick;
        const coastingAge = coasting.lastSeenTick - coasting.startTick;
        if (activeAge > coastingAge) dropLost.add(coasting);
        else dropTracked.add(active);
      }
    }
    for (const track of dropTracked) track.state = 'removed';
    for (const track of dropLost) track.state = 'removed';
  }
}

/**
 * Advance a box by a fraction of one detection interval.
 * Bounded to one tick so a stale velocity can never throw the privacy box
 * across the frame; the render loop calls this between detections.
 */
export function advanceBox(box, velocityX, velocityY, fraction, frame) {
  const step = fraction < 0 ? 0 : (fraction > 1 ? 1 : fraction);
  if (step <= 0) return box;
  return shiftBox(box, velocityX * step, velocityY * step, frame.width, frame.height);
}

/**
 * Render box of a track: the (coasted) position advanced by the fraction of
 * one detection interval that has already elapsed.
 */
export function predictBox(track, fraction, frame) {
  return advanceBox(track.displayBox, track.velocityX, track.velocityY, fraction, frame);
}
