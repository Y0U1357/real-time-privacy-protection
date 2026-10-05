/**
 * Local privacy verification aid.
 *
 * The pipeline must never upload a camera frame. This module wraps fetch,
 * XMLHttpRequest and sendBeacon with pass-through counters so the dashboard can
 * show how many requests the page made and how many of them carried image or
 * video data. Nothing here blocks a request; it only observes.
 *
 * Known limit: binary WebSocket frames are not inspected. Do not open one; the
 * browser-local pipeline has no reason to.
 */

const IMAGE_TYPES = /^(image|video)\//;
const DATA_URL = /data:(image|video)\//i;
const JPEG_BASE64 = /^\/9j\//;

const report = {
  installed: false,
  requests: 0,
  uploads: 0,
  imageUploads: 0,
  websockets: 0,
  lastUrl: '—',
  lastUploadUrl: '—',
};

function track(url, method, body) {
  report.requests += 1;
  report.lastUrl = `${method} ${url}`;
  if (method === 'GET' || method === 'HEAD') return;
  report.uploads += 1;
  if (carriesImage(body)) {
    report.imageUploads += 1;
    report.lastUploadUrl = `${method} ${url}`;
  }
}

function carriesImage(body) {
  if (!body) return false;
  if (typeof body === 'string') return DATA_URL.test(body) || JPEG_BASE64.test(body);
  if (typeof Blob !== 'undefined' && body instanceof Blob) {
    return IMAGE_TYPES.test(body.type || '');
  }
  if (typeof FormData !== 'undefined' && body instanceof FormData) {
    for (const value of body.values()) {
      if (carriesImage(value)) return true;
    }
    return false;
  }
  if (typeof URLSearchParams !== 'undefined' && body instanceof URLSearchParams) {
    return DATA_URL.test(body.toString());
  }
  return false;
}

export function installNetworkGuard() {
  if (report.installed || typeof window === 'undefined') return report;
  report.installed = true;

  if (typeof window.fetch === 'function') {
    const originalFetch = window.fetch;
    window.fetch = function guardFetch(input, init) {
      try {
        const url = typeof input === 'string' ? input : (input && input.url) || '';
        const method = ((init && init.method) || (input && input.method) || 'GET').toUpperCase();
        track(url, method, init && init.body);
      } catch (error) {
        // Observation must never break a request.
      }
      return originalFetch.apply(this, arguments);
    };
  }

  if (typeof XMLHttpRequest === 'function') {
    const open = XMLHttpRequest.prototype.open;
    const send = XMLHttpRequest.prototype.send;
    XMLHttpRequest.prototype.open = function guardOpen(method, url) {
      try {
        this.__guardMethod = String(method || 'GET').toUpperCase();
        this.__guardUrl = String(url || '');
      } catch (error) { /* ignore */ }
      return open.apply(this, arguments);
    };
    XMLHttpRequest.prototype.send = function guardSend(body) {
      try {
        track(this.__guardUrl || '', this.__guardMethod || 'GET', body);
      } catch (error) { /* ignore */ }
      return send.apply(this, arguments);
    };
  }

  if (typeof navigator !== 'undefined' && typeof navigator.sendBeacon === 'function') {
    const beacon = navigator.sendBeacon.bind(navigator);
    navigator.sendBeacon = function guardBeacon(url, data) {
      try {
        track(String(url || ''), 'POST', data);
      } catch (error) { /* ignore */ }
      return beacon(url, data);
    };
  }

  if (typeof WebSocket === 'function') {
    const Native = WebSocket;
    const Guarded = function guardWebSocket(url, protocols) {
      report.websockets += 1;
      return protocols === undefined ? new Native(url) : new Native(url, protocols);
    };
    Guarded.prototype = Native.prototype;
    Object.assign(Guarded, { CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3 });
    window.WebSocket = Guarded;
  }

  return report;
}

/** Snapshot for the diagnostics panel. Frames served by the page itself are
 *  still 'uploads' only if a script posts image data, so a healthy run shows
 *  imageUploads = 0 after the model files have been fetched. */
export function getNetworkReport() {
  return {
    installed: report.installed,
    requests: report.requests,
    uploads: report.uploads,
    imageUploads: report.imageUploads,
    websockets: report.websockets,
    lastUrl: report.lastUrl,
    lastUploadUrl: report.lastUploadUrl,
  };
}
