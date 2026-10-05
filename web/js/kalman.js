// Port of yolox/tracker/kalman_filter.py (MIT, Copyright 2021 Yifu Zhang).
const zeros = (n, m) => Array.from({length:n}, () => Array(m).fill(0));
const transpose = a => a[0].map((_, j) => a.map(row => row[j]));
const multiply = (a, b) => a.map(row => b[0].map((_, j) => row.reduce((s, v, k) => s + v * b[k][j], 0)));
function inverse(a) {
  const n = a.length, out = a.map((row, i) => [...row, ...Array.from({length:n}, (_, j) => +(i === j))]);
  for (let i = 0; i < n; i++) {
    let pivot = i;
    for (let k = i + 1; k < n; k++) if (Math.abs(out[k][i]) > Math.abs(out[pivot][i])) pivot = k;
    [out[i], out[pivot]] = [out[pivot], out[i]];
    const scale = out[i][i];
    if (Math.abs(scale) < 1e-20) throw new Error('Singular Kalman covariance');
    out[i] = out[i].map(v => v / scale);
    for (let k = 0; k < n; k++) if (k !== i) {
      const factor = out[k][i];
      out[k] = out[k].map((v, j) => v - factor * out[i][j]);
    }
  }
  return out.map(row => row.slice(n));
}
export class KalmanFilter {
  constructor() {
    this.p = 1 / 20; this.v = 1 / 160;
    this.F = zeros(8, 8);
    for (let i = 0; i < 8; i++) { this.F[i][i] = 1; if (i < 4) this.F[i][i+4] = 1; }
  }
  initiate(z) {
    const h = z[3], p = this.p*h, v = this.v*h;
    const sd = [2*p,2*p,0.01,2*p,10*v,10*v,0.00001,10*v];
    return {mean:[...z,0,0,0,0], covariance:sd.map((s,i) => sd.map((_,j) => i === j ? s*s : 0))};
  }
  predict(track) {
    const m = track.mean.slice();
    if (track.state !== 'tracked') m[7] = 0;
    const p = this.p*m[3], v = this.v*m[3], sd = [p,p,0.01,p,v,v,0.00001,v];
    track.mean = m.map((val,i) => val + (i < 4 ? m[i+4] : 0));
    track.covariance = multiply(multiply(this.F, track.covariance), transpose(this.F));
    sd.forEach((s,i) => {track.covariance[i][i] += s*s;});
  }
  update(track, z) {
    const p = this.p*track.mean[3], sd = [p,p,0.1,p];
    const projected = track.covariance.slice(0,4).map((row,i) => row.slice(0,4).map((v,j) => v + (i === j ? sd[i]**2 : 0)));
    const gain = multiply(track.covariance.map(row => row.slice(0,4)), inverse(projected));
    const innovation = z.map((v,i) => v-track.mean[i]);
    track.mean = track.mean.map((v,i) => v + gain[i].reduce((s,g,j) => s+g*innovation[j],0));
    const correction = multiply(multiply(gain, projected), transpose(gain));
    track.covariance = track.covariance.map((row,i) => row.map((v,j) => v-correction[i][j]));
  }
}
