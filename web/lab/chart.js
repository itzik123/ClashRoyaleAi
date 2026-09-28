// The learning curve: the learner's greedy score on the five challenge attacks
// as it trains, against the exhaustive-search best, a random drop, and the
// visitor's own score.
(function (root) {
  'use strict';
  const FONT = "'Unbounded', system-ui, sans-serif";
  const COLORS = { ai: '#4a9eff', best: '#4ade80', random: '#6E6E73', you: '#facc15' };

  class LabChart {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.points = [];
      this.refs = {};
    }
    setRefs(refs) { this.refs = refs; }
    setPoints(points) { this.points = points; }

    render() {
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
      if (!w || !h) return;
      if (this.canvas.width !== Math.round(w * dpr)) { this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(h * dpr); }
      const c = this.ctx;
      c.setTransform(dpr, 0, 0, dpr, 0, 0);
      c.clearRect(0, 0, w, h);
      const L = 34, R = 68, T = 8, B = 20;
      const pw = w - L - R, ph = h - T - B;
      const last = this.points.length ? this.points[this.points.length - 1].tries : 0;
      const xMax = niceMax(Math.max(2000, last * 1.08));
      const X = t => L + (t / xMax) * pw, Y = v => T + (1 - Math.max(-0.05, Math.min(1, v))) * ph;

      c.font = `500 9px ${FONT}`;
      c.fillStyle = '#6E6E73';
      c.strokeStyle = 'rgba(255,255,255,.06)';
      c.lineWidth = 1;
      c.textAlign = 'right'; c.textBaseline = 'middle';
      for (const v of [0, 0.25, 0.5, 0.75, 1]) {
        c.beginPath(); c.moveTo(L, Y(v)); c.lineTo(L + pw, Y(v)); c.stroke();
        c.fillText(`${Math.round(v * 100)}%`, L - 6, Y(v));
      }
      c.textAlign = 'center'; c.textBaseline = 'top';
      for (let k = 0; k <= 4; k++) {
        const t = xMax * k / 4;
        c.fillText(t >= 1000 ? `${+(t / 1000).toFixed(1)}k` : `${t}`, X(t), T + ph + 6);
      }
      c.textAlign = 'right';
      c.fillText('tries', L + pw + R - 4, T + ph + 6);

      // Reference lines, labelled at the right edge.
      const refs = [['random', 'random'], ['best', 'best'], ['you', 'you']];
      const labels = [];
      for (const [key, label] of refs) {
        const v = this.refs[key];
        if (v == null) continue;
        c.save();
        c.setLineDash([5, 4]);
        c.strokeStyle = COLORS[key]; c.lineWidth = 1.5;
        c.beginPath(); c.moveTo(L, Y(v)); c.lineTo(L + pw, Y(v)); c.stroke();
        c.restore();
        labels.push({ y: Y(v), text: `${label} ${Math.round(v * 100)}%`, color: COLORS[key] });
      }

      // The learner: an area under its line, then the line.
      if (this.points.length) {
        c.beginPath();
        c.moveTo(X(this.points[0].tries), Y(0));
        for (const p of this.points) c.lineTo(X(p.tries), Y(p.v));
        c.lineTo(X(last), Y(0));
        c.closePath();
        c.fillStyle = 'rgba(74,158,255,.10)'; c.fill();
        c.beginPath();
        this.points.forEach((p, i) => (i ? c.lineTo(X(p.tries), Y(p.v)) : c.moveTo(X(p.tries), Y(p.v))));
        c.strokeStyle = COLORS.ai; c.lineWidth = 2.2; c.lineJoin = 'round'; c.stroke();
        const lp = this.points[this.points.length - 1];
        c.beginPath(); c.arc(X(lp.tries), Y(lp.v), 3.5, 0, Math.PI * 2); c.fillStyle = COLORS.ai; c.fill();
        labels.push({ y: Y(lp.v), text: `AI ${Math.round(lp.v * 100)}%`, color: COLORS.ai });
      }

      // Right-edge labels, nudged apart so they never overlap.
      labels.sort((a, b) => a.y - b.y);
      for (let i = 1; i < labels.length; i++) if (labels[i].y - labels[i - 1].y < 11) labels[i].y = labels[i - 1].y + 11;
      c.textAlign = 'left'; c.textBaseline = 'middle'; c.font = `600 9px ${FONT}`;
      for (const l of labels) { c.fillStyle = l.color; c.fillText(l.text, L + pw + 5, Math.min(T + ph, l.y)); }
    }
  }

  function niceMax(v) {
    const p = Math.pow(10, Math.floor(Math.log10(v)));
    for (const m of [1, 2, 2.5, 4, 5, 8, 10]) if (m * p >= v) return m * p;
    return 10 * p;
  }

  LabChart.COLORS = COLORS;
  root.LabChart = LabChart;
})(window);
