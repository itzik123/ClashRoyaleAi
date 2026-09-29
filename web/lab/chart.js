// The learning curve: the learner's greedy score on the five challenge attacks
// as it trains, against the exhaustive-search best, a random drop, and the
// visitor's own score.
(function (root) {
  'use strict';
  const FONT = "'Inter', system-ui, sans-serif";
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
      // The line labels at the right edge, gathered first so the margins can
      // be measured from them: Unbounded is wide, and a fixed margin clipped
      // "random 41%" and "100%".
      const labels = [];
      for (const [key, label] of [['random', 'random'], ['best', 'best'], ['you', 'you']]) {
        const v = this.refs[key];
        if (v != null) labels.push({ key, v, text: `${label} ${Math.round(v * 100)}%`, color: COLORS[key], dash: true });
      }
      const lp = this.points.length ? this.points[this.points.length - 1] : null;
      if (lp) labels.push({ key: 'ai', v: lp.v, text: `AI ${Math.round(lp.v * 100)}%`, color: COLORS.ai });
      c.font = `600 9px ${FONT}`;
      const labelW = Math.max(c.measureText('random 100%').width, ...labels.map(l => c.measureText(l.text).width));
      c.font = `500 9px ${FONT}`;
      const L = Math.ceil(c.measureText('100%').width) + 10, R = Math.ceil(labelW) + 12, T = 9, B = 20;
      const pw = w - L - R, ph = h - T - B;
      const last = lp ? lp.tries : 0;
      const xMax = niceMax(Math.max(2000, last * 1.08));
      const X = t => L + (t / xMax) * pw, Y = v => T + (1 - Math.max(-0.05, Math.min(1, v))) * ph;

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
      c.fillText('tries', w - 2, T + ph + 6);

      // Reference lines, dashed.
      for (const l of labels) {
        if (!l.dash) continue;
        c.save();
        c.setLineDash([5, 4]);
        c.strokeStyle = l.color; c.lineWidth = 1.5;
        c.beginPath(); c.moveTo(L, Y(l.v)); c.lineTo(L + pw, Y(l.v)); c.stroke();
        c.restore();
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
        c.beginPath(); c.arc(X(lp.tries), Y(lp.v), 3.5, 0, Math.PI * 2); c.fillStyle = COLORS.ai; c.fill();
      }

      // Right-edge labels, 11 px apart so they never overlap: pushed down
      // from the top, then back up from the bottom so none leaves the plot.
      const GAP = 11, top = T + 4, bottom = T + ph - 3;
      labels.forEach(l => { l.y = Y(l.v); });
      labels.sort((a, b) => a.y - b.y);
      for (let i = 0; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, i ? labels[i - 1].y + GAP : top);
      for (let i = labels.length - 1; i >= 0; i--)
        labels[i].y = Math.min(labels[i].y, i < labels.length - 1 ? labels[i + 1].y - GAP : bottom);
      c.textAlign = 'left'; c.textBaseline = 'middle'; c.font = `600 9px ${FONT}`;
      this.labelBoxes = [];
      for (const l of labels) {
        c.fillStyle = l.color;
        c.fillText(l.text, L + pw + 6, l.y);
        this.labelBoxes.push({ text: l.text, x: L + pw + 6, right: L + pw + 6 + c.measureText(l.text).width, y: l.y });
      }
      this.width = w;
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
