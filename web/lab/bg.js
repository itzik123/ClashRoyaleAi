// The page background: a field of small squares washed by slow blue and red
// waves, blue rising from the bottom of the screen and red from the top, as
// the two teams sit on the board. Every square also carries a fixed jitter of
// its own, so no two neighbours are quite the same shade.
//
// It loops exactly: every motion term turns a whole number of times per
// LOOP_S, so the last frame of the loop meets the first. Squares are batched
// into a few dozen colour buckets, so a frame is a few dozen fills, not
// thousands, and it redraws at most FPS times a second. With reduced motion
// requested it draws one still frame.
(function () {
  'use strict';

  const canvas = document.getElementById('bgfx');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const CELL = 18;          // px between square origins
  const GAP = 4;            // px between squares
  const LOOP_S = 16;        // the loop's length
  const FPS = 24;
  const LEVELS = 16;        // alpha steps per colour
  const MIN_ALPHA = 0.035;  // every square shows, faintly: the grid itself
  const MAX_ALPHA = 0.38;
  const BLUE = [74, 158, 255], RED = [255, 82, 82];

  const still = window.matchMedia('(prefers-reduced-motion: reduce)');
  let cols = 0, rows = 0, jitter = null, last = -1e9, dpr = 1;

  function resize() {
    dpr = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
    const w = window.innerWidth, h = window.innerHeight;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    cols = Math.ceil(w / CELL) + 1;
    rows = Math.ceil(h / CELL) + 1;
    jitter = new Float32Array(cols * rows);
    for (let i = 0; i < jitter.length; i++) {
      // A hash, not Math.random(): the same squares keep the same shade.
      const s = Math.sin(i * 12.9898 + 78.233) * 43758.5453;
      jitter[i] = s - Math.floor(s);
    }
    draw(performance.now(), true);
  }

  // One wave field per team, in [0, 1]. t runs 0..2*PI over the loop.
  function wave(u, v, t, phase) {
    return 0.5 + 0.5 * (
      0.55 * Math.sin(u * 0.33 + v * 0.10 - 2 * t + phase) +
      0.30 * Math.sin(u * 0.12 - v * 0.27 + 3 * t + phase * 2) +
      0.15 * Math.sin((u + v) * 0.21 - 1 * t));
  }

  function draw(now, force) {
    if (!force && now - last < 1000 / FPS) return;
    last = now;
    const t = ((now / 1000) % LOOP_S) / LOOP_S * Math.PI * 2;
    const buckets = [[], []];
    for (let k = 0; k < 2; k++) for (let l = 0; l < LEVELS; l++) buckets[k].push([]);
    for (let r = 0; r < rows; r++) {
      const down = r / Math.max(1, rows - 1);          // 0 at the top, 1 at the bottom
      for (let c = 0; c < cols; c++) {
        const j = jitter[r * cols + c];
        const b = wave(c, rows - r, t, 0) * (0.25 + 0.75 * down);
        const rd = wave(cols - c, r, t, 1.7) * (0.25 + 0.75 * (1 - down));
        const team = b >= rd ? 0 : 1;
        const crest = Math.max(b, rd);
        const level = Math.min(LEVELS - 1, Math.floor(Math.pow(crest, 1.8) * (0.65 + 0.7 * j) * LEVELS));
        buckets[team][level].push(c, r);
      }
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const size = CELL - GAP, radius = 3;
    for (let k = 0; k < 2; k++) {
      const [R, G, B] = k === 0 ? BLUE : RED;
      for (let l = 0; l < LEVELS; l++) {
        const list = buckets[k][l];
        if (!list.length) continue;
        const a = MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * Math.pow(l / (LEVELS - 1), 1.3);
        ctx.fillStyle = `rgba(${R},${G},${B},${a.toFixed(3)})`;
        ctx.beginPath();
        for (let i = 0; i < list.length; i += 2) {
          const x = list[i] * CELL + GAP / 2, y = list[i + 1] * CELL + GAP / 2;
          if (ctx.roundRect) ctx.roundRect(x, y, size, size, radius);
          else ctx.rect(x, y, size, size);
        }
        ctx.fill();
      }
    }
  }

  function frame(now) {
    draw(now, false);
    if (!still.matches) requestAnimationFrame(frame);
  }

  window.addEventListener('resize', resize);
  still.addEventListener('change', () => { if (!still.matches) requestAnimationFrame(frame); });
  resize();
  requestAnimationFrame(frame);
})();
