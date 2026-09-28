// The Reflex Lab's learner: a tiny policy network that learns WHERE to drop a
// defender and WHEN, from nothing but the tower damage the engine reports.
//
// No ML library -- every multiply and every gradient is written out below, so
// you can read the whole thing. It runs in the page's worker and in Node
// (tools/lab/test_learner.mjs, tools/lab/curate.mjs).
//
//   state   where the attacker was dropped (x, y)
//   action  a legal cell for the defender, then a drop delay given that cell
//           (autoregressive, like the real agent: card, then placement)
//   reward  the fraction of no-defence tower damage prevented
//
// Trained with REINFORCE: raise the log-probability of what was tried in
// proportion to how much better it did than usual for that spawn (a running
// per-spawn baseline), plus a small entropy bonus so it keeps exploring.
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.Learner = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // Small, fast, seedable PRNG (mulberry32).
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gauss(rand) {
    const u = Math.max(rand(), 1e-12), v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function softmaxInPlace(v) {
    let m = -Infinity;
    for (let i = 0; i < v.length; i++) if (v[i] > m) m = v[i];
    let s = 0;
    for (let i = 0; i < v.length; i++) { v[i] = Math.exp(v[i] - m); s += v[i]; }
    for (let i = 0; i < v.length; i++) v[i] /= s;
    return v;
  }

  function entropy(p) {
    let h = 0;
    for (let i = 0; i < p.length; i++) if (p[i] > 0) h -= p[i] * Math.log(p[i]);
    return h;
  }

  function sample(p, rand) {
    let u = rand(), i = 0;
    for (; i < p.length - 1; i++) { u -= p[i]; if (u <= 0) break; }
    return i;
  }

  function argmax(v) {
    let b = 0;
    for (let i = 1; i < v.length; i++) if (v[i] > v[b]) b = i;
    return b;
  }

  // Soft one-hot: a position becomes bumps on a grid, so neighbouring
  // spawns start out with similar features instead of unrelated ones.
  function bumps(value, n, sigma, out, offset) {
    for (let k = 0; k < n; k++) {
      const d = (value - k) / sigma;
      out[offset + k] = Math.exp(-0.5 * d * d);
    }
  }

  const DEFAULTS = {
    hidden: 48,        // hidden units
    basisStep: 2,      // placement basis: one bump every 2 cells...
    basisSigma: 1.1,   // ...this wide, in cells
    lr: 0.01,          // Adam step size
    // Entropy bonus, as a fraction of each head's maximum, annealed linearly
    // from `entropy` to `entropyEnd` over `entropyTries` tries (entropyEnd
    // null: held constant). Starting high keeps it from settling on the first
    // decent answer. Giant vs Cannon, 3 seeds at batches of 16 and 64 (the
    // tools/lab outcome tables): held at 0.01 it missed 90% of the best on 5
    // of the 6 runs, mostly stuck on the lane Cannon at 75%; annealed 0.1 ->
    // 0.005 over 10,000 tries, on 1 of 6, and on none at the page's 16.
    entropy: 0.1,
    entropyEnd: 0.005,
    entropyTries: 10000,
    normalize: false,  // scale advantages by their running RMS
    seed: 1,
    float64: false,
  };

  class Learner {
    // spawns, cells: [[x, y], ...] from the engine; width/height: the board.
    constructor(opts) {
      const o = Object.assign({}, DEFAULTS, opts);
      this.opts = o;
      // float64 for the gradient check (tools/lab/test_learner.mjs); float32 otherwise.
      this.F = o.float64 ? Float64Array : Float32Array;
      this.rand = rng(o.seed);
      this.spawns = o.spawns;
      this.cells = o.cells;
      this.D = o.delaySteps;
      this.C = o.cells.length;
      this.H = o.hidden;

      // State encoding: x bumps over the board width, y bumps over the rows
      // the spawns actually use.
      this.width = o.width;
      this.rowMin = Math.min(...o.spawns.map(s => s[1]));
      this.rows = Math.max(...o.spawns.map(s => s[1])) - this.rowMin + 1;
      this.I = this.width + this.rows;
      this.inputs = o.spawns.map(([x, y]) => {
        const v = new this.F(this.I);
        bumps(x, this.width, 1.0, v, 0);
        bumps(y - this.rowMin, this.rows, 1.0, v, this.width);
        return v;
      });

      // Placement basis: radial bumps on a coarse grid over the legal cells.
      // A cell's logit is a weighted sum of the bumps around it, so a reward
      // at one cell also moves its neighbours -- the reason the real agent's
      // placement head is convolutional.
      const xs = o.cells.map(c => c[0]), ys = o.cells.map(c => c[1]);
      const centres = [];
      for (let y = Math.min(...ys); y <= Math.max(...ys) + 0.01; y += o.basisStep)
        for (let x = Math.min(...xs); x <= Math.max(...xs) + 0.01; x += o.basisStep)
          if (o.cells.some(c => Math.hypot(c[0] - x, c[1] - y) <= o.basisStep)) centres.push([x, y]);
      this.K = centres.length;
      this.phi = new this.F(this.C * this.K);
      for (let c = 0; c < this.C; c++)
        for (let k = 0; k < this.K; k++) {
          const dx = o.cells[c][0] - centres[k][0], dy = o.cells[c][1] - centres[k][1];
          this.phi[c * this.K + k] = Math.exp(-(dx * dx + dy * dy) / (2 * o.basisSigma * o.basisSigma));
        }

      // Parameters. The output layers start near zero, so the first policy is
      // close to uniform: every cell and delay equally likely.
      const I = this.I, H = this.H, K = this.K, C = this.C, D = this.D;
      const init = (n, scale) => {
        const a = new this.F(n);
        for (let i = 0; i < n; i++) a[i] = gauss(this.rand) * scale;
        return a;
      };
      this.params = {
        W1: init(H * I, 1 / Math.sqrt(I)), b1: new this.F(H),
        Wp: init(K * H, 0.01), bp: new this.F(K), bc: new this.F(C),
        Wd: init(D * (H + K), 0.01), bd: new this.F(D),
      };
      this.adam = {};
      for (const k in this.params) {
        this.adam[k] = { m: new this.F(this.params[k].length), v: new this.F(this.params[k].length) };
      }
      this.step = 0;
      this.baseline = new this.F(o.spawns.length);
      this.seen = new Uint32Array(o.spawns.length);
      this.tries = 0;
    }

    // ---- forward ----------------------------------------------------------

    hidden(spawn) {
      const { W1, b1 } = this.params, x = this.inputs[spawn], I = this.I;
      const h = new this.F(this.H);
      for (let j = 0; j < this.H; j++) {
        let a = b1[j];
        for (let i = 0; i < I; i++) a += W1[j * I + i] * x[i];
        h[j] = Math.tanh(a);
      }
      return h;
    }

    cellLogits(h) {
      const { Wp, bp, bc } = this.params, H = this.H, K = this.K;
      const z = new this.F(K);
      for (let k = 0; k < K; k++) {
        let a = bp[k];
        for (let j = 0; j < H; j++) a += Wp[k * H + j] * h[j];
        z[k] = a;
      }
      const logits = new this.F(this.C);
      for (let c = 0; c < this.C; c++) {
        let a = bc[c];
        const row = c * K;
        for (let k = 0; k < K; k++) a += this.phi[row + k] * z[k];
        logits[c] = a;
      }
      return logits;
    }

    delayLogits(h, cell) {
      const { Wd, bd } = this.params, H = this.H, K = this.K, U = H + K;
      const logits = new this.F(this.D);
      for (let d = 0; d < this.D; d++) {
        let a = bd[d];
        for (let j = 0; j < H; j++) a += Wd[d * U + j] * h[j];
        for (let k = 0; k < K; k++) a += Wd[d * U + H + k] * this.phi[cell * K + k];
        logits[d] = a;
      }
      return logits;
    }

    // Probability of each legal cell for this spawn: the heatmap.
    cellProbs(spawn) {
      return softmaxInPlace(this.cellLogits(this.hidden(spawn)));
    }

    delayProbs(spawn, cell) {
      return softmaxInPlace(this.delayLogits(this.hidden(spawn), cell));
    }

    // Its best guess right now: the most likely cell, then its most likely delay.
    greedy(spawn) {
      const h = this.hidden(spawn);
      const cell = argmax(this.cellLogits(h));
      return { spawn, cell, delay: argmax(this.delayLogits(h, cell)) };
    }

    // ---- acting ------------------------------------------------------------

    // n tries, each at a random spawn, each action sampled from the policy.
    sampleBatch(n) {
      const out = [];
      for (let i = 0; i < n; i++) {
        const spawn = Math.floor(this.rand() * this.spawns.length);
        const h = this.hidden(spawn);
        const cell = sample(softmaxInPlace(this.cellLogits(h)), this.rand);
        const delay = sample(softmaxInPlace(this.delayLogits(h, cell)), this.rand);
        out.push({ spawn, cell, delay });
      }
      return out;
    }

    // ---- learning ------------------------------------------------------------

    // Loss and gradients for tries with given advantages:
    //   loss = mean over tries of  -A * (log p(cell) + log p(delay | cell))
    //                              - beta_c * H(cell) - beta_d * H(delay | cell)
    lossAndGrad(batch, advantages) {
      const P = this.params, H = this.H, K = this.K, C = this.C, D = this.D, I = this.I, U = H + K;
      const g = {};
      for (const k in P) g[k] = new this.F(P[k].length);
      const ent = this.entropyNow();
      const betaC = ent / Math.log(C), betaD = ent / Math.log(D);
      let loss = 0, entC = 0, entD = 0;
      const n = batch.length;

      for (let b = 0; b < n; b++) {
        const { spawn, cell, delay } = batch[b], A = advantages[b], x = this.inputs[spawn];
        const h = this.hidden(spawn);
        const pc = softmaxInPlace(this.cellLogits(h));
        const pd = softmaxInPlace(this.delayLogits(h, cell));
        const hc = entropy(pc), hd = entropy(pd);
        loss += (-A * (Math.log(pc[cell]) + Math.log(pd[delay])) - betaC * hc - betaD * hd) / n;
        entC += hc / n;
        entD += hd / n;

        // d loss / d logits: A * (p - onehot) from the policy term, and
        // beta * p * (log p + H) from the entropy term.
        const dlc = new this.F(C);
        for (let c = 0; c < C; c++) {
          dlc[c] = (A * (pc[c] - (c === cell ? 1 : 0)) + betaC * pc[c] * (Math.log(pc[c]) + hc)) / n;
        }
        const dld = new this.F(D);
        for (let d = 0; d < D; d++) {
          dld[d] = (A * (pd[d] - (d === delay ? 1 : 0)) + betaD * pd[d] * (Math.log(pd[d]) + hd)) / n;
        }

        const dh = new this.F(H);
        // Placement head: logits = phi . z + bc,  z = Wp . h + bp.
        for (let c = 0; c < C; c++) g.bc[c] += dlc[c];
        for (let k = 0; k < K; k++) {
          let dz = 0;
          for (let c = 0; c < C; c++) dz += this.phi[c * K + k] * dlc[c];
          g.bp[k] += dz;
          for (let j = 0; j < H; j++) {
            g.Wp[k * H + j] += dz * h[j];
            dh[j] += P.Wp[k * H + j] * dz;
          }
        }
        // Delay head: logits = Wd . [h, phi(cell)] + bd.
        for (let d = 0; d < D; d++) {
          g.bd[d] += dld[d];
          for (let j = 0; j < H; j++) {
            g.Wd[d * U + j] += dld[d] * h[j];
            dh[j] += P.Wd[d * U + j] * dld[d];
          }
          for (let k = 0; k < K; k++) g.Wd[d * U + H + k] += dld[d] * this.phi[cell * K + k];
        }
        // Hidden layer: h = tanh(W1 . x + b1).
        for (let j = 0; j < H; j++) {
          const da = dh[j] * (1 - h[j] * h[j]);
          g.b1[j] += da;
          for (let i = 0; i < I; i++) g.W1[j * I + i] += da * x[i];
        }
      }
      return { loss, grads: g, entropyCell: entC, entropyDelay: entD };
    }

    // The entropy bonus now: explore hard at first, commit later.
    entropyNow() {
      const o = this.opts;
      if (o.entropyEnd == null || !o.entropyTries) return o.entropy;
      const t = Math.min(1, this.tries / o.entropyTries);
      return o.entropy + (o.entropyEnd - o.entropy) * t;
    }

    // One learning step from tries and the rewards the engine gave them.
    update(batch, rewards) {
      const adv = new this.F(batch.length);
      let mean = 0;
      for (let b = 0; b < batch.length; b++) {
        const s = batch[b].spawn;
        // Running mean per spawn: early on a plain average, later a slow EMA.
        const alpha = Math.max(0.05, 1 / (this.seen[s] + 1));
        adv[b] = this.seen[s] === 0 ? 0 : rewards[b] - this.baseline[s];
        this.baseline[s] += alpha * (rewards[b] - this.baseline[s]);
        this.seen[s]++;
        mean += rewards[b] / batch.length;
      }
      if (this.opts.normalize) {
        // Divide by the advantages' running RMS, so the step size does not
        // depend on how much reward a matchup has to give.
        let sq = 0;
        for (let b = 0; b < adv.length; b++) sq += adv[b] * adv[b];
        this.advMs = this.advMs == null ? sq / adv.length : 0.95 * this.advMs + 0.05 * sq / adv.length;
        const rms = Math.max(Math.sqrt(this.advMs), 0.02);
        for (let b = 0; b < adv.length; b++) adv[b] /= rms;
      }
      const { loss, grads, entropyCell, entropyDelay } = this.lossAndGrad(batch, adv);
      this.applyAdam(grads);
      this.tries += batch.length;
      return { loss, meanReward: mean, entropyCell, entropyDelay };
    }

    applyAdam(grads) {
      const lr = this.opts.lr, b1 = 0.9, b2 = 0.999, eps = 1e-8;
      this.step++;
      const c1 = 1 - Math.pow(b1, this.step), c2 = 1 - Math.pow(b2, this.step);
      for (const k in this.params) {
        const p = this.params[k], g = grads[k], m = this.adam[k].m, v = this.adam[k].v;
        for (let i = 0; i < p.length; i++) {
          m[i] = b1 * m[i] + (1 - b1) * g[i];
          v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
          p[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + eps);
        }
      }
    }
  }

  Learner.rng = rng;
  Learner.DEFAULTS = DEFAULTS;
  return Learner;
});
