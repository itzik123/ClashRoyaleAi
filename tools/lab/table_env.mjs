// Outcome tables (lab_cli table) as an environment for web/lab/learner.js.
//
// The engine is deterministic, so looking a try up in an exhaustive table is
// exactly what the live engine would have answered -- which lets the learner
// be tested and tuned in Node at millions of tries a minute, and lets the
// curation suite compare it against the true best.
import fs from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
export const Learner = require('../../web/lab/learner.js');

// `r` is the share of the damage prevented (what the page scores and the
// reference lines measure) and `sv` how much of the defender survived; the
// learner turns the two into its reward itself (Learner.rewards), on the page
// and here alike.
export function loadTable(prefix) {
  const meta = JSON.parse(fs.readFileSync(prefix + '.json', 'utf8'));
  const read = file => {
    const buf = fs.readFileSync(file);
    return new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
  };
  const data = read(prefix + '.bin');
  const [S, C, D] = meta.shape;
  if (data.length !== S * C * D) throw new Error(`${prefix}: ${data.length} values, want ${S * C * D}`);
  if (!fs.existsSync(prefix + '.surv.bin')) throw new Error(`${prefix}: no .surv.bin (rebuild with lab_cli table)`);
  const surv = read(prefix + '.surv.bin');
  if (surv.length !== data.length) throw new Error(`${prefix}.surv.bin: ${surv.length} values, want ${data.length}`);
  const r = (s, c, d) => data[(s * C + c) * D + d];
  const sv = (s, c, d) => surv[(s * C + c) * D + d];
  const best = new Float32Array(S), randomMean = new Float32Array(S), bestAction = [];
  for (let s = 0; s < S; s++) {
    let b = -Infinity, ba = null, sum = 0;
    for (let c = 0; c < C; c++)
      for (let d = 0; d < D; d++) {
        const v = r(s, c, d);
        sum += v;
        if (v > b) { b = v; ba = { cell: c, delay: d }; }
      }
    best[s] = b;
    randomMean[s] = sum / (C * D);
    bestAction.push(ba);
  }
  return { meta, matchup: meta.matchup, S, C, D, r, sv, best, randomMean, bestAction };
}

// Evaluation spawns: evenly spaced through the spawn list, which runs row by
// row, so they cover both lanes and every depth.
export function evalSpawns(S, n = 16) {
  const out = [];
  for (let i = 0; i < Math.min(n, S); i++) out.push(Math.floor((i + 0.5) * S / Math.min(n, S)));
  return out;
}

export function makeLearner(table, opts = {}) {
  const m = table.matchup;
  return new Learner(Object.assign({
    spawns: m.spawns, cells: m.cells, delaySteps: m.delaySteps, width: 18,
  }, opts));
}

// The greedy policy's share of the damage prevented against the best and a
// random drop, and how much of the defender its answers leave standing.
export function scoreGreedy(table, learner, spawns) {
  let got = 0, best = 0, rand = 0, surv = 0;
  for (const s of spawns) {
    const a = learner.greedy(s);
    got += table.r(s, a.cell, a.delay);
    surv += table.sv(s, a.cell, a.delay);
    best += table.best[s];
    rand += table.randomMean[s];
  }
  const n = spawns.length;
  return { greedy: got / n, best: best / n, random: rand / n, ratio: best > 0 ? got / best : 1, survival: surv / n };
}

// Train on a try budget; the curve samples the greedy score as it goes.
// `stopAt90` ends the run at the first evaluation that reaches 90% of the
// best, which is all the curation gate needs.
export function runLearner(table, opts = {}, { budget = 30000, batch = 64, evalEvery = 2048, stopAt90 = false } = {}) {
  const learner = makeLearner(table, opts);
  const spawns = evalSpawns(table.S);
  const curve = [];
  let next = 0;
  while (learner.tries < budget) {
    if (learner.tries >= next) {
      curve.push(Object.assign({ tries: learner.tries }, scoreGreedy(table, learner, spawns)));
      next += evalEvery;
      if (stopAt90 && curve[curve.length - 1].ratio >= 0.9) break;
    }
    const tries = learner.sampleBatch(batch);
    learner.update(tries, learner.rewards(tries, tries.map(t => table.r(t.spawn, t.cell, t.delay)),
                                          tries.map(t => table.sv(t.spawn, t.cell, t.delay))));
  }
  const final = Object.assign({ tries: learner.tries }, scoreGreedy(table, learner, spawns));
  curve.push(final);
  // Tries until the greedy policy first reached 90% of the best.
  const hit = curve.find(p => p.ratio >= 0.9);
  return { learner, curve, final, triesTo90: hit ? hit.tries : null };
}
