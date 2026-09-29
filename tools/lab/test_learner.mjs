// Tests for web/lab/learner.js.
//
//   node tools/lab/test_learner.mjs [table_prefix ...]
//
// 1. Gradient check: the hand-written backprop against central finite
//    differences, in float64.
// 2. Synthetic problem whose best cell AND delay move with the spawn: the
//    greedy policy must find them (a policy that ignores the state cannot).
// 3. The reward puts the tower strictly first: defender survival only breaks
//    ties between tries that prevented exactly as much.
// 4. For each outcome table given (lab_cli table), 3 seeds on the default
//    budget: tries until the greedy policy reached 90% of the exhaustive best.
// Exits non-zero on any failure.
import { Learner, loadTable, runLearner } from './table_env.mjs';

let failures = 0;
function check(ok, msg) {
  console.log(`  ${ok ? 'ok  ' : 'FAIL'}  ${msg}`);
  if (!ok) failures++;
}

// A small synthetic board: spawns along x, cells on a 6x5 patch, 4 delays.
function synthetic() {
  const spawns = [], cells = [];
  for (let x = 0; x < 18; x += 2) spawns.push([x, 19]);
  for (let y = 2; y < 7; y++) for (let x = 3; x < 15; x += 2) cells.push([x, y]);
  // Best cell tracks the spawn's x; best delay is 0 on the left, 3 on the right.
  const r = (s, c, d) => {
    const tx = 3 + (spawns[s][0] / 16) * 11, ty = 4;
    const dist = Math.hypot(cells[c][0] - tx, cells[c][1] - ty);
    const want = spawns[s][0] < 9 ? 0 : 3;
    return Math.exp(-dist * dist / 3) * (d === want ? 1 : 0.4);
  };
  return { spawns, cells, D: 4, r };
}

function gradientCheck() {
  console.log('gradient check (float64)');
  const p = synthetic();
  const L = new Learner({ spawns: p.spawns, cells: p.cells, delaySteps: p.D, width: 18,
                          hidden: 6, entropy: 0.05, float64: true, seed: 3 });
  // Move the output layers off their near-zero init so every term is exercised.
  for (const k of ['Wp', 'Wd', 'bc', 'bd']) for (let i = 0; i < L.params[k].length; i++) L.params[k][i] = (Math.sin(i * 1.7 + k.length) * 0.5);
  const batch = L.sampleBatch(12);
  const adv = batch.map((_, i) => Math.cos(i) * 0.8);
  const { grads } = L.lossAndGrad(batch, adv);
  // eps 1e-4: truncation error ~eps^2. Smaller steps make it WORSE, not better
  // (roundoff on W1's tiny gradients -- the input bumps are ~0 far from the
  // spawn), which is how a correct gradient behaves; a wrong one does not
  // improve with any step size.
  const eps = 1e-4;
  let worst = 0;
  for (const k in L.params) {
    const P = L.params[k];
    for (let t = 0; t < Math.min(P.length, 40); t++) {
      const i = Math.floor((t + 0.5) * P.length / Math.min(P.length, 40));
      const keep = P[i];
      P[i] = keep + eps; const lp = L.lossAndGrad(batch, adv).loss;
      P[i] = keep - eps; const lm = L.lossAndGrad(batch, adv).loss;
      P[i] = keep;
      const num = (lp - lm) / (2 * eps), ana = grads[k][i];
      const rel = Math.abs(num - ana) / Math.max(1e-8, Math.abs(num) + Math.abs(ana));
      if (Math.abs(num) + Math.abs(ana) > 1e-6) worst = Math.max(worst, rel);
    }
  }
  check(worst < 1e-5, `worst relative error ${worst.toExponential(2)} < 1e-5`);
}

function syntheticConvergence() {
  console.log('synthetic problem: best cell and delay move with the spawn');
  const p = synthetic();
  const L = new Learner({ spawns: p.spawns, cells: p.cells, delaySteps: p.D, width: 18, seed: 5 });
  for (let i = 0; i < 400; i++) {
    const b = L.sampleBatch(64);
    L.update(b, b.map(t => p.r(t.spawn, t.cell, t.delay)));
  }
  let got = 0, best = 0;
  for (let s = 0; s < p.spawns.length; s++) {
    const a = L.greedy(s);
    got += p.r(s, a.cell, a.delay);
    let b = 0;
    for (let c = 0; c < p.cells.length; c++) for (let d = 0; d < p.D; d++) b = Math.max(b, p.r(s, c, d));
    best += b;
  }
  check(got / best >= 0.9, `greedy reaches ${(100 * got / best).toFixed(1)}% of the best (>= 90%)`);
}

function towerFirst() {
  console.log('reward: tower first, survival only breaks ties');
  const p = synthetic();
  const L = new Learner({ spawns: p.spawns, cells: p.cells, delaySteps: p.D, width: 18 });
  const at = s => ({ spawn: s, cell: 0, delay: 0 });
  // Battle Ram vs Musketeer, from the maintainer's brief: the whole tower
  // saved with a dead defender beats 91% saved with her at 93% HP.
  let r = L.rewards([at(0), at(0)], [1.0, 0.91], [0, 0.93]);
  check(r[0] > r[1], `100% saved, defender dead (${r[0].toFixed(3)}) beats 91% saved at 93% HP (${r[1].toFixed(3)})`);
  // Even one hit point of 2,534 outweighs a defender at full HP.
  const hp = 1 / 2534;
  r = L.rewards([at(1), at(1)], [0.5 + hp, 0.5], [0, 1]);
  check(r[0] > r[1], `one tower hit point more (${r[0].toFixed(6)}) beats a defender at 100% HP (${r[1].toFixed(6)})`);
  // A tie on the tower: the defender still standing wins.
  r = L.rewards([at(2), at(2)], [0.8, 0.8], [0.6, 0.1]);
  check(r[0] > r[1], `equal tower saved: the defender at 60% (${r[0].toFixed(3)}) beats 10% (${r[1].toFixed(3)})`);
  // A later, better try takes the bonus away from the old best.
  L.rewards([at(3)], [0.7], [1]);
  r = L.rewards([at(3), at(3)], [0.75, 0.7], [0, 1]);
  check(r[0] === 0.75 && r[1] === 0.7, `once 75% has been seen, 70% at full HP earns no bonus (${r[1]})`);
}

function tables(prefixes) {
  for (const prefix of prefixes) {
    const table = loadTable(prefix);
    const m = table.matchup;
    console.log(`table ${prefix}: attacker ${m.attacker} vs defender ${m.defender}, ` +
                `${table.S} spawns x ${table.C} cells x ${table.D} delays`);
    for (const seed of [1, 2, 3]) {
      const t0 = Date.now();
      const { final, triesTo90, curve } = runLearner(table, { seed });
      const pts = curve.filter((_, i) => i % 3 === 0).map(p => `${p.tries}:${(100 * p.ratio).toFixed(0)}%`).join(' ');
      console.log(`    seed ${seed}: ${pts}`);
      check(triesTo90 !== null,
            `seed ${seed}: 90% of best after ${triesTo90 ?? '-'} tries; final greedy ${final.greedy.toFixed(3)} ` +
            `vs best ${final.best.toFixed(3)}, random ${final.random.toFixed(3)} (${Date.now() - t0} ms)`);
    }
  }
}

gradientCheck();
syntheticConvergence();
towerFirst();
tables(process.argv.slice(2));
console.log(failures ? `\n${failures} failure(s)` : '\nall learner checks passed');
process.exit(failures ? 1 : 0);
