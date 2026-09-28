// The Reflex Lab's curation suite: decides which cards the page offers.
//
//   node tools/lab/curate.mjs [--jobs 8] [--tables-only] [--cached-only]
//
// For every candidate attacker x defender pair:
//   1. lab_cli table   every spawn x cell x delay (cached in tools/lab/out/tables)
//   2. lab_cli check   no unit may stall (the engine soak's criterion)
//   3. the learner, 3 seeds on the page's try budget, must on every seed
//      either reach 90% of the exhaustive best on the evaluation spawns or
//      close at least MIN_GAP_CLOSED of the gap from a random drop to it
// The page lets a visitor combine ANY attacker with ANY defender, so every
// pair among the shipped cards must pass. Cards are dropped greedily -- the
// one in the most failing pairs first -- until none fail. Writes
// web/lab/roster.json, which the page reads; nobody hand-edits it.
import { spawn } from 'node:child_process';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { evalSpawns, loadTable, runLearner } from './table_env.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');
const CLI = path.join(HERE, 'out', 'lab_cli.exe');
const TABLES = path.join(HERE, 'out', 'tables');
const ROSTER = path.join(REPO, 'web', 'lab', 'roster.json');

// Candidates, in order of preference: Hog Rider, Royal Giant, Battle Ram,
// Goblin Barrel, Giant, Balloon, Prince, Ram Rider. The page offers the first
// MAX_ATTACKERS that pass. Defenders: Cannon, Tesla, Inferno Tower, Bomb
// Tower, Mini P.E.K.K.A, Valkyrie, Musketeer, Skeleton Army. The Knight was
// cut as a near-copy of the Valkyrie with less to show: against a Goblin
// Barrel she saves 99% at best averaged over its landing spots, he 80%.
export const ATTACKERS = [15, 18, 81, 109, 2, 45, 14, 87];
export const DEFENDERS = [25, 26, 28, 27, 5, 10, 6, 12];
export const MAX_ATTACKERS = 6;
export const BUDGET = 30000;        // tries; the page shows about this many
export const BATCH = 16;            // tries per update, as in the page's Watch mode
const DEFAULT_MATCHUP = '15_25';    // Hog Rider vs Cannon: the page opens on it
export const SEEDS = [1, 2, 3];
const NOTHING_TO_LEARN = 0.05;     // best < 5% prevented: a pointless pairing
// The gate used to be "90% of the best on every seed", which cut the Royal
// Giant, the Valkyrie, the Tesla and the Balloon over four near-misses and
// one honest local optimum (Royal Giant vs Bomb Tower: 75% of the best on two
// seeds, 88% on the third). The page draws the best-possible line, so a
// plateau short of it reads honestly; the gate is for pairings where the
// learner visibly fails, and a learner that closes two thirds of the gap from
// random to best has not.
const MIN_GAP_CLOSED = 2 / 3;
const gapClosed = f => (f.best - f.random > 1e-9 ? (f.greedy - f.random) / (f.best - f.random) : 1);

const argJobs = process.argv.indexOf('--jobs');
const JOBS = argJobs > 0 ? Number(process.argv[argJobs + 1]) : Math.max(1, os.cpus().length - 2);

function run(args) {
  return new Promise((resolve, reject) => {
    const p = spawn(CLI, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '', err = '';
    p.stdout.on('data', d => { out += d; });
    p.stderr.on('data', d => { err += d; });
    p.on('close', code => (code === 0 ? resolve(out) : reject(new Error(`${args.join(' ')}: ${err || out}`))));
  });
}

async function pool(items, worker) {
  const results = new Array(items.length);
  let next = 0;
  await Promise.all(Array.from({ length: JOBS }, async () => {
    while (next < items.length) {
      const i = next++;
      results[i] = await worker(items[i], i);
    }
  }));
  return results;
}

// Five fixed challenge attacks, spread over both lanes and both depths: the
// spawns nearest to five anchor points in the band.
function challengeSpawns(spawns) {
  const ys = spawns.map(s => s[1]);
  const near = Math.min(...ys), far = Math.max(...ys), mid = (near + far) / 2;
  const anchors = [[3, near], [14, far], [8.5, mid], [3, far], [14, near]];
  const picked = [];
  for (const [ax, ay] of anchors) {
    let best = -1, bd = Infinity;
    spawns.forEach((s, i) => {
      if (picked.includes(i)) return;
      const d = Math.hypot(s[0] - ax, s[1] - ay);
      if (d < bd) { bd = d; best = i; }
    });
    picked.push(best);
  }
  return picked;
}

async function main() {
  fs.mkdirSync(TABLES, { recursive: true });
  let pairs = [];
  for (const a of ATTACKERS) for (const d of DEFENDERS) pairs.push([a, d]);
  // --cached-only: just the pairs whose tables already exist (a provisional
  // roster while a full run is still building).
  if (process.argv.includes('--cached-only')) {
    pairs = pairs.filter(([a, d]) => fs.existsSync(path.join(TABLES, `${a}_${d}.bin`)));
  }
  const pairAtk = [...new Set(pairs.map(p => p[0]))], pairDef = [...new Set(pairs.map(p => p[1]))];
  const cards = JSON.parse(execFileSync(CLI, ['serve'], {
    input: `cards ${[...new Set([...ATTACKERS, ...DEFENDERS])].join(' ')}\n`,
  }).toString().trim());
  const name = id => (cards.find(c => c.id === id) || { name: String(id) }).name;

  console.log(`${pairs.length} pairs, ${JOBS} parallel jobs`);
  const t0 = Date.now();
  let done = 0;
  const reports = await pool(pairs, async ([a, d]) => {
    const prefix = path.join(TABLES, `${a}_${d}`);
    if (!fs.existsSync(prefix + '.bin')) await run(['table', String(a), String(d), prefix]);
    const check = JSON.parse((await run(['check', String(a), String(d)])).trim());
    done++;
    if (done % 10 === 0) console.log(`  engine: ${done}/${pairs.length} (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
    return { a, d, prefix, check };
  });

  // --tables-only: build the outcome tables and stop (so the slow part can
  // run while the learner is still being tuned).
  if (process.argv.includes('--tables-only')) {
    console.log(`tables built (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
    return;
  }
  console.log('learner gate...');
  const matchups = {};
  const failing = [];
  for (const r of reports) {
    const table = loadTable(r.prefix);
    const ev = evalSpawns(table.S);
    const best = ev.reduce((s, i) => s + table.best[i], 0) / ev.length;
    const random = ev.reduce((s, i) => s + table.randomMean[i], 0) / ev.length;
    const seeds = SEEDS.map(seed => runLearner(table, { seed }, { budget: BUDGET, batch: BATCH, stopAt90: true }));
    const pointless = best < NOTHING_TO_LEARN;
    const learnerOk = pointless || seeds.every(s => s.triesTo90 !== null || gapClosed(s.final) >= MIN_GAP_CLOSED);
    const reasons = [];
    if (r.check.stalls > 0) reasons.push(`stall: ${r.check.firstStall}`);
    if (!learnerOk) reasons.push(`learner: final ${seeds.map(s => (100 * s.final.ratio).toFixed(0) + '%').join('/')} of best, ` +
                                 `${seeds.map(s => (100 * gapClosed(s.final)).toFixed(0) + '%').join('/')} of the gap closed`);
    if (table.S === 0) reasons.push('no spawn reaches a tower');
    const m = table.matchup;
    const d0 = m.d0.reduce((s, v) => s + v, 0) / m.d0.length;
    // The page scores the visitor and the learner on the five challenge
    // attacks, so its reference lines are measured on exactly those.
    const challenge = challengeSpawns(m.spawns);
    const over = f => +(challenge.reduce((s, i) => s + f(i), 0) / challenge.length).toFixed(4);
    matchups[`${r.a}_${r.d}`] = {
      attacker: r.a, defender: r.d, spawns: table.S, cells: table.C, d0Mean: Math.round(d0),
      evalSpawns: ev, best: +best.toFixed(4), random: +random.toFixed(4),
      triesTo90: seeds.map(s => s.triesTo90), finalRatio: seeds.map(s => +s.final.ratio.toFixed(3)),
      gapClosed: seeds.map(s => +gapClosed(s.final).toFixed(3)),
      challenge, challengeD0: over(i => m.d0[i]), challengeBest: over(i => table.best[i]),
      challengeRandom: over(i => table.randomMean[i]),
      pointless, ok: reasons.length === 0, reasons,
    };
    const tag = reasons.length ? `FAIL ${reasons.join('; ')}` : (pointless ? 'ok (nothing to learn)' : 'ok');
    console.log(`  ${name(r.a).padEnd(14)} vs ${name(r.d).padEnd(14)} best ${(100 * best).toFixed(0).padStart(3)}%  ` +
                `random ${(100 * random).toFixed(0).padStart(3)}%  90% at ${seeds.map(s => s.triesTo90 ?? '-').join('/')}  ` +
                `gap ${seeds.map(s => (100 * gapClosed(s.final)).toFixed(0)).join('/')}%  ${tag}`);
    if (reasons.length) failing.push([r.a, r.d]);
  }

  // Drop cards until no failing pair remains: first the card that fails in
  // the largest SHARE of its pairs (a Tombstone failing 1 of 1 goes before a
  // Hog Rider failing 1 of 9), then the one in more failing pairs.
  let atk = ATTACKERS.filter(a => pairAtk.includes(a)), def = DEFENDERS.filter(d => pairDef.includes(d));
  const dropped = [];
  for (;;) {
    const live = failing.filter(([a, d]) => atk.includes(a) && def.includes(d));
    if (!live.length) break;
    const fails = new Map();
    for (const [a, d] of live) {
      fails.set(`a${a}`, (fails.get(`a${a}`) || 0) + 1);
      fails.set(`d${d}`, (fails.get(`d${d}`) || 0) + 1);
    }
    const share = k => fails.get(k) / (k[0] === 'a' ? def.length : atk.length);
    // The default matchup's two cards are never the ones dropped.
    const [da, dd] = DEFAULT_MATCHUP.split('_');
    const keep = k => k === `a${da}` || k === `d${dd}`;
    const [worst] = [...fails.keys()].filter(k => !keep(k))
      .sort((x, y) => share(y) - share(x) || fails.get(y) - fails.get(x));
    if (!worst) throw new Error(`the default matchup ${DEFAULT_MATCHUP} itself fails`);
    const id = Number(worst.slice(1)), isAtk = worst[0] === 'a';
    const of = isAtk ? def.length : atk.length;
    if (isAtk) atk = atk.filter(x => x !== id); else def = def.filter(x => x !== id);
    dropped.push(`${isAtk ? 'attacker' : 'defender'} ${name(id)} (failed ${fails.get(worst)} of ${of} pairs)`);
  }
  for (const id of atk.slice(MAX_ATTACKERS)) dropped.push(`attacker ${name(id)} (passed; over the limit of ${MAX_ATTACKERS})`);
  atk = atk.slice(0, MAX_ATTACKERS);

  const commit = execFileSync('git', ['rev-parse', '--short', 'HEAD'], { cwd: REPO }).toString().trim();
  const roster = {
    generatedBy: 'tools/lab/curate.mjs', engineCommit: commit, budget: BUDGET,
    attackers: atk.map(id => cards.find(c => c.id === id)),
    defenders: def.map(id => cards.find(c => c.id === id)),
    defaultMatchup: DEFAULT_MATCHUP,
    matchups: Object.fromEntries(Object.entries(matchups).filter(([, m]) => atk.includes(m.attacker) && def.includes(m.defender))),
    dropped,
  };
  fs.writeFileSync(ROSTER, JSON.stringify(roster, null, 1) + '\n');
  console.log(`\n${atk.length} attackers, ${def.length} defenders pass; dropped: ${dropped.join(', ') || 'none'}`);
  console.log(`wrote ${ROSTER} (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
}

main().catch(e => { console.error(e); process.exit(1); });
