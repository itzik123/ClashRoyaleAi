// The Reflex Lab's background thread: owns the engine and the learner, runs
// the training loop, and steps the live challenge in real time, so the page
// never stutters. Talks to lab.js by messages only.
/* global Learner, EngineClient */
importScripts('learner.js', 'engine_client.js');

const BASE = new URL('./', self.location).href;
// "Watch" mode: slow enough at the start to see the heatmap sharpen, then
// quicker, so a hard matchup does not take minutes. The counter shows the rate.
const WATCH_RATE = tries => (tries < 1500 ? 60 : 300);
const TICK_MS = 100;              // the engine's 10 ticks per second, in real time

let engine = null, roster = null, arena = null;
let matchup = null, entry = null, learner = null, preview = 0;
let training = false, looping = false, speed = 'watch';
let curve = [], heatHist = [], elapsedMs = 0, seedCounter = 1;
let live = null;

const post = (type, data = {}, transfer) => self.postMessage(Object.assign({ type }, data), transfer || []);

self.onmessage = async (ev) => {
  const m = ev.data;
  try {
    switch (m.type) {
      case 'init': return await init();
      case 'matchup': return await setMatchup(m.key);
      case 'train': training = !!m.on; if (training) loop(); return;
      case 'speed': speed = m.speed; return;
      case 'reset': training = false; await waitLoop(); newLearner(); return progress();
      case 'preview': preview = m.spawn; return progress();
      case 'showme': return await showMe();
      case 'liveStart': return await liveStart(m.round);
      case 'livePlace': return await livePlace(m.cell);
      case 'liveStop': stopLive(); return;
      case 'compare': return await compare(m.attempts);
    }
  } catch (err) {
    post('error', { message: String(err && err.message || err) });
  }
};

async function init() {
  engine = await EngineClient.connect(BASE);
  roster = await (await fetch(BASE + 'roster.json', { cache: 'no-store' })).json();
  arena = await engine.arena();
  const demo = await recordDemo();
  post('ready', { arena, roster, backend: engine.kind, demo });
}

// The how-to-play demo is always a Hog Rider against a Cannon, so it gives
// away no matchup: the Cannon dropped 2 s after the Hog, left of the lane,
// where it pulls the Hog off the tower and the towers finish it. The whole
// rollout plays, to the Hog's last hit point: the tower is untouched and the
// Cannon is still standing (a drop a cell to the right, (11,9), saves the
// tower too and loses the Cannon). Recorded from the engine, like every
// other frame on the page.
const DEMO = { attacker: 15, defender: 25, spawn: [14, 20], cell: [9, 10], dropTick: 20 };
async function recordDemo() {
  try {
    const m = await engine.matchup(DEMO.attacker, DEMO.defender);
    const at = (list, [x, y]) => list.findIndex(p => p[0] === x && p[1] === y);
    const s = at(m.spawns, DEMO.spawn), c = at(m.cells, DEMO.cell);
    if (s < 0 || c < 0) return null;
    const data = await engine.frames(s, c, DEMO.dropTick);
    return {
      attacker: DEMO.attacker, defender: DEMO.defender, spawn: DEMO.spawn,
      drop: { x: DEMO.cell[0], y: DEMO.cell[1], tick: DEMO.dropTick },
      frames: data.frames, damage: data.damage, survival: data.survival, cells: m.cells,
    };
  } catch (err) {
    return null;   // the page simply skips the demo
  }
}

async function setMatchup(key) {
  training = false;
  stopLive();
  await waitLoop();
  entry = roster.matchups[key];
  if (!entry) throw new Error(`no matchup ${key} in the roster`);
  matchup = await engine.matchup(entry.attacker, entry.defender);
  newLearner();
  preview = entry.challenge[0];
  post('matchup', { key, matchup, entry, params: paramCount() });
  progress();
}

function newLearner() {
  learner = new Learner({
    spawns: matchup.spawns, cells: matchup.cells, delaySteps: matchup.delaySteps,
    width: arena.width, seed: (Date.now() ^ (seedCounter++ * 2654435761)) >>> 0,
  });
  curve = [];
  heatHist = [];
  elapsedMs = 0;
}

function paramCount() {
  let n = 0;
  for (const k in learner.params) n += learner.params[k].length;
  return n;
}

// Share of the no-defence damage prevented: the score on the page, for the
// visitor and the learner alike.
function prevented(spawn, damage) {
  const d0 = matchup.d0[spawn];
  return Math.max(-1, Math.min(1, (d0 - damage) / d0));
}

// -> { damage, survival } per action.
async function rollouts(actions) {
  const triples = new Int32Array(actions.length * 3);
  actions.forEach((a, i) => { triples[3 * i] = a.spawn; triples[3 * i + 1] = a.cell; triples[3 * i + 2] = a.delay; });
  return engine.batch(triples);
}

// The learner's current best answer on the five challenge attacks, scored as
// the visitor is: damage prevented.
async function evalChallenge() {
  const acts = entry.challenge.map(s => learner.greedy(s));
  const { damage } = await rollouts(acts);
  let sum = 0;
  acts.forEach((a, i) => { sum += prevented(a.spawn, damage[i]); });
  return sum / acts.length;
}

function evalEvery(tries) {
  return tries < 1000 ? 64 : tries < 5000 ? 256 : 1024;
}

async function loop() {
  if (looping) return;
  looping = true;
  const start = performance.now() - elapsedMs;
  let lastPost = 0, nextEval = curve.length ? learner.tries + evalEvery(learner.tries) : 0;
  const rateWin = [];
  let paceAt = performance.now();
  try {
    while (training) {
      if (learner.tries >= nextEval) {
        curve.push({ tries: learner.tries, v: await evalChallenge() });
        // Where it would drop the card on each challenge attack, for the
        // results page to replay the heatmap converging.
        heatHist.push({ tries: learner.tries, p: entry.challenge.map(s => learner.cellProbs(s)) });
        nextEval = learner.tries + evalEvery(learner.tries);
      }
      const batch = learner.sampleBatch(speed === 'watch' ? 16 : 64);
      const out = await rollouts(batch);
      // Tower first, defender second: Learner.rewards, shared with tools/lab.
      learner.update(batch, learner.rewards(batch, batch.map((a, i) => prevented(a.spawn, out.damage[i])), out.survival));

      const now = performance.now();
      rateWin.push([now, learner.tries]);
      while (rateWin.length > 2 && now - rateWin[0][0] > 1000) rateWin.shift();
      elapsedMs = now - start;
      if (now - lastPost > 120) {
        const [t0, n0] = rateWin[0];
        progress(now - t0 > 50 ? (learner.tries - n0) * 1000 / (now - t0) : 0);
        lastPost = now;
      }
      if (speed === 'watch') {
        // Hold to WATCH_RATE so the heatmap can be seen sharpening.
        paceAt += batch.length * 1000 / WATCH_RATE(learner.tries);
        const wait = paceAt - performance.now();
        if (wait < -250) paceAt = performance.now();   // never "catch up" in a burst
        await sleep(Math.max(0, wait));
      } else {
        paceAt = performance.now();
        await sleep(0);   // let messages in (pause, preview drags) between batches
      }
    }
  } finally {
    looping = false;
    progress(0);
  }
}

function waitLoop() {
  return new Promise(resolve => {
    const check = () => (looping ? setTimeout(check, 10) : resolve());
    check();
  });
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function progress(rate = 0) {
  if (!learner) return;
  const heat = learner.cellProbs(preview);
  const g = learner.greedy(preview);
  post('progress', {
    tries: learner.tries, rate, elapsed: elapsedMs, heat, greedy: { cell: g.cell, delay: g.delay },
    spawn: preview, curve, training,
  }, [heat.buffer]);
}

async function showMe() {
  const g = learner.greedy(preview);
  const tick = g.delay * matchup.delayTickStep;
  const data = await engine.frames(preview, g.cell, tick);
  post('frames', { purpose: 'showme', data, drop: { cell: g.cell, tick }, spawn: preview });
}

// ---- the live challenge ----------------------------------------------------

async function liveStart(round) {
  stopLive();
  training = false;
  await waitLoop();
  const spawn = entry.challenge[round];
  await engine.liveStart(spawn);
  const session = { spawn, round, stopped: false, t0: performance.now(), n: 0 };
  live = session;
  const step = async () => {
    if (session.stopped) return;
    const frames = await engine.liveStep(1);
    if (session.stopped) return;
    post('liveFrames', { frames, round });
    const state = await engine.liveState();
    if (state.done) {
      session.stopped = true;
      post('liveDone', { round, spawn, damage: state.damage, survival: state.survival, placedTick: state.placedTick,
                         cell: session.cell ?? -1, d0: matchup.d0[spawn] });
      return;
    }
    session.n++;
    // Drift-corrected: tick n is due at t0 + n * TICK_MS.
    setTimeout(step, Math.max(0, session.t0 + session.n * TICK_MS - performance.now()));
  };
  step();
}

async function livePlace(cell) {
  if (!live || live.stopped) return post('placed', { ok: false });
  const ok = await engine.livePlace(cell);
  if (ok) live.cell = cell;
  const state = await engine.liveState();
  post('placed', { ok, cell, tick: state.placedTick });
}

function stopLive() {
  if (live) live.stopped = true;
  live = null;
}

// ---- results: the visitor's five tries and the learner's, as replays ---------

async function compare(attempts) {
  const rounds = [];
  for (let i = 0; i < entry.challenge.length; i++) {
    const spawn = entry.challenge[i], d0 = matchup.d0[spawn];
    const g = learner.greedy(spawn);
    const aiTick = g.delay * matchup.delayTickStep;
    const ai = await engine.frames(spawn, g.cell, aiTick);
    const a = attempts && attempts[i];
    let you = null;
    if (a) {
      const f = await engine.frames(spawn, a.cell, a.cell >= 0 ? a.tick : 0);
      you = { frames: f.frames, damage: f.damage, survival: f.survival, drop: a.cell >= 0 ? { cell: a.cell, tick: a.tick } : null };
    }
    const heat = heatSteps(i, spawn);
    rounds.push({ spawn, d0, heat, ai: { frames: ai.frames, damage: ai.damage, survival: ai.survival, drop: { cell: g.cell, tick: aiTick } }, you });
  }
  post('compare', { rounds, tries: learner.tries, elapsed: elapsedMs });
}

// Up to HEAT_STEPS snapshots of challenge attack i's heatmap, from the first
// try to now, evenly spaced through the recorded history.
const HEAT_STEPS = 36;
function heatSteps(i, spawn) {
  const all = heatHist.map(h => ({ tries: h.tries, p: h.p[i] }));
  if (!all.length || all[all.length - 1].tries !== learner.tries) all.push({ tries: learner.tries, p: learner.cellProbs(spawn) });
  if (all.length <= HEAT_STEPS) return all;
  const out = [];
  for (let k = 0; k < HEAT_STEPS; k++) out.push(all[Math.round(k * (all.length - 1) / (HEAT_STEPS - 1))]);
  return out;
}
