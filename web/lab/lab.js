// The Reflex Lab page: intro -> matchup -> the visitor's challenge -> the
// learner training live -> results side by side. All engine work happens in
// worker.js; this file draws, listens and keeps score.
/* global LabBoard, LabChart */
(function () {
  'use strict';

  // Uncaught errors, kept for tools/lab/check_lab_ui.py.
  window.__labErrors = [];
  window.addEventListener('error', e => window.__labErrors.push(String(e.message)));
  window.addEventListener('unhandledrejection', e => window.__labErrors.push(String(e.reason)));

  const $ = id => document.getElementById(id);
  const TICK_MS = 100;
  const TOUCH_AIM_OFFSET = 60;      // px the aim point floats above a finger
  const SNAP = 0.75;                // cells: how far a drop may be from a legal cell
  const worker = new Worker('worker.js');

  const board = new LabBoard($('board'));
  const miniYou = new LabBoard($('cmpYou'), { rows: [0, 24] });
  const miniAi = new LabBoard($('cmpAi'), { rows: [0, 24] });
  const chart = new LabChart($('chart'));

  const S = {
    stage: 'intro', arena: null, roster: null,
    key: null, matchup: null, entry: null, attacker: null, defender: null,
    legal: new Set(), cellAt: new Map(),
    round: 0, attempts: [], roundLive: false, placed: false, youScore: null,
    live: { prev: null, cur: null, at: 0 },
    drag: null, ghost: null,
    heat: null, greedy: null, preview: 0, tries: 0, curve: [], training: false,
    playback: null, beatToastShown: false,
    compare: null, cmpIdx: 0, cmpPlay: null,
    chartDirty: true,
  };

  // ---- worker messages ------------------------------------------------------

  worker.onmessage = ev => {
    const m = ev.data;
    switch (m.type) {
      case 'ready': return onReady(m);
      case 'matchup': return onMatchup(m);
      case 'progress': return onProgress(m);
      case 'frames': return onFrames(m);
      case 'liveFrames': return onLiveFrames(m);
      case 'placed': return onPlaced(m);
      case 'liveDone': return onLiveDone(m);
      case 'compare': return onCompare(m);
      case 'error': return showError(m.message);
    }
  };
  worker.onerror = e => showError(e.message || 'The worker failed to start.');

  function showError(msg) {
    console.error(msg);
    window.__labErrors.push(String(msg));
    banner(msg.includes('not built') ? 'The engine is not built yet' : 'Something went wrong');
    toast(msg, 8000);
  }

  function onReady(m) {
    S.arena = m.arena;
    S.roster = m.roster;
    [board, miniYou, miniAi].forEach(b => b.setArena(m.arena));
    buildTiles();
    const [a, d] = (m.roster.defaultMatchup || '').split('_').map(Number);
    selectMatchup(pickPair(a, d));
  }

  function pickPair(a, d) {
    const atk = S.roster.attackers.some(c => c.id === a) ? a : S.roster.attackers[0].id;
    const def = S.roster.defenders.some(c => c.id === d) ? d : S.roster.defenders[0].id;
    return [atk, def];
  }

  // ---- matchup -------------------------------------------------------------

  function buildTiles() {
    const make = (list, host, kind) => {
      host.innerHTML = '';
      for (const card of list) {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = `tile ${kind}`;
        b.setAttribute('role', 'radio');
        b.dataset.id = card.id;
        b.innerHTML = `<span class="sym${card.isBuilding ? ' bldg' : ''}">${esc(card.symbol)}</span>` +
                      `<span class="nm">${esc(card.name)}</span><span class="cost">${card.cost}</span>`;
        b.addEventListener('click', () => {
          const a = kind === 'atk' ? card.id : S.attacker.id;
          const d = kind === 'def' ? card.id : S.defender.id;
          if (a !== S.attacker.id || d !== S.defender.id) selectMatchup([a, d]);
        });
        host.appendChild(b);
      }
    };
    make(S.roster.attackers, $('attackerTiles'), 'atk');
    make(S.roster.defenders, $('defenderTiles'), 'def');
  }

  function selectMatchup([a, d]) {
    stopPlayback();
    worker.postMessage({ type: 'liveStop' });
    S.attacker = S.roster.attackers.find(c => c.id === a);
    S.defender = S.roster.defenders.find(c => c.id === d);
    document.querySelectorAll('.tile').forEach(t => {
      const on = Number(t.dataset.id) === (t.classList.contains('atk') ? a : d);
      t.setAttribute('aria-checked', String(on));
    });
    worker.postMessage({ type: 'matchup', key: `${a}_${d}` });
  }

  function onMatchup(m) {
    S.key = m.key; S.matchup = m.matchup; S.entry = m.entry;
    S.legal = new Set(m.matchup.cells.map(c => c[0] + ',' + c[1]));
    S.cellAt = new Map(m.matchup.cells.map((c, i) => [c[0] + ',' + c[1], i]));
    S.attempts = new Array(m.entry.challenge.length).fill(null);
    S.round = 0; S.youScore = null; S.curve = []; S.tries = 0; S.beatToastShown = false;
    S.compare = null; S.training = false;
    $('aboutParams').textContent = m.params.toLocaleString();
    $('factD0').textContent = `${fmtHp(m.entry.challengeD0)} HP`;
    $('factRandom').textContent = pct(m.entry.challengeRandom);
    $('factBest').textContent = pct(m.entry.challengeBest);
    $('factNote').textContent = m.entry.pointless
      ? `A ${S.defender.name} can barely touch a ${S.attacker.name}: even the best drop saves almost nothing. An honest result, not a bug.`
      : 'Tower HP lost with no defence, and the share saved by a random drop and by the best drop, found by trying every cell and every delay on the five attacks.';
    $('matchupSummaryText').textContent = `${S.attacker.name} vs your ${S.defender.name}`;
    $('traySymbol').textContent = S.defender.symbol;
    $('traySymbol').classList.toggle('bldg', S.defender.isBuilding);
    $('trayName').textContent = S.defender.name;
    renderRounds();
    updateChartRefs();
    setStage(S.stage === 'intro' ? 'intro' : 'challenge');
  }

  // ---- stages --------------------------------------------------------------

  function setStage(stage) {
    S.stage = stage;
    const shown = stage === 'intro' ? 'challenge' : stage;
    $('stageChallenge').hidden = shown !== 'challenge';
    $('stageTrain').hidden = shown !== 'train';
    $('stageResults').hidden = shown !== 'results';
    const order = ['challenge', 'train', 'results'];
    document.querySelectorAll('#stepper li').forEach(li => {
      const i = order.indexOf(li.dataset.stage), cur = order.indexOf(shown);
      li.classList.toggle('active', i === cur);
      li.classList.toggle('done', i < cur);
    });
    $('tray').hidden = shown !== 'challenge';
    if (shown === 'challenge') {
      const done = S.attempts.filter(Boolean).length;
      $('btnRound').textContent = done >= S.attempts.length ? 'Watch the AI try' : `Start round ${done + 1}`;
      $('btnRound').disabled = false;
      $('btnSkip').hidden = done >= S.attempts.length;
      setTray(done >= S.attempts.length ? 'done' : 'idle');
      banner(null);
    }
    if (shown === 'train') {
      $('btnResults').disabled = S.tries < 500;
      banner(S.tries ? null : `Press Train. The network starts knowing nothing.`);
    }
    S.chartDirty = true;
  }

  // ---- the challenge ---------------------------------------------------------

  function renderRounds() {
    const host = $('rounds');
    host.innerHTML = '';
    S.attempts.forEach((a, i) => {
      const d = document.createElement('div');
      d.className = 'round' + (i === S.round && S.roundLive ? ' current' : '');
      d.innerHTML = `Round ${i + 1}<b>${a ? pct(a.score) : '-'}</b>`;
      host.appendChild(d);
    });
  }

  $('btnRound').addEventListener('click', () => {
    const done = S.attempts.filter(Boolean).length;
    if (done >= S.attempts.length) { goTrain(); return; }
    startRound(done);
  });
  $('btnSkip').addEventListener('click', goTrain);
  $('matchupSummary').addEventListener('click', () => {
    const open = !$('matchupPanel').classList.contains('open');
    $('matchupPanel').classList.toggle('open', open);
    $('matchupSummary').setAttribute('aria-expanded', String(open));
  });

  // On a phone the board is below the controls: bring it into view.
  function showBoard() {
    if (window.matchMedia('(max-width: 760px)').matches) {
      $('boardWrap').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  function startRound(round) {
    stopPlayback();
    S.round = round; S.placed = false; S.roundLive = false;
    S.live = { prev: null, cur: null, at: 0 };
    $('btnRound').disabled = true;
    $('btnSkip').hidden = true;
    setTray('idle');
    banner(null);
    showBoard();
    countdown(() => {
      S.roundLive = true;
      renderRounds();
      setTray('ready');
      banner(`Round ${round + 1} of ${S.attempts.length} · stop the ${S.attacker.name}`);
      worker.postMessage({ type: 'liveStart', round });
    });
  }

  function countdown(done) {
    const el = $('countdown');
    let n = 3;
    el.hidden = false;
    const next = () => {
      if (n === 0) { el.hidden = true; done(); return; }
      el.innerHTML = `<span>${n}</span>`;
      n--;
      setTimeout(next, 650);
    };
    next();
  }

  function onLiveFrames(m) {
    for (const f of m.frames) { S.live.prev = S.live.cur || f; S.live.cur = f; S.live.at = performance.now(); }
  }

  function onPlaced(m) {
    if (!m.ok) { S.placed = false; setTray('ready'); toast('Too late: this round is over.'); return; }
    S.attempts[S.round] = Object.assign(S.attempts[S.round] || {}, { cell: m.cell, tick: m.tick });
    $('trayHint').textContent = `Dropped at ${(m.tick / 10).toFixed(1)} s`;
  }

  function onLiveDone(m) {
    S.roundLive = false;
    S.drag = null; S.ghost = null;
    const score = Math.max(-1, Math.min(1, (m.d0 - m.damage) / m.d0));
    S.attempts[m.round] = { cell: m.placedTick >= 0 ? m.cell : -1, tick: Math.max(0, m.placedTick), damage: m.damage, d0: m.d0, score };
    renderRounds();
    const lost = m.damage > 0 ? `${fmtHp(m.damage)} of ${fmtHp(m.d0)} HP lost` : 'tower untouched';
    banner(m.placedTick < 0 ? `No drop: ${lost}` : `Saved ${pct(score)} · ${lost}`);
    const done = S.attempts.filter(Boolean).length;
    if (done >= S.attempts.length) {
      S.youScore = S.attempts.reduce((s, a) => s + a.score, 0) / S.attempts.length;
      updateChartRefs();
      $('challengeText').innerHTML = `Your average: <b>${pct(S.youScore)}</b> of the damage prevented. Now let a network that has never seen the game try the same five attacks.`;
    }
    setTimeout(() => { if (S.stage === 'challenge' && !S.roundLive) setStage('challenge'); }, 900);
  }

  function setTray(state) {
    const card = $('trayCard');
    card.classList.toggle('ready', state === 'ready');
    card.classList.toggle('used', state === 'used' || state === 'done' || state === 'idle');
    $('trayHint').textContent = state === 'ready' ? 'Drag onto the board'
      : state === 'used' ? $('trayHint').textContent
      : state === 'done' ? 'All five rounds played' : 'Starts with the round';
  }

  // Drag and drop, one code path for mouse, pen and touch (pointer events).
  const tray = $('trayCard'), avatar = $('dragAvatar');
  tray.addEventListener('pointerdown', e => {
    if (!S.roundLive || S.placed || e.button > 0) return;
    e.preventDefault();
    tray.setPointerCapture(e.pointerId);
    S.drag = { id: e.pointerId, touch: e.pointerType !== 'mouse' };
    tray.classList.add('dragging');
    avatar.textContent = S.defender.symbol;
    avatar.classList.toggle('bldg', S.defender.isBuilding);
    avatar.style.transition = 'none';
    avatar.hidden = false;
    moveDrag(e);
  });
  tray.addEventListener('pointermove', e => { if (S.drag && e.pointerId === S.drag.id) moveDrag(e); });
  tray.addEventListener('pointerup', e => { if (S.drag && e.pointerId === S.drag.id) endDrag(e, false); });
  tray.addEventListener('pointercancel', e => { if (S.drag && e.pointerId === S.drag.id) endDrag(e, true); });
  tray.addEventListener('lostpointercapture', e => { if (S.drag && e.pointerId === S.drag.id) endDrag(e, true); });

  function moveDrag(e) {
    const aimY = e.clientY - (S.drag.touch ? TOUCH_AIM_OFFSET : 0);
    avatar.style.transform = `translate(${e.clientX}px, ${aimY}px)`;
    const g = board.toGame(e.clientX, aimY);
    if (!g.inside || !S.roundLive) {
      S.ghost = null;
      avatar.classList.remove('over-board');
      return;
    }
    avatar.classList.add('over-board');
    const snap = nearestLegal(g.x, g.y);
    S.ghost = snap
      ? { x: snap[0], y: snap[1], legal: true, cell: S.cellAt.get(snap[0] + ',' + snap[1]), symbol: S.defender.symbol, building: S.defender.isBuilding }
      : { x: Math.round(g.x), y: Math.round(g.y), legal: false, building: S.defender.isBuilding };
  }

  function nearestLegal(x, y) {
    let best = null, bd = SNAP;
    for (const c of S.matchup.cells) {
      const d = Math.hypot(c[0] - x, c[1] - y);
      if (d < bd) { bd = d; best = c; }
    }
    return best;
  }

  function endDrag(e, cancelled) {
    const ghost = S.ghost;
    S.drag = null; S.ghost = null;
    tray.classList.remove('dragging');
    if (!cancelled && ghost && ghost.legal && S.roundLive && !S.placed) {
      S.placed = true;
      avatar.hidden = true;
      setTray('used');
      $('trayHint').textContent = 'Dropped';
      worker.postMessage({ type: 'livePlace', cell: ghost.cell });
      return;
    }
    if (!cancelled && ghost && !ghost.legal) toast(`You can't drop a ${S.defender.name} there.`);
    // Fly back to the tray.
    const r = tray.getBoundingClientRect();
    avatar.classList.remove('over-board');
    avatar.style.transition = 'transform .22s cubic-bezier(.2,.7,.2,1), opacity .22s';
    avatar.style.transform = `translate(${r.left + 30}px, ${r.top + r.height / 2}px)`;
    setTimeout(() => { avatar.hidden = true; avatar.style.transition = 'none'; }, 230);
  }

  // ---- training --------------------------------------------------------------

  function goTrain() {
    worker.postMessage({ type: 'liveStop' });
    S.roundLive = false;
    setStage('train');
  }

  $('btnTrain').addEventListener('click', () => {
    S.training = !S.training;
    $('btnTrain').textContent = S.training ? 'Pause' : 'Train';
    if (S.training) { stopPlayback(); banner(null); showBoard(); }
    worker.postMessage({ type: 'train', on: S.training });
  });
  $('btnReset').addEventListener('click', () => {
    S.training = false; S.curve = []; S.tries = 0; S.beatToastShown = false;
    $('btnTrain').textContent = 'Train';
    $('btnResults').disabled = true;
    worker.postMessage({ type: 'reset' });
  });
  document.querySelectorAll('.seg-btn').forEach(b => b.addEventListener('click', () => {
    document.querySelectorAll('.seg-btn').forEach(x => x.classList.toggle('active', x === b));
    worker.postMessage({ type: 'speed', speed: b.dataset.speed });
  }));
  $('btnShow').addEventListener('click', () => { showBoard(); worker.postMessage({ type: 'showme' }); });
  $('btnResults').addEventListener('click', () => {
    if (S.training) $('btnTrain').click();
    worker.postMessage({ type: 'compare', attempts: S.attempts });
  });

  function onProgress(m) {
    S.heat = m.heat; S.greedy = m.greedy; S.preview = m.spawn; S.tries = m.tries; S.curve = m.curve;
    $('statTries').textContent = m.tries.toLocaleString();
    $('statRate').textContent = Math.round(m.rate).toLocaleString();
    $('statTime').textContent = `${(m.elapsed / 1000).toFixed(1)}s`;
    if (S.stage === 'train') $('btnResults').disabled = m.tries < 500;
    const cell = S.matchup.cells[m.greedy.cell];
    const delay = (m.greedy.delay * S.matchup.delayTickStep / 10).toFixed(1);
    $('guess').innerHTML = m.tries === 0
      ? `Drag the red attacker around the enemy side: the heatmap shows where the network would drop your ${esc(S.defender.name)}. Right now it has no idea.`
      : `For this attack its best guess is column ${cell[0]}, row ${cell[1]}, <b>${delay} s</b> after the ${esc(S.attacker.name)} is dropped.`;
    const last = m.curve[m.curve.length - 1];
    if (last && S.youScore != null && !S.beatToastShown && last.v > S.youScore + 1e-6) {
      S.beatToastShown = true;
      toast(`The network just beat your score, after ${last.tries.toLocaleString()} tries.`);
    }
    if (!m.training && S.training) { S.training = false; $('btnTrain').textContent = 'Train'; }
    S.chartDirty = true;
  }

  // Dragging on the enemy side moves the preview attack; the heatmap follows.
  const canvas = $('board');
  canvas.addEventListener('pointerdown', e => {
    if (S.stage !== 'train' || !S.matchup || S.playback) return;
    const g = board.toGame(e.clientX, e.clientY);
    if (!g.inside || g.y < S.arena.riverStart) return;
    canvas.setPointerCapture(e.pointerId);
    S.previewDrag = e.pointerId;
    movePreview(g);
  });
  canvas.addEventListener('pointermove', e => {
    if (S.previewDrag === e.pointerId) movePreview(board.toGame(e.clientX, e.clientY));
  });
  const endPreview = e => { if (S.previewDrag === e.pointerId) S.previewDrag = null; };
  canvas.addEventListener('pointerup', endPreview);
  canvas.addEventListener('pointercancel', endPreview);

  function movePreview(g) {
    let best = S.preview, bd = Infinity;
    S.matchup.spawns.forEach((s, i) => {
      const d = Math.hypot(s[0] - g.x, s[1] - g.y);
      if (d < bd) { bd = d; best = i; }
    });
    if (best !== S.preview) { S.preview = best; worker.postMessage({ type: 'preview', spawn: best }); }
  }

  function onFrames(m) {
    if (m.purpose !== 'showme') return;
    const c = S.matchup.cells[m.drop.cell];
    startPlayback(m.data.frames, { x: c[0], y: c[1], tick: m.drop.tick, building: S.defender.isBuilding }, () => {
      const saved = (S.matchup.d0[m.spawn] - m.data.damage) / S.matchup.d0[m.spawn];
      banner(`Its answer saved ${pct(saved)} of the damage`);
    });
    banner('Its current best answer, played by the engine');
  }

  function startPlayback(frames, drop, onEnd) {
    S.playback = { frames, drop, start: performance.now(), onEnd };
  }

  function stopPlayback() { S.playback = null; }

  // ---- results ---------------------------------------------------------------

  function onCompare(m) {
    S.compare = m;
    const ai = m.rounds.reduce((s, r) => s + (r.d0 - r.ai.damage) / r.d0, 0) / m.rounds.length;
    const played = m.rounds.every(r => r.you);
    const secs = Math.max(1, Math.round(m.elapsed / 1000));
    $('verdict').innerHTML = played
      ? `You saved <span class="you">${pct(S.youScore)}</span>. The learner saved <span class="ai">${pct(ai)}</span> after ${secs} s of training.`
      : `The learner saved <span class="ai">${pct(ai)}</span> after ${secs} s and ${m.tries.toLocaleString()} tries. Take the challenge to see how you compare.`;
    const host = $('cmpRounds');
    host.innerHTML = '';
    m.rounds.forEach((r, i) => {
      const d = document.createElement('div');
      d.className = 'round';
      const aiPct = pct((r.d0 - r.ai.damage) / r.d0);
      d.innerHTML = `${i + 1}<b>${r.you ? pct((r.d0 - r.you.damage) / r.d0) : '-'}</b><b style="color:var(--blue-team)">${aiPct}</b>`;
      d.addEventListener('click', () => playCompare(i));
      host.appendChild(d);
    });
    $('btnAgain').textContent = played ? 'Try another matchup' : 'Take the challenge';
    setStage('results');
    playCompare(0);
  }

  function playCompare(i) {
    S.cmpIdx = i;
    document.querySelectorAll('#cmpRounds .round').forEach((d, k) => d.classList.toggle('current', k === i));
    const r = S.compare.rounds[i];
    const drop = side => {
      if (!side || !side.drop) return null;
      const c = S.matchup.cells[side.drop.cell];
      return { x: c[0], y: c[1], tick: side.drop.tick, building: S.defender.isBuilding };
    };
    S.cmpPlay = { start: performance.now(), you: r.you && r.you.frames, ai: r.ai.frames, dropYou: drop(r.you), dropAi: drop(r.ai) };
    $('cmpYouScore').textContent = r.you ? `saved ${pct((r.d0 - r.you.damage) / r.d0)}` : 'not played';
    $('cmpAiScore').textContent = `saved ${pct((r.d0 - r.ai.damage) / r.d0)}`;
  }

  $('btnBackTrain').addEventListener('click', () => setStage('train'));
  $('btnAgain').addEventListener('click', () => {
    if (S.compare && !S.compare.rounds.every(r => r.you)) {
      S.attempts = S.attempts.map(() => null);
      renderRounds();
      setStage('challenge');
      return;
    }
    document.querySelector('.matchup-panel').scrollIntoView({ behavior: 'smooth' });
    toast('Pick a new attacker or defender on the left.');
  });

  // ---- chart ------------------------------------------------------------------

  function updateChartRefs() {
    if (!S.entry) return;
    chart.setRefs({ best: S.entry.challengeBest, random: S.entry.challengeRandom, you: S.youScore });
    const C = LabChart.COLORS;
    $('legend').innerHTML =
      `<span><i style="border-color:${C.ai}"></i>The network</span>` +
      `<span><i class="dash" style="border-color:${C.best}"></i>Best possible</span>` +
      `<span><i class="dash" style="border-color:${C.random}"></i>Random drop</span>` +
      (S.youScore != null ? `<span><i class="dash" style="border-color:${C.you}"></i>You</span>` : '');
    S.chartDirty = true;
  }

  // ---- the render loop ------------------------------------------------------------

  function frameAt(frames, t0, now) {
    const pos = (now - t0) / TICK_MS;
    const i = Math.min(frames.length - 1, Math.max(0, Math.floor(pos)));
    return { a: frames[i], b: frames[Math.min(frames.length - 1, i + 1)], frac: Math.min(1, pos - i), i, done: pos >= frames.length - 1 };
  }

  function render(now) {
    if (S.arena && S.matchup) {
      const st = {};
      if (S.playback) {
        const f = frameAt(S.playback.frames, S.playback.start, now);
        Object.assign(st, { frameA: f.a, frameB: f.b, frac: f.frac, drop: S.playback.drop, clockTick: f.a.t });
        if (f.done && now - S.playback.start > S.playback.frames.length * TICK_MS + 1200) {
          const end = S.playback.onEnd;
          stopPlayback();
          if (end) end();
        }
      } else if (S.stage === 'challenge' && (S.roundLive || S.live.cur)) {
        const frac = Math.min(1, (now - S.live.at) / TICK_MS);
        Object.assign(st, { frameA: S.live.prev, frameB: S.live.cur, frac });
        if (S.drag) st.forbid = S.legal;
        st.ghost = S.ghost;
      } else if (S.stage === 'train' || S.stage === 'results') {
        const delay = S.greedy ? (S.greedy.delay * S.matchup.delayTickStep / 10).toFixed(1) + 's' : null;
        Object.assign(st, { heat: S.heat, cells: S.matchup.cells, spawns: S.matchup.spawns,
                            preview: S.preview, previewSymbol: S.attacker.symbol,
                            greedy: S.greedy && S.tries > 0 ? { cell: S.greedy.cell, label: delay } : null });
      }
      board.render(st);

      if (S.stage === 'results' && S.cmpPlay) {
        const p = S.cmpPlay;
        const len = Math.max(p.ai.length, p.you ? p.you.length : 0);
        const fa = frameAt(p.ai, p.start, now);
        miniAi.render({ frameA: fa.a, frameB: fa.b, frac: fa.frac, drop: p.dropAi, clockTick: fa.a.t });
        if (p.you) {
          const fy = frameAt(p.you, p.start, now);
          miniYou.render({ frameA: fy.a, frameB: fy.b, frac: fy.frac, drop: p.dropYou, clockTick: fy.a.t });
        } else {
          miniYou.render({});
        }
        if (now - p.start > len * TICK_MS + 1500) playCompare((S.cmpIdx + 1) % S.compare.rounds.length);
      }
    }
    if (S.chartDirty) {
      chart.setPoints(S.curve.map(p => ({ tries: p.tries, v: p.v })));
      chart.render();
      S.chartDirty = false;
    }
    requestAnimationFrame(render);
  }

  // ---- small helpers --------------------------------------------------------------

  function pct(v) { return v == null || Number.isNaN(v) ? '-' : `${Math.round(v * 100)}%`; }
  function fmtHp(v) { return Math.round(v).toLocaleString(); }
  function esc(s) { return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

  let bannerTimer = null;
  function banner(text) {
    const el = $('banner');
    clearTimeout(bannerTimer);
    if (!text) { el.hidden = true; return; }
    el.textContent = text;
    el.hidden = false;
    el.style.animation = 'none'; void el.offsetWidth; el.style.animation = '';
  }

  let toastTimer = null;
  function toast(text, ms = 3200) {
    const el = $('toast');
    el.textContent = text;
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.hidden = true; }, ms);
  }

  // ---- intro and about ----------------------------------------------------------------

  $('btnStart').addEventListener('click', () => {
    $('intro').hidden = true;
    setStage('challenge');
  });
  $('btnAbout').addEventListener('click', () => { $('about').hidden = false; });
  $('btnAboutClose').addEventListener('click', () => { $('about').hidden = true; });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !$('about').hidden) $('about').hidden = true;
  });
  window.addEventListener('resize', () => { S.chartDirty = true; });

  // Test hooks for tools/lab/check_lab_ui.py.
  window.__lab = { state: S, board, cellCenterClient(x, y) {
    const p = board.toCanvas(x, y), r = $('board').getBoundingClientRect();
    return { x: r.left + p.x, y: r.top + p.y };
  } };

  document.fonts.ready.then(() => { board.bg = null; S.chartDirty = true; });
  worker.postMessage({ type: 'init' });
  requestAnimationFrame(render);
})();
