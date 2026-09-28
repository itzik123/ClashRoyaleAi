// The Reflex Lab page: intro -> a short demo -> the visitor's challenge -> the
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
  const PREVIEW_REACH = 3;          // cells: how near a spawn a preview drag must start

  // The how-to-play demo, played slightly slower than real time.
  const DEMO_TICK_MS = 125;
  const DEMO_COUNT_MS = 520;        // per countdown number
  const DEMO_HAND_TICKS = 13;       // the hand starts this many ticks before the drop

  // Results: the heatmap converging, a beat on its answer, then the replay.
  const CMP_HEAT_MS = 3600;
  const CMP_ANSWER_MS = 900;
  const CMP_TICK_MS = 150;
  const CMP_HOLD_MS = 1800;

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
    compare: null, cmpIdx: 0, cmpPlay: null, verdictShown: false,
    demo: null, demoPlay: null, demoSeen: false,
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
    S.demo = m.demo;
    [board, miniYou, miniAi].forEach(b => b.setArena(m.arena));
    buildTiles();
    const [a, d] = (m.roster.defaultMatchup || '').split('_').map(Number);
    selectMatchup(pickPair(a, d));
  }

  const hasPair = (a, d) => !!S.roster.matchups[`${a}_${d}`];

  function pickPair(a, d) {
    const atk = S.roster.attackers.some(c => c.id === a) ? a : S.roster.attackers[0].id;
    let def = S.roster.defenders.some(c => c.id === d) ? d : S.roster.defenders[0].id;
    if (!hasPair(atk, def)) def = (S.roster.defenders.find(c => hasPair(atk, c.id)) || { id: def }).id;
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
          if (b.getAttribute('aria-disabled') === 'true') return;
          const a = kind === 'atk' ? card.id : S.attacker.id;
          const d = kind === 'def' ? card.id : S.defender.id;
          if (a !== S.attacker.id || d !== S.defender.id) selectMatchup(pickPair(a, d));
        });
        host.appendChild(b);
      }
    };
    make(S.roster.attackers, $('attackerTiles'), 'atk');
    make(S.roster.defenders, $('defenderTiles'), 'def');
  }

  function selectMatchup([a, d]) {
    stopPlayback();
    stopDemo();
    worker.postMessage({ type: 'liveStop' });
    S.roundLive = false;
    cancelCountdown();
    S.attacker = S.roster.attackers.find(c => c.id === a);
    S.defender = S.roster.defenders.find(c => c.id === d);
    document.querySelectorAll('.tile').forEach(t => {
      const isAtk = t.classList.contains('atk'), id = Number(t.dataset.id);
      t.setAttribute('aria-checked', String(id === (isAtk ? a : d)));
      // Only pairings the curation suite measured are offered.
      const ok = isAtk ? true : hasPair(a, id);
      t.setAttribute('aria-disabled', String(!ok));
      t.title = ok ? '' : `Not measured against the ${S.attacker.name}`;
    });
    worker.postMessage({ type: 'matchup', key: `${a}_${d}` });
  }

  function onMatchup(m) {
    S.key = m.key; S.matchup = m.matchup; S.entry = m.entry;
    S.legal = new Set(m.matchup.cells.map(c => c[0] + ',' + c[1]));
    S.cellAt = new Map(m.matchup.cells.map((c, i) => [c[0] + ',' + c[1], i]));
    S.attempts = new Array(m.entry.challenge.length).fill(null);
    S.round = 0; S.youScore = null; S.curve = []; S.tries = 0; S.beatToastShown = false;
    S.compare = null; S.cmpPlay = null; S.training = false;
    S.live = { prev: null, cur: null, at: 0 };
    $('aboutParams').textContent = m.params.toLocaleString();
    $('factD0').textContent = `${fmtHp(m.entry.challengeD0)} HP`;
    $('factRandom').textContent = pct(m.entry.challengeRandom);
    $('factBest').textContent = pct(m.entry.challengeBest);
    $('factNote').textContent = m.entry.pointless
      ? (S.attacker.isFlying
        ? `A ${S.defender.name} only hits ground units, so it cannot touch a flying ${S.attacker.name}: even the best drop saves nothing. An honest result, not a bug.`
        : `A ${S.defender.name} can barely touch a ${S.attacker.name}: even the best drop saves almost nothing. An honest result, not a bug.`)
      :'Tower HP lost with no defence, and the share saved by a random drop and by the best drop, found by trying every cell and every delay on the five attacks.';
    $('matchupSummaryText').textContent = `${S.attacker.name} vs your ${S.defender.name}`;
    setTrayCard(S.defender);
    $('challengeText').textContent = `Five attacks. Drag your ${S.defender.name} onto the board to drop it, where and when you like. Each round scores the share of tower damage you prevented.`;
    renderRounds();
    updateChartRefs();
    setStage(S.stage === 'intro' ? 'intro' : 'challenge');
  }

  function setTrayCard(card) {
    $('traySymbol').textContent = card.symbol;
    $('traySymbol').classList.toggle('bldg', !!card.isBuilding);
    $('trayName').textContent = card.name;
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
    $('boardBar').hidden = shown !== 'train';
    if (!S.demoPlay && !S.counting) banner(null);
    if (shown === 'challenge') {
      const done = roundsDone();
      $('btnRound').textContent = done >= S.attempts.length ? 'Watch the AI learn' : `Start round ${done + 1}`;
      $('btnRound').disabled = false;
      $('btnSkip').hidden = done >= S.attempts.length;
      setTray(done >= S.attempts.length ? 'done' : 'idle');
    }
    if (shown !== 'results') { S.cmpPlay = null; }
    syncTrainControls();
    updateCta();
    S.chartDirty = true;
  }

  const roundsDone = () => S.attempts.filter(Boolean).length;

  // The big card in the middle of the board: whatever to press next.
  function updateCta() {
    const n = S.attempts.length, done = roundsDone();
    let c = null;
    if (S.stage === 'challenge' && !S.roundLive && !S.counting && !S.demoPlay && S.matchup) {
      if (done === 0) {
        c = { kicker: `Your turn · round 1 of ${n}`, title: `Stop the ${S.attacker.name}`,
              sub: `Drag your ${S.defender.name} from the tray below onto your half of the board. Where and when you drop it both count.`,
              btn: 'Start round 1', onBtn: () => startRound(0),
              alt: S.demo ? 'How to play' : null, onAlt: startDemo };
      } else if (done < n) {
        const a = S.attempts[done - 1];
        c = { kicker: `Round ${done} of ${n}`, title: a && a.cell >= 0 ? `Saved ${pct(a.score)}` : 'No drop',
              sub: a ? lostText(a) : '', btn: `Next round`, onBtn: () => startRound(done),
              alt: 'Skip to the AI', onAlt: goTrain };
      } else {
        c = { kicker: `All ${n} rounds played`, title: `You saved ${pct(S.youScore)}`,
              sub: 'Now watch a network that has never seen the game learn the same five attacks.',
              btn: 'Watch the AI learn', onBtn: goTrain, alt: 'Play again', onAlt: playAgain };
      }
    } else if (S.stage === 'train' && S.tries === 0 && !S.training && !S.playback) {
      c = { kicker: "The AI's turn", title: 'It starts knowing nothing',
            sub: `Each try, the engine drops the ${S.attacker.name} somewhere new, the network picks a cell and a moment for your ${S.defender.name}, and the engine scores it.`,
            btn: 'Train', onBtn: () => setTraining(true) };
    }
    const el = $('boardCta');
    S.cta = c;
    if (!c) { el.hidden = true; return; }
    $('ctaKicker').textContent = c.kicker;
    $('ctaTitle').textContent = c.title;
    $('ctaSub').textContent = c.sub || '';
    $('ctaBtn').textContent = c.btn;
    $('ctaAlt').hidden = !c.alt;
    $('ctaAlt').textContent = c.alt || '';
    if (el.hidden) { el.hidden = false; el.style.animation = 'none'; void el.offsetWidth; el.style.animation = ''; }
  }
  $('ctaBtn').addEventListener('click', () => { if (S.cta && S.cta.onBtn) S.cta.onBtn(); });
  $('ctaAlt').addEventListener('click', () => { if (S.cta && S.cta.onAlt) S.cta.onAlt(); });

  function lostText(a) {
    return a.damage > 0 ? `${fmtHp(a.damage)} of ${fmtHp(a.d0)} tower HP lost` : 'Tower untouched';
  }

  // ---- the how-to-play demo --------------------------------------------------

  function startDemo() {
    if (!S.demo || !S.arena) return;
    stopPlayback();
    S.demoSeen = true;
    const card = id => (S.roster.attackers.concat(S.roster.defenders).find(c => c.id === id)) ||
                       { id, name: id === 15 ? 'Hog Rider' : 'Cannon', symbol: '?', isBuilding: id === 25 };
    S.demoPlay = { start: performance.now(), attacker: card(S.demo.attacker), defender: card(S.demo.defender), phase: '' };
    setTrayCard(S.demoPlay.defender);
    setTray('idle');
    $('trayHint').textContent = 'Watch the demo';
    $('demoSkip').hidden = false;
    $('btnRound').disabled = false;
    updateCta();
    showBoard();
  }

  function stopDemo() {
    if (!S.demoPlay) return;
    S.demoPlay = null;
    S.ghost = null;
    $('demoHand').hidden = true;
    $('demoSkip').hidden = true;
    $('countdown').hidden = true;
    avatar.hidden = true;
    tray.classList.remove('dragging', 'pressed');
    if (S.defender) setTrayCard(S.defender);
    if (S.stage === 'challenge') setStage('challenge');
  }
  $('demoSkip').addEventListener('click', stopDemo);

  // One demo frame: the countdown, then the recorded rollout, with a hand that
  // picks the card out of the tray and drops it on the tick the engine did.
  function demoFrame(now) {
    const D = S.demo, P = S.demoPlay;
    const t = now - P.start, count = 3 * DEMO_COUNT_MS;
    const phase = (name, text) => { if (P.phase !== name) { P.phase = name; banner(text); } };
    const cd = $('countdown');
    if (t < count) {
      const n = 3 - Math.floor(t / DEMO_COUNT_MS);
      if (cd.dataset.n !== String(n)) { cd.hidden = false; cd.dataset.n = n; cd.innerHTML = `<span>${n}</span>`; }
      phase('count', 'How to play: watch one round');
      return { frameA: D.frames[0] };
    }
    cd.hidden = true; cd.dataset.n = '';
    const pos = (t - count) / DEMO_TICK_MS;
    const i = Math.min(D.frames.length - 1, Math.floor(pos));
    const st = { frameA: D.frames[i], frameB: D.frames[Math.min(D.frames.length - 1, i + 1)], frac: Math.min(1, pos - i) };
    const handStart = D.drop.tick - DEMO_HAND_TICKS;
    const hand = $('demoHand');
    if (pos < handStart) {
      phase('attack', `The enemy drops a ${P.attacker.name}: it charges your tower`);
    } else if (pos < D.drop.tick) {
      phase('drag', `Drag your ${P.defender.name} from the tray onto your half`);
      // Press on the card, then carry it along a gentle arc to the cell.
      const u = (pos - handStart) / DEMO_HAND_TICKS;
      const press = Math.min(1, u / 0.18), carry = ease(Math.max(0, (u - 0.18) / 0.72));
      const r = tray.getBoundingClientRect();
      const from = { x: r.left + 30, y: r.top + r.height / 2 };
      const to = cellClient(D.drop.x, D.drop.y);
      const mid = { x: (from.x + to.x) / 2, y: Math.min(from.y, to.y) - 60 };
      const x = quad(from.x, mid.x, to.x, carry), y = quad(from.y, mid.y, to.y, carry);
      hand.hidden = false;
      hand.classList.toggle('down', press >= 1);
      hand.style.transform = `translate(${x}px, ${y}px)`;
      tray.classList.toggle('pressed', press >= 1 && carry === 0);
      tray.classList.toggle('dragging', carry > 0);
      const g = board.toGame(x, y);
      if (carry > 0 && g.inside) {
        avatar.hidden = true;
        S.ghost = { x: Math.round(g.x), y: Math.round(g.y), legal: true, symbol: P.defender.symbol, building: P.defender.isBuilding };
        if (carry >= 1) Object.assign(S.ghost, { x: D.drop.x, y: D.drop.y });
      } else if (carry > 0) {
        S.ghost = null;
        avatar.textContent = P.defender.symbol;
        avatar.classList.toggle('bldg', !!P.defender.isBuilding);
        avatar.classList.remove('over-board');
        avatar.style.transition = 'none';
        avatar.style.transform = `translate(${x}px, ${y}px)`;
        avatar.hidden = false;
      }
    } else {
      S.ghost = null;
      avatar.hidden = true;
      tray.classList.remove('dragging', 'pressed');
      hand.classList.remove('down');
      hand.hidden = pos > D.drop.tick + 6;
      if (pos < D.drop.tick + 12) phase('drop', 'Let go to drop it. The timing counts too');
      else phase('pull', `The ${P.defender.name} pulls the ${P.attacker.name} away from your tower`);
      setTray('used');
      $('trayHint').textContent = 'Dropped';
    }
    if (pos >= D.frames.length - 1 + 12) { stopDemo(); return null; }
    return st;
  }

  function cellClient(x, y) {
    const p = board.toCanvas(x, y), r = $('board').getBoundingClientRect();
    return { x: r.left + p.x, y: r.top + p.y };
  }
  const ease = u => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);
  const quad = (a, b, c, u) => (1 - u) * (1 - u) * a + 2 * (1 - u) * u * b + u * u * c;

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
    const done = roundsDone();
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

  function playAgain() {
    S.attempts = S.attempts.map(() => null);
    S.youScore = null;
    S.live = { prev: null, cur: null, at: 0 };
    updateChartRefs();
    renderRounds();
    setStage('challenge');
  }

  function startRound(round) {
    stopPlayback();
    stopDemo();
    S.round = round; S.placed = false; S.roundLive = false; S.counting = true;
    S.live = { prev: null, cur: null, at: 0 };
    $('btnRound').disabled = true;
    $('btnSkip').hidden = true;
    $('boardCta').hidden = true;
    S.cta = null;
    setTray('idle');
    banner(null);
    showBoard();
    countdown(() => {
      S.roundLive = true;
      S.counting = false;
      renderRounds();
      setTray('ready');
      banner(round === 0
        ? `Drag your ${S.defender.name} from below onto your half`
        : `Round ${round + 1} of ${S.attempts.length} · stop the ${S.attacker.name}`);
      worker.postMessage({ type: 'liveStart', round });
    });
  }

  // A new matchup or stage cancels a countdown still running.
  function cancelCountdown() {
    S.countToken = (S.countToken || 0) + 1;
    S.counting = false;
    $('countdown').hidden = true;
  }

  function countdown(done) {
    const el = $('countdown');
    const token = S.countToken = (S.countToken || 0) + 1;
    let n = 3;
    el.hidden = false;
    const next = () => {
      if (token !== S.countToken) return;
      if (n === 0) { el.hidden = true; el.dataset.n = ''; done(); return; }
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
    avatar.hidden = true;
    tray.classList.remove('dragging');
    const score = Math.max(-1, Math.min(1, (m.d0 - m.damage) / m.d0));
    S.attempts[m.round] = { cell: m.placedTick >= 0 ? m.cell : -1, tick: Math.max(0, m.placedTick), damage: m.damage, d0: m.d0, score };
    renderRounds();
    const a = S.attempts[m.round];
    banner(m.placedTick < 0 ? `No drop: ${lostText(a)}` : `Saved ${pct(score)} · ${lostText(a)}`);
    if (roundsDone() >= S.attempts.length) {
      S.youScore = S.attempts.reduce((s, x) => s + x.score, 0) / S.attempts.length;
      updateChartRefs();
      $('challengeText').innerHTML = `Your average: <b>${pct(S.youScore)}</b> of the damage prevented. Now let a network that has never seen the game try the same five attacks.`;
    }
    setTimeout(() => { if (S.stage === 'challenge' && !S.roundLive && !S.counting) setStage('challenge'); }, 900);
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
    stopDemo();
    cancelCountdown();
    worker.postMessage({ type: 'liveStop' });
    S.roundLive = false;
    setStage('train');
    showBoard();
  }

  function setTraining(on) {
    if (on === S.training) return;
    S.training = on;
    if (on) { stopPlayback(); banner(null); showBoard(); }
    worker.postMessage({ type: 'train', on });
    syncTrainControls();
    updateCta();
  }

  // The side panel and the bar under the board carry the same three controls.
  function syncTrainControls() {
    const label = S.training ? 'Pause' : S.tries ? 'Resume' : 'Train';
    $('btnTrain').textContent = label;
    $('barTrain').textContent = label;
    $('barTrain').classList.toggle('primary', !S.training);
    const noResults = S.tries < 500;
    $('btnResults').disabled = noResults;
    $('barResults').disabled = noResults;
    $('barResults').classList.toggle('primary', !noResults && S.training);
    $('btnShow').disabled = S.tries === 0;
    $('barShow').disabled = S.tries === 0;
  }

  $('btnTrain').addEventListener('click', () => setTraining(!S.training));
  $('barTrain').addEventListener('click', () => setTraining(!S.training));
  $('btnReset').addEventListener('click', () => {
    S.training = false; S.curve = []; S.tries = 0; S.beatToastShown = false;
    worker.postMessage({ type: 'reset' });
    syncTrainControls();
    updateCta();
  });
  document.querySelectorAll('.seg-btn').forEach(b => b.addEventListener('click', () => {
    document.querySelectorAll('.seg-btn').forEach(x => x.classList.toggle('active', x === b));
    worker.postMessage({ type: 'speed', speed: b.dataset.speed });
  }));
  const showMe = () => { showBoard(); worker.postMessage({ type: 'showme' }); };
  $('btnShow').addEventListener('click', showMe);
  $('barShow').addEventListener('click', showMe);
  const seeResults = () => {
    setTraining(false);
    worker.postMessage({ type: 'compare', attempts: S.attempts });
  };
  $('btnResults').addEventListener('click', seeResults);
  $('barResults').addEventListener('click', seeResults);

  function onProgress(m) {
    S.heat = m.heat; S.greedy = m.greedy; S.preview = m.spawn; S.tries = m.tries; S.curve = m.curve;
    $('statTries').textContent = m.tries.toLocaleString();
    $('statRate').textContent = Math.round(m.rate).toLocaleString();
    $('statTime').textContent = `${(m.elapsed / 1000).toFixed(1)}s`;
    const cell = S.matchup.cells[m.greedy.cell];
    const delay = (m.greedy.delay * S.matchup.delayTickStep / 10).toFixed(1);
    $('guess').innerHTML = m.tries === 0
      ? `Drag the red ${esc(S.attacker.name)} on the board: the heatmap shows where the network would drop your ${esc(S.defender.name)}. Right now it has no idea.`
      : `For this attack its best guess is column ${cell[0]}, row ${cell[1]}, <b>${delay} s</b> after the ${esc(S.attacker.name)} is dropped.`;
    const last = m.curve[m.curve.length - 1];
    if (last && S.youScore != null && !S.beatToastShown && last.v > S.youScore + 1e-6) {
      S.beatToastShown = true;
      toast(`The network just beat your score, after ${last.tries.toLocaleString()} tries.`);
    }
    if (!m.training && S.training) S.training = false;
    syncTrainControls();
    if (S.stage === 'train') updateCta();
    S.chartDirty = true;
  }

  // Dragging the attacker moves the preview attack; the heatmap follows.
  const canvas = $('board');
  canvas.addEventListener('pointerdown', e => {
    if (S.stage !== 'train' || !S.matchup || S.playback) return;
    const g = board.toGame(e.clientX, e.clientY);
    if (!g.inside || nearestSpawn(g).d > PREVIEW_REACH) return;
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

  function nearestSpawn(g) {
    let best = S.preview, bd = Infinity;
    S.matchup.spawns.forEach((s, i) => {
      const d = Math.hypot(s[0] - g.x, s[1] - g.y);
      if (d < bd) { bd = d; best = i; }
    });
    return { i: best, d: bd };
  }

  function movePreview(g) {
    const best = nearestSpawn(g).i;
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
    updateCta();
  }

  function stopPlayback() { S.playback = null; }

  // ---- results ---------------------------------------------------------------

  function onCompare(m) {
    S.compare = m;
    m.rounds.forEach(r => { r.heatClock = heatClock(r.heat || []); });
    S.verdictShown = false;
    const verdict = $('verdict');
    verdict.classList.remove('shown');
    verdict.innerHTML = '&nbsp;';
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
    $('btnAgain').textContent = m.rounds.every(r => r.you) ? 'Try another matchup' : 'Take the challenge';
    setStage('results');
    playCompare(0);
    showBoard();
  }

  function showVerdict() {
    if (S.verdictShown || !S.compare) return;
    S.verdictShown = true;
    const m = S.compare;
    const ai = m.rounds.reduce((s, r) => s + (r.d0 - r.ai.damage) / r.d0, 0) / m.rounds.length;
    const played = m.rounds.every(r => r.you);
    const secs = Math.max(1, Math.round(m.elapsed / 1000));
    const el = $('verdict');
    el.innerHTML = played
      ? `You saved <span class="you">${pct(S.youScore)}</span>. The learner saved <span class="ai">${pct(ai)}</span> after ${secs} s of training.`
      : `The learner saved <span class="ai">${pct(ai)}</span> after ${secs} s and ${m.tries.toLocaleString()} tries. Take the challenge to see how you compare.`;
    el.classList.add('shown');
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
    S.cmpPlay = { start: performance.now(), r, you: r.you && r.you.frames, ai: r.ai.frames,
                  dropYou: drop(r.you), dropAi: drop(r.ai), scored: false };
    $('cmpYouScore').textContent = r.you ? '' : 'not played';
    $('cmpAiScore').textContent = '';
  }

  // The heatmap's playback clock: each step between snapshots gets screen
  // time in proportion to how much the map changes (its entropy), plus a
  // little for every step, so the animation lingers where the network is
  // actually making up its mind instead of on the early tries where little
  // moves. Returns the cumulative share of the animation at each snapshot.
  function heatClock(heat) {
    const H = heat.map(h => {
      let e = 0;
      for (const v of h.p) if (v > 0) e -= v * Math.log(v);
      return e / Math.log(Math.max(2, h.p.length));
    });
    const w = [0];
    for (let k = 1; k < heat.length; k++) w.push(w[k - 1] + Math.abs(H[k] - H[k - 1]) + 0.02);
    const total = w[w.length - 1] || 1;
    return w.map(v => v / total);
  }

  // Where the results timeline is for round r at time t: the heatmap
  // converging, a beat on the answer, then the replay.
  function cmpState(p, now) {
    const t = now - p.start;
    const heat = p.r.heat || [], clock = p.r.heatClock || [];
    const replayAt = CMP_HEAT_MS + CMP_ANSWER_MS;
    if (t < CMP_HEAT_MS && heat.length > 1) {
      const x = Math.min(1, t / CMP_HEAT_MS);
      let k = 0;
      while (k < heat.length - 2 && clock[k + 1] < x) k++;
      const f = Math.min(1, Math.max(0, (x - clock[k]) / Math.max(1e-9, clock[k + 1] - clock[k])));
      const a = heat[k].p, b = heat[k + 1].p, h = new Float32Array(a.length);
      for (let c = 0; c < a.length; c++) h[c] = a[c] + (b[c] - a[c]) * f;
      const tries = Math.round(heat[k].tries + (heat[k + 1].tries - heat[k].tries) * f);
      return { phase: 'heat', heat: h, tries };
    }
    if (t < replayAt) return { phase: 'answer', heat: heat.length ? heat[heat.length - 1].p : null };
    const pos = (t - replayAt) / CMP_TICK_MS;
    return { phase: 'replay', pos };
  }

  function replayFrame(frames, pos) {
    const i = Math.min(frames.length - 1, Math.max(0, Math.floor(pos)));
    return { frameA: frames[i], frameB: frames[Math.min(frames.length - 1, i + 1)], frac: Math.min(1, pos - i), clockTick: frames[i].t };
  }

  function renderResults(now) {
    const p = S.cmpPlay, r = p.r;
    const cs = cmpState(p, now);
    const spawn = S.matchup.spawns[r.spawn];
    const aiCell = r.ai.drop.cell;
    const round = `Attack ${S.cmpIdx + 1} of ${S.compare.rounds.length}`;
    if (cs.phase === 'replay') {
      if (!p.scored) {
        p.scored = true;
        showVerdict();
        $('cmpYouScore').textContent = r.you ? `saved ${pct((r.d0 - r.you.damage) / r.d0)}` : 'not played';
        $('cmpAiScore').textContent = `saved ${pct((r.d0 - r.ai.damage) / r.d0)}`;
      }
      const fa = replayFrame(p.ai, cs.pos);
      const main = Object.assign({ drop: p.dropAi, caption: `${round} · its answer, played by the engine` }, fa);
      board.render(main);
      miniAi.render(Object.assign({ drop: p.dropAi }, fa));
      if (p.you) miniYou.render(Object.assign({ drop: p.dropYou }, replayFrame(p.you, cs.pos)));
      else miniYou.render({});
      const len = Math.max(p.ai.length, p.you ? p.you.length : 0);
      if (cs.pos > len + CMP_HOLD_MS / CMP_TICK_MS) playCompare((S.cmpIdx + 1) % S.compare.rounds.length);
      return;
    }
    const answer = cs.phase === 'answer';
    const heatState = {
      heat: cs.heat, cells: S.matchup.cells, spawns: [spawn], preview: 0, previewSymbol: S.attacker.symbol, previewLabel: '',
      greedy: answer ? { cell: aiCell, label: `${(r.ai.drop.tick / 10).toFixed(1)}s` } : null,
    };
    board.render(Object.assign({}, heatState, {
      caption: answer ? `${round} · its answer` : `${round} · learning where to drop · ${cs.tries.toLocaleString()} tries`,
    }));
    miniAi.render(Object.assign({}, heatState, { caption: answer ? 'its answer' : `${cs.tries.toLocaleString()} tries` }));
    if (p.you) miniYou.render({ frameA: p.you[0], drop: p.dropYou, clockTick: 0 });
    else miniYou.render({});
  }

  $('btnBackTrain').addEventListener('click', () => setStage('train'));
  $('btnAgain').addEventListener('click', () => {
    if (S.compare && !S.compare.rounds.every(r => r.you)) { playAgain(); return; }
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
      if (S.stage === 'results' && S.cmpPlay) {
        renderResults(now);
      } else {
        const st = {};
        if (S.demoPlay) {
          Object.assign(st, demoFrame(now) || {});
          st.ghost = S.ghost;
          if (S.demoPlay && S.demoPlay.phase === 'drag') st.forbid = demoLegal();
        } else if (S.playback) {
          const f = frameAt(S.playback.frames, S.playback.start, now);
          Object.assign(st, { frameA: f.a, frameB: f.b, frac: f.frac, drop: S.playback.drop, clockTick: f.a.t });
          if (f.done && now - S.playback.start > S.playback.frames.length * TICK_MS + 1200) {
            const end = S.playback.onEnd;
            stopPlayback();
            if (end) end();
            updateCta();
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
      }
    }
    if (S.chartDirty) {
      chart.setPoints(S.curve.map(p => ({ tries: p.tries, v: p.v })));
      chart.render();
      S.chartDirty = false;
    }
    requestAnimationFrame(render);
  }

  // Where the demo's Cannon may go, from the engine (worker.js recordDemo).
  function demoLegal() {
    if (!S.demoLegalSet) S.demoLegalSet = new Set(S.demo.cells.map(c => c[0] + ',' + c[1]));
    return S.demoLegalSet;
  }

  // ---- small helpers --------------------------------------------------------------

  function pct(v) { return v == null || Number.isNaN(v) ? '-' : `${Math.round(v * 100)}%`; }
  function fmtHp(v) { return Math.round(v).toLocaleString(); }
  function esc(s) { return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

  function banner(text) {
    const el = $('banner');
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
    if (!S.demoSeen) startDemo();
  });
  $('btnAbout').addEventListener('click', () => { $('about').hidden = false; });
  $('btnAboutClose').addEventListener('click', () => { $('about').hidden = true; });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !$('about').hidden) $('about').hidden = true;
  });
  window.addEventListener('resize', () => { S.chartDirty = true; });

  // Test hooks for tools/lab/check_lab_ui.py.
  window.__lab = { state: S, board, chart, stopDemo, cellCenterClient(x, y) {
    const p = board.toCanvas(x, y), r = $('board').getBoundingClientRect();
    return { x: r.left + p.x, y: r.top + p.y };
  } };

  document.fonts.ready.then(() => { board.bg = null; S.chartDirty = true; });
  worker.postMessage({ type: 'init' });
  requestAnimationFrame(render);
})();
