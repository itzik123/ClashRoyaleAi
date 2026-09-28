// One async API over the two ways the lab can reach the engine:
//
//   wasm  web/lab/engine/engine.js + engine.wasm, the real C++ engine compiled
//         to WebAssembly, running in this worker (the published page)
//   dev   lab_cli serve behind tools/lab/dev_server.mjs (a local machine
//         without Emscripten)
//
// Both run LabEngine (web/lab/engine/lab_engine.h), so every answer is the
// same; tools/lab/parity.mjs checks it. Loaded into the worker with
// importScripts.
(function (root) {
  'use strict';

  class WasmEngine {
    constructor(module) {
      this.kind = 'wasm';
      this.e = new module.LabEngine(1);
    }
    async cards(ids) { return JSON.parse(this.e.cardsJson(ids)); }
    async arena() { return JSON.parse(this.e.arenaJson()); }
    async matchup(a, d) { return JSON.parse(this.e.setMatchup(a, d)); }
    async batch(triples) { return this.e.rolloutBatch(triples); }
    async frames(spawn, cell, dropTick) { return JSON.parse(this.e.framesJson(spawn, cell, dropTick)); }
    async liveStart(spawn) { this.e.liveStart(spawn); }
    async liveStep(n) { return JSON.parse(this.e.liveStepJson(n)); }
    async livePlace(cell) { return this.e.livePlace(cell); }
    async liveState() { return JSON.parse(this.e.liveStateJson()); }
  }

  class DevEngine {
    constructor() { this.kind = 'dev'; }
    async ask(line) {
      const r = await fetch('/engine', { method: 'POST', body: line });
      const text = await r.text();
      if (text.startsWith('error')) throw new Error(text);
      return text;
    }
    async cards(ids) { return JSON.parse(await this.ask(`cards ${ids.join(' ')}`)); }
    async arena() { return JSON.parse(await this.ask('arena')); }
    async matchup(a, d) { return JSON.parse(await this.ask(`matchup ${a} ${d}`)); }
    async batch(triples) {
      const n = triples.length / 3;
      const text = await this.ask(`batch ${n} ${Array.prototype.join.call(triples, ' ')}`);
      return Float32Array.from(text.trim().split(/\s+/), Number);
    }
    async frames(spawn, cell, dropTick) { return JSON.parse(await this.ask(`frames ${spawn} ${cell} ${dropTick}`)); }
    async liveStart(spawn) { await this.ask(`live_start ${spawn}`); }
    async liveStep(n) { return JSON.parse(await this.ask(`live_step ${n}`)); }
    async livePlace(cell) { return (await this.ask(`live_place ${cell}`)) === 'true'; }
    async liveState() { return JSON.parse(await this.ask('live_state')); }
  }

  // The WASM build if it is there, else the dev backend if it answers.
  async function connect(base) {
    try {
      const probe = await fetch(base + 'engine/engine.wasm', { method: 'HEAD' });
      if (probe.ok) {
        root.importScripts(base + 'engine/engine.js');
        const module = await root.createLabEngine({ locateFile: f => base + 'engine/' + f });
        return new WasmEngine(module);
      }
    } catch (e) { /* fall through to the dev backend */ }
    const dev = new DevEngine();
    try {
      await dev.ask('live_state');
      return dev;
    } catch (e) {
      if (String(e.message).startsWith('error')) return dev;   // it answered: it is there
    }
    throw new Error('The engine is not built: run web/lab/engine/build.ps1, ' +
                    'or tools/lab/dev_server.mjs for local development.');
  }

  root.EngineClient = { connect, WasmEngine, DevEngine };
})(self);
