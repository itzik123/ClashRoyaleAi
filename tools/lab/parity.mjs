// The WebAssembly engine must answer exactly as the native one.
//
//   node tools/lab/parity.mjs [--cli path/to/lab_cli]
//
// Loads web/lab/engine/engine.js (built by web/lab/engine/build.ps1) and the
// native lab_cli (tools/lab/build_cli.ps1), sets up every roster matchup in
// both, and compares the matchup JSON, the arena JSON, and 200 seeded tries
// of tower damage and defender survival per matchup. The engine is deterministic and its gameplay
// path uses no libm transcendental and iterates no hash container, so the
// comparison is EQUALITY, not a tolerance. Exits non-zero on any difference.
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');
const require = createRequire(import.meta.url);
const cliArg = process.argv.indexOf('--cli');
const CLI = cliArg > 0 ? process.argv[cliArg + 1]
  : path.join(HERE, 'out', process.platform === 'win32' ? 'lab_cli.exe' : 'lab_cli');
const ENGINE = path.join(REPO, 'web', 'lab', 'engine', 'engine.js');
const TRIES = 200;

if (!fs.existsSync(ENGINE)) {
  console.error(`no WASM build at ${ENGINE}: run web/lab/engine/build.ps1 first`);
  process.exit(2);
}
const createLabEngine = require(ENGINE);
const mod = await createLabEngine();
const wasm = new mod.LabEngine(1);
const roster = JSON.parse(fs.readFileSync(path.join(REPO, 'web', 'lab', 'roster.json'), 'utf8'));

let fails = 0, compared = 0;
const native = cmds => execFileSync(CLI, ['serve'], { input: cmds.join('\n') + '\n', maxBuffer: 1 << 28 })
  .toString().trim().split('\n');

const arenaW = wasm.arenaJson(), arenaN = native(['arena'])[0];
if (arenaW !== arenaN) { fails++; console.log('  FAIL  arena JSON differs'); } else console.log('  ok    arena JSON identical');

let seed = 12345;
const rand = n => { seed = (seed * 1103515245 + 12345) >>> 0; return seed % n; };
for (const [key, m] of Object.entries(roster.matchups)) {
  const mw = wasm.setMatchup(m.attacker, m.defender);
  const parsed = JSON.parse(mw);
  const triples = [];
  for (let i = 0; i < TRIES; i++) {
    triples.push(rand(parsed.spawns.length), rand(parsed.cells.length), rand(parsed.delaySteps));
  }
  // Damage and survival, interleaved, in both.
  const dw = Array.from(wasm.rolloutBatch(Int32Array.from(triples)));
  const [mn, bn] = native([`matchup ${m.attacker} ${m.defender}`, `batch ${TRIES} ${triples.join(' ')}`]);
  const dn = bn.split(' ').map(Number);
  let diff = mw === mn ? 0 : 1;
  if (dw.length !== 2 * TRIES || dn.length !== 2 * TRIES) diff++;
  // Native prints 9 significant digits, which round-trip a float32 exactly.
  for (let i = 0; i < 2 * TRIES; i++) if (dw[i] !== Math.fround(dn[i])) diff++;
  compared += TRIES;
  if (diff) { fails++; console.log(`  FAIL  ${key}: ${diff} differences${mw === mn ? '' : ' (matchup JSON differs)'}`); }
  else console.log(`  ok    ${key}: matchup JSON and ${TRIES}/${TRIES} tries identical`);
}
console.log(fails ? `\n${fails} matchup(s) differ` : `\nall ${compared} tries identical across WASM and native`);
process.exit(fails ? 1 : 0);
