// Local development server for the Reflex Lab, for a machine without
// Emscripten: serves web/ and answers POST /engine with the NATIVE engine
// (tools/lab/out/lab_cli.exe serve), one command line in, one line out.
//
//   node tools/lab/dev_server.mjs [port]      then open http://127.0.0.1:8766/lab/
//
// The page prefers the WebAssembly engine when web/lab/engine/engine.wasm
// exists and falls back to this backend only when it does not. Both run the
// same LabEngine (web/lab/engine/lab_engine.h). Development only: it binds
// to 127.0.0.1 and serves nothing outside web/.
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.resolve(HERE, '..', '..', 'web');
const CLI = path.join(HERE, 'out', 'lab_cli.exe');
const PORT = Number(process.argv[2] || 8766);

const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8', '.wasm': 'application/wasm',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.ico': 'image/x-icon', '.woff2': 'font/woff2',
};

// One engine process; requests are answered strictly in order.
const engine = spawn(CLI, ['serve'], { stdio: ['pipe', 'pipe', 'inherit'] });
const lines = readline.createInterface({ input: engine.stdout });
const waiting = [];
lines.on('line', line => { const w = waiting.shift(); if (w) w(line); });
engine.on('exit', code => { console.error(`lab_cli exited (${code})`); process.exit(1); });
function ask(command) {
  return new Promise(resolve => {
    waiting.push(resolve);
    engine.stdin.write(command.replace(/[\r\n]+/g, ' ') + '\n');
  });
}

http.createServer((req, res) => {
  if (req.method === 'POST' && req.url === '/engine') {
    let body = '';
    req.on('data', d => { body += d; if (body.length > 1e6) req.destroy(); });
    req.on('end', async () => {
      const answer = await ask(body.trim());
      res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8', 'Cache-Control': 'no-store' });
      res.end(answer);
    });
    return;
  }
  const url = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  let file = path.normalize(path.join(WEB, url));
  if (!file.startsWith(WEB)) { res.writeHead(403); res.end(); return; }
  if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, 'index.html');
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); res.end('not found'); return; }
    res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream',
                         'Cache-Control': 'no-store' });
    res.end(data);
  });
}).listen(PORT, '127.0.0.1', () => console.log(`Reflex Lab dev server: http://127.0.0.1:${PORT}/lab/`));
