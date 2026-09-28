// The Reflex Lab's arena renderer: the replay viewer's look (web/viewer.html),
// drawn from the ENGINE's geometry -- river band, bridge columns and towers all
// come from LabEngine::arenaJson, so nothing here restates an engine constant.
//
// Frames are LabEngine frame JSON: {t, towers, e: [[id, cardId, team, x, y, hp,
// maxHp, flags, radius, symbol, name], ...]}.
(function (root) {
  'use strict';

  const F = { FLYING: 1, BUILDING: 2, TOWER: 4, DEPLOYING: 8, PROJECTILE: 16, SPELL: 32 };
  const BLUE = '#4a9eff', RED = '#ff5252', BLUE_DIM = '#1a3a66', RED_DIM = '#661a1a';
  const FONT = "'Unbounded', system-ui, sans-serif";

  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function lerp(a, b, t) { return a + (b - a) * t; }

  // Heat colour: the viewer's blue at the cold end, its yellow at the hot end.
  function heatColor(v) {
    const a = [74, 158, 255], b = [250, 204, 21];
    const t = Math.min(1, Math.max(0, v));
    return [Math.round(lerp(a[0], b[0], t)), Math.round(lerp(a[1], b[1], t)), Math.round(lerp(a[2], b[2], t))];
  }

  class LabBoard {
    // opts.rows: [yMin, yMax] game rows to show (default: the whole board).
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.opts = opts;
      this.arena = null;
      this.bg = null;
      this.state = {};
    }

    setArena(arena) {
      this.arena = arena;
      this.rows = this.opts.rows || [0, arena.height - 1];
      this.bridgeGroups = [];
      for (const x of arena.bridgeColumns) {
        const g = this.bridgeGroups[this.bridgeGroups.length - 1];
        if (g && g[1] === x - 1) g[1] = x; else this.bridgeGroups.push([x, x]);
      }
      this.bg = null;
    }

    // ---- geometry ---------------------------------------------------------

    layout() {
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
      if (!w || !h || !this.arena) return false;
      if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
        this.canvas.width = Math.round(w * dpr);
        this.canvas.height = Math.round(h * dpr);
        this.bg = null;
      }
      this.dpr = dpr;
      this.w = w; this.h = h;
      const cols = this.arena.width, rows = this.rows[1] - this.rows[0] + 1;
      const cs = Math.min(w / (cols + 1.2), h / (rows + 1.2));
      this.cs = cs;
      this.pad = cs * 0.6;
      this.ox = (w - (cols * cs + 2 * this.pad)) / 2;
      this.oy = (h - (rows * cs + 2 * this.pad)) / 2;
      return true;
    }

    // Game coordinates (cell centres are integers) -> canvas pixels.
    toCanvas(gx, gy) {
      return {
        x: this.ox + this.pad + (gx + 0.5) * this.cs,
        y: this.oy + this.pad + (this.rows[1] + 0.5 - gy) * this.cs,
      };
    }

    toGame(clientX, clientY) {
      const r = this.canvas.getBoundingClientRect();
      const cx = clientX - r.left, cy = clientY - r.top;
      return {
        x: (cx - this.ox - this.pad) / this.cs - 0.5,
        y: this.rows[1] + 0.5 - (cy - this.oy - this.pad) / this.cs,
        inside: cx >= this.ox && cx <= this.w - this.ox && cy >= this.oy && cy <= this.h - this.oy,
      };
    }

    // Pixel rectangle of game rectangle [x0, x1] x [y0, y1] in cell units.
    rect(x0, y0, x1, y1) {
      // Cell i spans [i - 0.5, i + 0.5], so these are the outer edges.
      const a = this.toCanvas(x0 - 0.5, y1 + 0.5), b = this.toCanvas(x1 + 0.5, y0 - 0.5);
      return [a.x, a.y, b.x - a.x, b.y - a.y];
    }

    cellRect(x, y) {
      const p = this.toCanvas(x, y), s = this.cs;
      return [p.x - s / 2, p.y - s / 2, s, s];
    }

    // ---- background (lawn, lanes, river, bridges), cached per size ----------

    background() {
      if (this.bg) return this.bg;
      const cv = document.createElement('canvas');
      cv.width = this.canvas.width; cv.height = this.canvas.height;
      const c = cv.getContext('2d');
      c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      const A = this.arena, cs = this.cs, [r0, r1] = this.rows;
      const x0 = this.ox, y0 = this.oy;
      const bw = A.width * cs + 2 * this.pad, bh = (r1 - r0 + 1) * cs + 2 * this.pad;

      c.fillStyle = css('--arena-surround');
      roundRect(c, x0, y0, bw, bh, Math.max(6, cs * 0.6));
      c.fill();

      for (let y = r0; y <= r1; y++)
        for (let x = 0; x < A.width; x++) {
          c.fillStyle = (x + y) % 2 === 0 ? css('--grass-light') : css('--grass-dark');
          c.fillRect(...this.cellRect(x, y));
        }

      // Lane paths along each bridge's columns, then the river over them.
      c.fillStyle = css('--lane-path');
      for (const [a, b] of this.bridgeGroups) c.fillRect(...this.rect(a, r0, b, r1));
      // riverStart/riverEnd are edges in game y, not cell indices.
      const riverTop = this.oy + this.pad + (r1 + 0.5 - A.riverEnd) * cs;
      const riverBottom = this.oy + this.pad + (r1 + 0.5 - A.riverStart) * cs;
      if (riverBottom > this.oy + this.pad && riverTop < this.oy + this.pad + (r1 - r0 + 1) * cs) {
        const left = this.ox + this.pad, width = A.width * cs;
        c.fillStyle = css('--river');
        c.fillRect(left, riverTop, width, riverBottom - riverTop);
        c.fillStyle = css('--river-edge');
        c.fillRect(left, riverTop, width, Math.max(1, cs * 0.12));
        for (const [a, b] of this.bridgeGroups) {
          const bl = left + a * cs, br = left + (b + 1) * cs;
          c.fillStyle = css('--bridge-deck');
          c.fillRect(bl, riverTop, br - bl, riverBottom - riverTop);
          c.strokeStyle = css('--bridge-seam');
          c.lineWidth = Math.max(1, cs * 0.06);
          for (let k = 1; k < 4; k++) {
            const py = riverTop + (riverBottom - riverTop) * k / 4;
            c.beginPath(); c.moveTo(bl, py); c.lineTo(br, py); c.stroke();
          }
        }
      }

      // Faint tile grid for judging placement.
      c.strokeStyle = 'rgba(0,0,0,0.055)';
      c.lineWidth = 1;
      const left = this.ox + this.pad, topY = this.oy + this.pad;
      for (let x = 0; x <= A.width; x++) {
        c.beginPath(); c.moveTo(left + x * cs, topY); c.lineTo(left + x * cs, topY + (r1 - r0 + 1) * cs); c.stroke();
      }
      for (let y = 0; y <= r1 - r0 + 1; y++) {
        c.beginPath(); c.moveTo(left, topY + y * cs); c.lineTo(left + A.width * cs, topY + y * cs); c.stroke();
      }
      this.bg = cv;
      return cv;
    }

    // ---- per-frame drawing ------------------------------------------------

    // state: { frameA, frameB, frac, heat, cells, greedy, spawns, preview,
    //          previewSymbol, forbid, ghost, drop, clockTick }
    render(state) {
      if (!this.layout()) return;
      const c = this.ctx;
      c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      c.clearRect(0, 0, this.w, this.h);
      c.drawImage(this.background(), 0, 0, this.w, this.h);
      c.save();
      c.beginPath();
      c.rect(this.ox + this.pad, this.oy + this.pad, this.arena.width * this.cs,
             (this.rows[1] - this.rows[0] + 1) * this.cs);
      c.clip();
      if (state.heat) this.drawHeat(state.heat, state.cells, state.greedy);
      if (state.spawns) this.drawSpawnBand(state.spawns, state.preview);
      if (state.forbid) this.drawForbidden(state.forbid);
      if (state.drop) this.drawDrop(state.drop, state.clockTick);
      c.restore();

      const frame = state.frameA || (this.arena && this.arena.towers);
      if (frame) this.drawEntities(frame, state.frameB, state.frac || 0);
      if (state.spawns && state.preview != null && !state.frameA)
        this.drawPreviewAttacker(state.spawns[state.preview], state.previewSymbol, state.previewLabel);
      if (state.ghost) this.drawGhost(state.ghost);
      if (state.caption) this.drawCaption(state.caption);
    }

    // A small label chip across the top of the arena.
    drawCaption(text) {
      const c = this.ctx, fs = Math.max(8, Math.min(12, this.cs * 0.42));
      c.font = `600 ${fs}px ${FONT}`;
      const maxW = this.arena.width * this.cs - 8;
      let t = text;
      while (c.measureText(t).width + 18 > maxW && t.length > 4) t = t.slice(0, -2).trimEnd() + '…';
      const tw = c.measureText(t).width + 18, th = fs + 12;
      const x = this.w / 2 - tw / 2, y = this.oy + this.pad + Math.max(4, this.cs * 0.3);
      c.fillStyle = 'rgba(11,11,12,.82)';
      roundRect(c, x, y, tw, th, th / 2); c.fill();
      c.fillStyle = '#fff';
      c.textAlign = 'center'; c.textBaseline = 'middle';
      c.fillText(t, this.w / 2, y + th / 2 + 0.5);
    }

    drawHeat(heat, cells, greedy) {
      const c = this.ctx;
      let max = 0;
      for (let i = 0; i < heat.length; i++) if (heat[i] > max) max = heat[i];
      if (max <= 0) return;
      // Scaled to the peak, but never below six times uniform: an untrained
      // network's near-even spread reads as a faint wash, not a board on fire,
      // and it brightens as the probability gathers on a few cells.
      max = Math.max(max, 6 / heat.length);
      const inset = Math.max(0.5, this.cs * 0.06), rr = Math.max(1, this.cs * 0.18);
      for (let i = 0; i < cells.length; i++) {
        const v = heat[i] / max;
        if (v < 0.02) continue;
        const [r, g, b] = heatColor(Math.pow(v, 0.8));
        c.fillStyle = `rgba(${r},${g},${b},${(0.10 + 0.72 * Math.pow(v, 0.7)).toFixed(3)})`;
        const [x, y, w, h] = this.cellRect(cells[i][0], cells[i][1]);
        roundRect(c, x + inset, y + inset, w - 2 * inset, h - 2 * inset, rr);
        c.fill();
      }
      if (greedy) {
        const p = this.toCanvas(cells[greedy.cell][0], cells[greedy.cell][1]);
        c.strokeStyle = '#fff';
        c.lineWidth = 2;
        c.beginPath(); c.arc(p.x, p.y, this.cs * 0.75, 0, Math.PI * 2); c.stroke();
        if (greedy.label) {
          c.font = `700 ${Math.max(9, this.cs * 0.42)}px ${FONT}`;
          c.textAlign = 'center'; c.textBaseline = 'middle';
          const tw = c.measureText(greedy.label).width + 10, th = Math.max(14, this.cs * 0.7);
          const ly = p.y - this.cs * 1.35;
          c.fillStyle = 'rgba(11,11,12,.85)';
          roundRect(c, p.x - tw / 2, ly - th / 2, tw, th, th / 2); c.fill();
          c.fillStyle = '#fff';
          c.fillText(greedy.label, p.x, ly + 0.5);
        }
      }
    }

    drawSpawnBand(spawns, preview) {
      const c = this.ctx;
      c.fillStyle = 'rgba(255, 82, 82, 0.10)';
      for (const [x, y] of spawns) c.fillRect(...this.cellRect(x, y));
      c.fillStyle = 'rgba(255, 82, 82, 0.35)';
      for (const [x, y] of spawns) {
        const p = this.toCanvas(x, y);
        c.beginPath(); c.arc(p.x, p.y, Math.max(1.2, this.cs * 0.07), 0, Math.PI * 2); c.fill();
      }
    }

    drawPreviewAttacker(spawn, symbol, label = 'drag me') {
      if (!spawn) return;
      const c = this.ctx, p = this.toCanvas(spawn[0], spawn[1]);
      const r = this.cs * 0.62, t = performance.now() / 1000;
      c.strokeStyle = `rgba(255,82,82,${0.35 + 0.25 * Math.sin(t * 4)})`;
      c.lineWidth = 2;
      c.beginPath(); c.arc(p.x, p.y, r + 5 + 2 * Math.sin(t * 4), 0, Math.PI * 2); c.stroke();
      this.drawUnitBody(p, r, 1, symbol || '?', 1, false, 1);
      c.font = `600 ${Math.max(8, this.cs * 0.36)}px ${FONT}`;
      c.textAlign = 'center'; c.textBaseline = 'top';
      c.fillStyle = 'rgba(255,255,255,.9)';
      if (label) c.fillText(label, p.x, p.y + r + 6);
    }

    // The engine's forbidden region, tinted the way the real game tints it.
    drawForbidden(legal) {
      const c = this.ctx, A = this.arena;
      c.fillStyle = 'rgba(255, 60, 60, 0.20)';
      for (let y = this.rows[0]; y <= this.rows[1]; y++)
        for (let x = 0; x < A.width; x++)
          if (!legal.has(x + ',' + y)) c.fillRect(...this.cellRect(x, y));
    }

    // A pending drop in a replay: dashed footprint until the drop tick.
    drawDrop(drop, tick) {
      if (drop.tick == null || tick >= drop.tick) return;
      const c = this.ctx, p = this.toCanvas(drop.x, drop.y);
      const r = this.cs * (drop.building ? 1.0 : 0.6);
      c.save();
      c.setLineDash([4, 4]);
      c.strokeStyle = 'rgba(255,255,255,.85)';
      c.lineWidth = 1.5;
      if (drop.building) { roundRect(c, p.x - r, p.y - r, 2 * r, 2 * r, 4); c.stroke(); }
      else { c.beginPath(); c.arc(p.x, p.y, r, 0, Math.PI * 2); c.stroke(); }
      c.restore();
      const secs = Math.max(0, (drop.tick - tick) / 10).toFixed(1);
      c.font = `700 ${Math.max(9, this.cs * 0.4)}px ${FONT}`;
      c.textAlign = 'center'; c.textBaseline = 'middle';
      c.fillStyle = '#fff';
      c.fillText(`${secs}s`, p.x, p.y - r - this.cs * 0.45);
    }

    drawGhost(g) {
      const c = this.ctx;
      const p = this.toCanvas(g.x, g.y);
      const r = this.cs * (g.building ? 1.0 : 0.62);
      c.save();
      c.globalAlpha = 0.9;
      if (g.legal) {
        c.fillStyle = 'rgba(74,158,255,.30)';
        c.strokeStyle = BLUE;
      } else {
        c.fillStyle = 'rgba(255,82,82,.22)';
        c.strokeStyle = RED;
      }
      c.lineWidth = 2;
      if (g.building) { roundRect(c, p.x - r, p.y - r, 2 * r, 2 * r, 5); }
      else { c.beginPath(); c.arc(p.x, p.y, r, 0, Math.PI * 2); }
      c.fill(); c.stroke();
      if (g.legal) {
        c.globalAlpha = 0.95;
        this.drawUnitBody(p, this.cs * (g.building ? 0.8 : 0.5), 0, g.symbol, 1, g.building, 1);
      } else {
        c.strokeStyle = RED; c.lineWidth = 2.5;
        const k = r * 0.45;
        c.beginPath(); c.moveTo(p.x - k, p.y - k); c.lineTo(p.x + k, p.y + k);
        c.moveTo(p.x + k, p.y - k); c.lineTo(p.x - k, p.y + k); c.stroke();
      }
      c.restore();
    }

    // Draws the NEWER frame, each body eased in from where it was in the older
    // one: a body that just spawned appears at once and one that just died is
    // gone at once, instead of both lagging a tick.
    drawEntities(frameA, frameB, frac) {
      const cur = frameB || frameA;
      const prev = frameB && frameA && frameA !== frameB ? new Map(frameA.e.map(e => [e[0], e])) : null;
      const list = cur.e.slice().sort((a, b) => order(a) - order(b));
      for (const e of list) {
        let x = e[3], y = e[4];
        const p = prev && prev.get(e[0]);
        if (p) { x = lerp(p[3], x, frac); y = lerp(p[4], y, frac); }
        this.drawEntity(e, x, y);
      }
    }

    drawEntity(e, x, y) {
      const c = this.ctx, flags = e[7], team = e[2], p = this.toCanvas(x, y);
      if (flags & F.PROJECTILE) {
        c.beginPath(); c.arc(p.x, p.y, Math.max(2, this.cs * 0.13), 0, Math.PI * 2);
        c.fillStyle = team === 0 ? BLUE : RED; c.fill();
        return;
      }
      if (flags & F.SPELL) {
        // A spell in the air (the Goblin Barrel): where it will land, pulsing,
        // and the card itself over it.
        const t = performance.now() / 1000, r = this.cs * 0.9;
        c.save();
        c.setLineDash([4, 3]);
        c.strokeStyle = team === 0 ? BLUE : RED;
        c.lineWidth = 2;
        c.beginPath(); c.arc(p.x, p.y, r + 2 * Math.sin(t * 10), 0, Math.PI * 2); c.stroke();
        c.restore();
        this.drawUnitBody(p, this.cs * 0.45, team, e[9], 0.95, false, 1, 1);
        return;
      }
      const building = !!(flags & F.BUILDING);
      const hp = e[5], maxHp = e[6] || hp, symbol = e[9];
      // Only a placed card shows its deploy second (faded, with a ring). A
      // body another unit spawns (a Battle Ram's Barbarians, a Goblin Barrel's
      // Goblins: negative card ids) waits its deploy time too, but appears
      // solid, as in the game.
      const deploying = !!(flags & F.DEPLOYING) && e[1] > 0;
      if (building) {
        const half = Math.max(0.8, e[8] || 1) * this.cs;
        this.drawBuilding(p, half, team, symbol, flags & F.TOWER);
        this.drawHpBar(p.x, p.y - half - 7, half * 1.6, hp / maxHp, !!(flags & F.TOWER));
      } else {
        const r = this.cs * troopRadius(maxHp);
        const lift = flags & F.FLYING ? 1 : 0;
        this.drawUnitBody(p, r, team, symbol, deploying ? 0.55 : 1, false, 1, lift);
        if (deploying) {
          c.strokeStyle = 'rgba(255,255,255,.8)';
          c.lineWidth = 1.5;
          c.beginPath(); c.arc(p.x, p.y, r + 3, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * ((performance.now() / 1000) % 1)); c.stroke();
        }
        this.drawHpBar(p.x, p.y - r - 6 - lift * 4, Math.max(14, r * 2.2), hp / maxHp, false);
      }
    }

    drawUnitBody(p, r, team, symbol, alpha, building, scale, lift = 0) {
      const c = this.ctx;
      const main = team === 0 ? BLUE : RED, dim = team === 0 ? BLUE_DIM : RED_DIM;
      c.save();
      c.globalAlpha *= alpha;
      c.beginPath(); c.arc(p.x + 1, p.y + 2 + lift * 5, r, 0, Math.PI * 2);
      c.fillStyle = `rgba(0,0,0,${lift ? 0.18 : 0.25})`; c.fill();
      const py = p.y - lift * 3;
      const g = c.createRadialGradient(p.x - r * 0.3, py - r * 0.3, 1, p.x, py, r);
      g.addColorStop(0, main); g.addColorStop(1, dim);
      c.beginPath();
      if (building) roundRect(c, p.x - r, py - r, 2 * r, 2 * r, Math.max(2, r * 0.3));
      else c.arc(p.x, py, r, 0, Math.PI * 2);
      c.fillStyle = g; c.fill();
      c.strokeStyle = main; c.lineWidth = 1.5; c.stroke();
      c.fillStyle = '#fff';
      c.font = `700 ${Math.max(8, r * 0.95 * scale)}px ${FONT}`;
      c.textAlign = 'center'; c.textBaseline = 'middle';
      c.fillText(symbol, p.x, py + 0.5);
      c.restore();
    }

    drawBuilding(p, half, team, symbol, tower) {
      const c = this.ctx, main = team === 0 ? BLUE : RED, dim = team === 0 ? BLUE_DIM : RED_DIM;
      const x0 = p.x - half, y0 = p.y - half, sz = 2 * half, r = Math.max(2, this.cs * 0.18);
      c.beginPath(); roundRect(c, x0 + 1, y0 + 2, sz, sz, r);
      c.fillStyle = 'rgba(0,0,0,.25)'; c.fill();
      c.beginPath(); roundRect(c, x0, y0, sz, sz, r);
      const g = c.createLinearGradient(x0, y0, x0, y0 + sz);
      g.addColorStop(0, team === 0 ? '#3E6E9E' : '#9E4444'); g.addColorStop(1, dim);
      c.fillStyle = g; c.fill();
      c.strokeStyle = main; c.lineWidth = 2; c.stroke();
      c.fillStyle = '#fff';
      c.font = `700 ${Math.max(9, half * (tower ? 0.62 : 0.8))}px ${FONT}`;
      c.textAlign = 'center'; c.textBaseline = 'middle';
      c.fillText(symbol, p.x, p.y + 1);
    }

    drawHpBar(cx, y, w, ratio, big) {
      const c = this.ctx, h = big ? 4 : 3;
      ratio = Math.max(0, Math.min(1, ratio || 0));
      c.fillStyle = 'rgba(0,0,0,.5)';
      c.fillRect(cx - w / 2 - 1, y - 1, w + 2, h + 2);
      c.fillStyle = ratio > 0.5 ? '#4ade80' : ratio > 0.25 ? '#facc15' : '#ff5252';
      c.fillRect(cx - w / 2, y, w * ratio, h);
    }
  }

  // Draw order: spells, buildings, ground, air, projectiles.
  function order(e) {
    const f = e[7];
    if (f & F.SPELL) return 0;
    if (f & F.BUILDING) return 1;
    if (f & F.PROJECTILE) return 4;
    if (f & F.FLYING) return 3;
    return 2;
  }

  // Bigger bodies for tougher units: a Skeleton reads small, a Giant large.
  function troopRadius(maxHp) {
    const t = Math.min(1, Math.max(0, Math.log10(Math.max(1, maxHp) / 80) / Math.log10(60)));
    return 0.3 + 0.32 * t;
  }

  function roundRect(c, x, y, w, h, r) {
    c.beginPath();
    if (c.roundRect) c.roundRect(x, y, w, h, r);
    else c.rect(x, y, w, h);
  }

  root.LabBoard = LabBoard;
  root.LabBoard.FLAGS = F;
})(window);
