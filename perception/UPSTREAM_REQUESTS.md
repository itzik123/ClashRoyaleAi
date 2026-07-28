# Requests and findings for the C++ core

Written from `perception/`, which changes nothing outside itself. Nothing here
has been applied.

Three items. **Item 1 is a bug in the engine that affects the training run
right now and has nothing to do with perception** — it should be read first.
Items 2 and 3 are a fidelity gap and a convenience request, neither blocking.

---

## 1. Team 1 has one row less placeable ground than team 0

**Severity: affects self-play training. Not a perception issue.**

### What was measured

20 trials per row, cheap cards only so affordability is never the limiting
factor, each team placing in **its own mirrored frame** — the frame its own
observation and its own policy use:

| row (own frame) | team 0 | team 1 |
|---|---|---|
| 13 | 17/20 | 13/20 |
| 14 | 17/20 | 15/20 |
| **15** | **15/20** | **0/20** |
| 16 | 0/20 | 0/20 |

(The sub-20 rates at rows 13–14 are the building-overlap check against units
already on the board, not the half boundary.)

Team 0 can place on row 15. Team 1 cannot. Ever.

### Why

`Board.h`:

```cpp
float riverY_start = 16.0f;
float riverY_end   = 18.0f;
```

The river band is `[16, 18)`, centred on **17.0**. But the tower layout is
symmetric about **16.5** — King 2.5 ↔ 30.5, Princess 6.0 ↔ 27.0, mirrored by
`y → 33 - y` exactly as `ClashEnv::extractObservationForTeam` does it.

The river is half a tile off-centre relative to everything else, and
`GameManager::isValidPlacement` gates the two teams on *different edges* of
that band:

```cpp
if (team == 0 && y > board.getRiverStart() - OWN_HALF_RIVER_BUFFER) return false;  // y <= 15.5
if (team == 1 && y < board.getRiverEnd()   + OWN_HALF_RIVER_BUFFER) return false;  // y >= 18.5
```

Mirrored into team 1's own frame, `y >= 18.5` becomes `y' <= 14.5`, against
team 0's `y <= 15.5`.

### Why it matters beyond aesthetics

`python_ai/model.py` computes `own_half_rows` **once**, from
`get_own_half_max_y()` (15.5 → 16 rows), and applies the same placement mask
to both sides. `train_selfplay.py` drives team 1 with that same network.

So a policy playing team 1 has row 15 marked legal in its mask, proposes
placements there, and the engine silently rejects them — `playCard` returns
`false` and the step proceeds as a no-op. The agent spends gradient on an
action that can never do anything, and only when it is playing team 1.

This is the same class of problem as the Giant-never-played incident already
documented in `gym_wrapper.py`: an action that is nominally available and
never actually works.

### Options (my preference first)

1. **Move the river to `[15.5, 17.5)`** so it is centred on 16.5 like
   everything else. Both teams then get `y <= 15.5` in their own frame.
   Fixes the asymmetry at its source. Changes pathing geometry slightly, so it
   needs a full `ClashRoyaleTests` run.
2. **Gate both teams on the same distance from their own side**, leaving the
   band alone — e.g. team 1 rejects `y < (BOARD_HEIGHT - 1) - getOwnHalfMaxY()`.
   Smaller blast radius; leaves the river visually off-centre.
3. **Leave it and make `model.py` use a per-team row count.** Cheapest, but
   it encodes the asymmetry into the training code permanently.

I have not applied any of these. Note that option 1 or 2 changes the legal
action set, which may invalidate `model_weights.pth`'s win-rate history.

---

## 2. The King Tower has no activation condition

**Severity: fidelity gap. Already worked around.**

`Tower.h` builds the King like any other tower and `GameManager::reset()`
gives it range 7.0 and a 10-tick cooldown. Nothing anywhere makes it dormant,
so it fires from tick 0. In the real game the King is inert until activated.

For perception this means predicted King HP diverges systematically from
observed King HP from the first second of every match, no matter how good the
event stream is.

**Already handled here, no change requested.** `SimDriver.divergence` excludes
the King towers and `readers/towers.king_divergence` reports them separately,
so the quality metric measures perception rather than this known gap.

Flagging it only because it is a real behavioural difference that will also
affect any policy trained against this engine — an agent learns that chip
damage to the King is punished immediately, which is not true of the real game.

---

## 3. Requested: `inject(cardId, x, y, team)` and `get_hand(team)`

**Severity: convenience. NOT blocking — a workaround is implemented and
tested.**

I want to be clear that this is no longer needed for correctness. I asked
about it before measuring; having measured, the pure-Python workaround works
and the control experiment passes with divergence identically zero. This is a
request to delete complexity, not to unblock anything.

### What is awkward today

`ClashEnv::injectEnemy` hardcodes team 1:

```cpp
void injectEnemy(int cardId, float x, float y) {
    const auto* card = CardRegistry::getInstance().getCard(cardId);
    if (card) card->spawnEntity(x, y, 1, game.getBoard());
}
```

There is no ally equivalent, so our own placements must go through
`playCard`, which requires the card to be in the simulator's own hand — and
that hand is shuffled at reset by an unseeded `std::mt19937`, cannot be set,
and the queue behind it cannot be read (`getHand()` is team 0 only).

### The workaround, and what it costs

`bridge/sim_driver.py` searches for the right shuffle: draw ~20,000 resets
(0.135 ms each), keep the ~200 whose hand-set matches our real opening hand,
carry them all forward, and eliminate those that could not have dealt what
reality dealt. After four confirmed deals the survivors have our exact cycle.

It works. It costs:

- ~2.7 s of startup per match, and ~1.4 s of redundant simulation
  (200 environments stepping in lockstep until the pool narrows);
- ~150 lines of machinery whose only purpose is to reverse-engineer a shuffle;
- residual play refusals, because `playCard` still checks elixir and placement
  legality against a board the estimate may have slightly wrong. Zero on the
  control, non-zero on approximate input.

### The change

Two small additions, both purely additive:

```cpp
// ClashEnv.h -- keep injectEnemy exactly as it is, so no existing caller
// changes. This is the general form.
void inject(int cardId, float x, float y, int team) {
    const auto* card = CardRegistry::getInstance().getCard(cardId);
    if (card) card->spawnEntity(x, y, team, game.getBoard());
}

std::vector<int> getHandForTeam(int team) const { return game.getHand(team); }
```

```cpp
// bindings.cpp
.def("inject", &ClashEnv::inject,
     py::arg("card_id"), py::arg("x"), py::arg("y"), py::arg("team"))
.def("get_hand_for_team", &ClashEnv::getHandForTeam, py::arg("team"))
```

No behaviour change to any existing path. `injectEnemy` untouched.
`get_hand()` untouched. No effect on the observation vector, the action
space, or any checkpoint — a rebuilt `.pyd` stays compatible with
`model_weights.pth`.

`game.getHand(int)` and `GameManager::playCard(team, ...)` already exist and
are already public; this only exposes them.

### What it buys

The entire candidate pool deletes. Own placements become one `inject` call,
exactly like opponent placements, bypassing hand/elixir/legality — which is
correct for an estimator, since the real game already validated the play and
elixir is derived independently. Startup drops to zero and the residual
refusal class disappears entirely.

**If the answer is no, nothing breaks.** The workaround stays and this
document records why it exists.

---

## Not requested

- **Game phases / overtime.** The brief asked for a clock that knows
  single/double/triple/overtime. `ELIXIR_REGEN_RATE` is a single `const float`
  applied to both players with no phase concept, and `oppElixirMultiplier` is
  a training-curriculum knob, not a game phase. The phase is carried as a
  field in `ClockState`, marked not-consumed, and deliberately left
  unconnected. **Not asking for the engine to grow phases.**
- **The 2% elixir gap.** `0.035/tick × 10 ticks/s` = 2.857 s per elixir
  against the real 2.8. Changing it would be a gameplay change mid-training
  run. `perception/` uses the real rate for the real opponent and the engine's
  rate when reasoning about the engine, and keeps them explicitly separate.
- **Card levels.** The registry has none. Recorded matches will have them,
  which is why `readers/towers.py` takes max-HP as a caller input rather than
  assuming the engine's values.
