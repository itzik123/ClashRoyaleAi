# perception/

Turns a real Clash Royale match on screen into a stream of placement events,
and drives the existing simulator from them as a state estimator.

**Vision detects events only. The simulator holds the state.** The battle is
deterministic, so once *what* was played, *where* and *when* are known, the
engine derives everything else. Nothing here observes unit or tower HP as an
input to the engine — the required bandwidth is ~20 events per minute, not a
full state at 30fps.

Self-contained: nothing outside `perception/` is modified. `python_ai/` and
the C++ core are read-only from here.

---

## Status

| Stage | State | Evidence |
|---|---|---|
| 0 — capture, calibration | **calibrated; acceptance blocked upstream** | held-out landmark error **0.66 tiles** against the engine's geometry vs **0.34** against corrected geometry — same pixels, same solver. See *Findings*. |
| 1 — elixir reader | **PASSES** | 587 samples over a full match, mean confidence **0.990**, 13 low-confidence. Regen interval measures **2.80 s**, matching the real game exactly. |
| 1 — clock reader | **PASSES with confidence gating** | **97.1%** of readings correct at confidence ≥ 0.7 (207 of 295 samples). Free-running drift over the match is **~0.6 ms** (CFR measured at 30.0001 fps). |
| 2 — our hand and cycle | **logic done, icon templates pending** | cycle model reproduces the engine's FIFO exactly across a real replay, 0 desyncs. |
| 3 — opponent placement detection | **blocked on data** | `detect/placements.py` raises. Needs the next batch. |
| 4 — opponent deck, cycle, elixir | **done and tested** | deck discovery, exact elixir derivation, negative-balance alarm. |
| 5 — bridge + divergence | **done and tested** | zero-error control: divergence identically **0**. |

### Recordings

Three matches, 1920×1080 desktop capture (the emulator window occupies
686–1236 × 40–1012 of it), H.264, **constant frame rate confirmed by
measurement — jitter 0.0000**. Calibration profile:
`config/profile_gpg_1920x1080.json`.

`capture/window.py` (live capture) is deliberately not built. Nothing here
sends input to the game.

### Test suite

```bash
perception/.venv/Scripts/python.exe -m pytest perception/tests -q
```

58 tests, no emulator required. Every test runs against a frozen replay
fixture, a synthetic camera, or synthetic video generated at test time.

---

## The one hard problem, and how it was solved without touching the core

Feeding the **opponent's** placements to the simulator is easy —
`inject_enemy` spawns a card directly for team 1.

Feeding **our own** was the real blocker:

- `inject_enemy` hardcodes team 1; there is no `injectAlly`.
- `step()` also runs `opponentTurn()`, which places phantom heuristic cards.
- `step_self_play()` avoids that, but our play goes through
  `GameManager::playCard`, which needs the card to be in the simulator's own
  hand — and that hand is shuffled at reset by an unseeded `std::mt19937`,
  cannot be set, and the queue behind it cannot even be read.

**Solved by search.** `reset()` costs 0.135 ms (measured) and its shuffle is
uniform over all 70 hand-sets (measured), while `get_hand()` reads the result.
So: draw many resets, keep those whose hand-set matches our real opening hand,
and carry them all forward. Each play eliminates candidates that could not
have dealt what reality dealt. The queue holds four cards, so after four
confirmed deals every survivor has our exact cycle — and from there the two
FIFOs can never diverge again.

Measured: **0 refusals, 0 desyncs, divergence identically 0** on the control.

Three things this cost, all documented in `bridge/sim_driver.py`:

- The pool never collapses to one *object*. Hand slot arrangement is a free
  permutation nothing observable depends on, so ~24 exact duplicates survive
  together. What gets pinned is the cycle, not the object.
- The pool must be sized by coupon-collector on 24 queue orders, not by
  intuition. At ~31 candidates it silently failed about one match in four.
  It is 200.
- Opponent injections must be recorded in the replay history too. Omitting
  them made rebuilt candidates fight a match with no opponent at all.

---

## Findings reported upstream

Details and the requested change are in **`UPSTREAM_REQUESTS.md`**. Summary:

1. **The engine's board geometry does not match the real arena, and now there
   is a measurement of it.** Calibrating against the real game puts held-out
   landmarks at **0.66 tiles** using the engine's stated coordinates and
   **0.34 tiles** using corrected ones — identical pixels, identical solver,
   only the target coordinates differ. Three separate offsets:
   the left Princess tower sits a full tile left of its own bridge (reality
   aligns them), the Kings sit half a tile off the board centre, and the river
   band is half a tile off the towers' symmetry axis. The last of these is
   also why **team 1 has one row less placeable ground than team 0**
   (confirmed separately, 20 trials/row) — which affects **self-play
   training**, since `model.py` applies `own_half_rows = 16` to both sides and
   team 1's row-15 placements are silently rejected.
2. **The King Tower never sleeps.** `Tower.h` has no activation condition, so
   it fires from tick 0 while the real King is dormant until activated. King
   HP therefore diverges systematically from the first second regardless of
   perception quality — so the divergence metric excludes the Kings and
   reports them separately.
3. **Elixir runs ~2% slow.** `0.035/tick × 10 ticks/s` = 2.857 s per elixir
   against the real 2.8. Not corrected here; the opponent-elixir model uses
   the *real* rate because it models the real opponent.
4. **Requested (not blocking):** generalise `injectEnemy` to take a `team`,
   and bind `get_hand(team)`. Would remove the whole candidate-pool
   machinery. Everything works without it.

Corrections to the original brief:

- The river is a **two-row band `y ∈ [16,18)`**, and the bridges are the
  **points `(4,17)` and `(14,17)`**. The `x ∈ {3,4} / {13,14}` band is only
  the observation's channel-8 marker, not the geometry — using it as a
  calibration anchor puts every bridge landmark half a tile off.
- `replays/` is at `python_ai/replays/`, and **the training run rewrites it
  continuously** (observed dropping from 8 files to 1 mid-session). Fixtures
  are frozen into `tests/assets/`.
- Those replays carry `actionCardId` / `actionX` / `actionY` per tick, added
  by `train.py` — a labelled placement stream that let stages 4 and 5 be
  built and tested with no video at all. **They only label our own plays**;
  the heuristic opponent's are logged nowhere.
- The registry holds **173** entries: 132 playable + 41 Evolutions (123–163).
  Evolutions reuse the base card's name verbatim, so a name can never
  identify a card on its own.
- `.pyd` is current and `sample_random_deck` is bound. Fireball can cross the
  river — `placement_mask` gives spells all 34 rows.

---

## Open questions — these need answers, and are not guessed

1. **Phase boundaries.** `clock/match_clock.py` raises
   `PhaseScheduleUnknownError` until real single/double/triple/overtime
   boundaries are supplied. They move with balance updates, and the one
   consumer that acts on phase (opponent elixir) is corrupted silently by a
   wrong multiplier.
2. **Card map review — 52 rows.** `mapping/card_map.json` is generated from
   the registry, but 41 Evolution pairings need confirming and 11 ids ≥ 165
   ("Hero Knight", "Spirit Empress", …) need confirming as real cards in the
   build being recorded. Until verified they are excluded from the detector's
   vocabulary entirely. **Real cards absent from the simulator cannot be
   enumerated from here** and must be added by hand with `sim_id: null`.
3. **Own-play labelling.** No tap logger is needed —
   `readers/hand.infer_play_from_hand_change` reads our plays off the hand
   exactly. Confirm that's the preferred route before any input-side tooling
   is considered.

---

## What to record

Nothing was found on this machine, so this is still the gating item for
stages 1–3.

**First, one answer:** emulator on this PC, or a phone? It decides the profile
and whether a fixed window size can be assumed.

**Technical**
- Native resolution, no scaling. **The same resolution for every file** — a
  change invalidates every pixel constant and needs a new profile.
- MP4 / H.264, **constant frame rate**, 30 fps, ≥8 Mbps. OBS: Rate Control =
  CBR. `capture/video.py` measures this and **refuses** a variable-rate file
  rather than producing timestamps that drift during fights, which is exactly
  when placements happen. Phone recorders are usually VFR.
- No overlays: no FPS counter, no touch indicator, nothing that can sit over
  the elixir bar or the clock.
- **Start recording before pressing Battle.** The countdown and the 0:00 state
  both need to be on tape.
- Full matches, uncut, through to the result screen.

**Calibration frame:** just after the battle starts, before anything is
placed — both bridges unobstructed, all six towers visible, elixir bar and
clock visible. No cropping. Anchors are the four **Princess tower bases**
(where the tower meets the ground, *not* its visual centre — only the base
lies on the plane the homography models).

**Batches**

| Batch | Count | Unblocks |
|---|---|---|
| **now** | **3 matches** | stages 0–2 in full |
| next | +5 | one elixir-starved match (dimmed slots), one overtime |
| later | ~100–200 | stage 3's detector |

For the first eight, play the **same deck** and say which 8 real cards it is.

Put them in `perception/assets/recordings/` (gitignored).

---

## Layout

```
contracts.py        dataclasses only, no logic, no dependencies
timebase.py         the ONLY seconds<->ticks conversion. 10 ticks = 1 second
geometry.py         board landmarks, live from the engine where derivable
simlog.py           reader for the simulator's replay JSON
capture/            source.py (abstract) | video.py | frames.py
calib/homography.py screen<->tile, solved from 4 anchors, scored on held-out
readers/            elixir.py (pixels, no OCR) | clock.py (templates, no
                    Tesseract) | hand.py | towers.py (VALIDATION ONLY)
detect/placements.py opponent detection — blocked on recorded data
track/              cycle.py | opp_deck.py | opp_elixir.py
clock/match_clock.py free-running clock + resync; phase carried, not inferred
bridge/sim_driver.py drives ClashEnv as an estimator
mapping/            card_map.json + the unmapped-card contract
tools/              gen_card_map.py | divergence_report.py
```

### Why `timebase.py` says 10 ticks = 1 second

Not from the elixir rate. From `CardStats::attackCooldown`, which is stored in
ticks while the real game publishes the same figure in seconds — Archers
9/0.9s, Musketeer 10/1.0s, Knight 12/1.2s, Valkyrie 15/1.5s, Hog 16/1.6s,
King Tower 10/1.0s. 132 independent rows agreeing on one ratio beats any
single constant.

### The alarm worth knowing about

Derived opponent elixir going **negative is arithmetic proof that a placement
was missed** — they cannot spend what they do not have. Free, exact, no ground
truth. `track/opp_elixir.py` records it *before* clamping. The same logic
applies when their cycle wraps before all 8 cards are known
(`track/opp_deck.py`).
