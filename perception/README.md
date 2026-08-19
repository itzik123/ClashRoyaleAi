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
| 0 — capture, calibration | **PASSES** (2026-07-30) | independent check against the arena's **own rendered tile seams**: **0.105–0.224 tiles** across all 8 recordings — `tools/validate_grid.py`. Landmark residuals, for reference only: in-sample 0.31, leave-one-out 0.78. See *Findings* 1. |
| 1 — elixir reader | **PASSES** | 587 samples over a full match, mean confidence **0.990**, 13 low-confidence. Regen interval measures **2.80 s**, matching the real game exactly. |
| 1 — clock reader | **PASSES with confidence gating** | **97.1%** of readings correct at confidence ≥ 0.7 (207 of 295 samples). Free-running drift over the match is **~0.6 ms** (CFR measured at 30.0001 fps). |
| 2 — our hand and cycle | **templates exist; identity measurably WRONG** | cycle model reproduces the engine's FIFO exactly across a real replay, 0 desyncs — but on real video the icon template agrees with the elixir ledger on card cost only **33.8%** of the time (328 in-match plays, 8 recordings) and over-predicts Giant at 35% against a 12.5% prior. See *Findings* 5. |
| 3 — opponent placement detection | **blocked on data** | `detect/placements.py` raises. Needs the next batch. |
| 4 — opponent deck, cycle, elixir | **done and tested** | deck discovery, exact elixir derivation, negative-balance alarm. |
| 5 — bridge + divergence | **done and tested** | zero-error control: divergence identically **0**. |
| 6 — `State` → `GameState` adapter | **built, running live** (2026-07-31) | `live/adapter.py`. Closes the loop against the live emulator at **917 ms/iteration** inside a 1000 ms budget. Still missing: the match clock and cumulative elixir spend. |
| 6 — live frame source | **built and measured** (2026-07-31) | `capture/window.py`, Windows.Graphics.Capture. ~100 ms/frame, works while the window is covered. See *Live capture*. |
| 6 — per-unit HP | **fitted against 60 hand labels** | "is this unit damaged", population-weighted over 226 detections: precision **0.79 → 0.98**, recall **0.34 → 0.56**. See *Findings* 7. |

The live sensor's output contract is `GameState` in `contracts.py` — **shared,
neither side changes it alone**. Perception emits it and stops; encoding the
13,606 floats belongs to the training side, which owns the layout.

### Recordings

Eight matches, 1920×1080 desktop capture (the emulator window occupies
686–1236 × 40–1012 of it), H.264, **constant frame rate confirmed by
measurement — jitter 0.0000**. Calibration profile:
`config/profile_gpg_1920x1080.json`.

### The clock reader does not transfer to 549×976 (2026-07-31)

`readers/clock.py` scores 97.1% at confidence ≥ 0.7 **on the 1920×1080 desktop
recordings**. On live capture, where the same glyphs are ~2× smaller, it does
not work, and the cause is segmentation rather than the templates.

Measured, in two steps:

| ROI | verification |
|---|---|
| `(454, 0, 95, 54)` — whole panel | 5.65% correct |
| `(469, 21, 68, 24)` — digits only | 37.54% correct |

The first ROI included the "Time left:" caption, so the segmenter was chopping
up letters. The digit rows were then measured properly — near-white pixels
occupy rows 23–43 and the caption contributes **zero**, because it is cream
rather than white — which is where the second ROI comes from.

37.5% is still unusable, and `_split_mss_cells` is why: it returns widths of
`[19, 19, 22]` for *every* clock value sampled, when a content-based split
would vary (a `1` is narrower than a `0`), and it glues the colon into the
first cell at values like `0:58`.

**The templates themselves are fine.** Their sample counts are exactly the
distribution a correct M:SS countdown produces — 131/140/133 for digits 0/1/2,
which appear in the minutes, tens and units places; ~52-64 for 3-5, tens and
units; ~21 for 6-9, units only. The auto-labelling works; the reader that
consumes it does not, at this scale.

Two things worth keeping from this:

- **The verification is what caught it.** Every internal signal at build time
  looked healthy. Only reading the whole recording back and checking against
  independent arithmetic exposed it. Without that step the pipeline would have
  gained a clock reader emitting confident, plausible, wrong timestamps — and
  the clock anchors every timestamp downstream.
- **The live loop is not blocked on it.** `clock/match_clock.py` free-runs from
  frame timestamps and only *resyncs* against the screen, so the digit reader
  is a corrector, not the source. Anchoring the free-run on CRBAB's
  `screen == in_game` transition works today, with one measured caveat: that
  transition fires at t≈29.5 s while the clock reaches 3:00 at t≈19.9 s, so the
  anchor is **~10 s late** and needs the reader (or another landmark) to remove
  that offset.

### Live capture (2026-07-31)

`capture/window.py` reads the emulator window through
**Windows.Graphics.Capture**, chosen against two measured alternatives:

| method | per frame | fails how |
|---|---|---|
| `adb exec-out screencap -p` | 2243–4886 ms | an order of magnitude over budget |
| `adb exec-out screencap` (raw) | 1414–1885 ms | same |
| `CopyFromScreen` | fast | reads the *composited desktop* — returns whatever covers the emulator |
| **Windows.Graphics.Capture** | **~100 ms** | works while the window is fully covered |

The budget is 1000 ms, because the policy acts at 1 Hz. Measured live loop —
capture → resize → detect → `GameState`:

```
capture  74–127 ms | resize 40–102 | detect 514–818 | adapt 98–158
mean total 917 ms — fits, detector is 70% of it
```

Two things the surface is not: it is the **whole window** (title bar, right
toolbar, black pillarbox), so every frame is cropped to a per-frame-detected
game rect — the detector already invents Knights from player avatars and must
never see the chrome. And it is in **physical pixels**: `GetWindowRect` says
1536×816 while the surface is 1920×1020, because the display runs at 125% DPI.
Sizing anything from the window rect underestimates by 1.25×.

Game area on this display is **549×976**, i.e. 0.76 of native 720×1280. That
costs nothing measurable — see *Findings* 7.

### Test suite

```bash
perception/.venv/Scripts/python.exe -m pytest perception/tests -q
```

**73 passed, 4 skipped** — no emulator required. Every test runs against a
frozen replay fixture, a synthetic camera, or synthetic video generated at test
time.

The 4 skips are the ClashRoyaleBuildABot-dependent tests, which need onnxruntime
and opencv. To run those too, use the CRBAB environment instead — **121 passed,
1 skipped**:

```bash
perception/.venv-crbab/Scripts/python.exe -m pytest perception/tests -q
```

**215 passed, 1 skipped** as of 2026-08-05.

### Execution providers — `.venv-dml`

A third environment, identical to `.venv-crbab` except that `onnxruntime` is
replaced by `onnxruntime-directml`. The two cannot coexist: they install the
same `onnxruntime` package.

```bash
perception/.venv-dml/Scripts/python.exe -u perception/live/mvp_loop.py --policy neural
```

`onnx_detector.choose_providers()` picks the best available in the order
DirectML → CUDA → CPU, and `CRBAB_EP` forces one. The override matters for
measurement: comparing providers by switching venv also switches the ONNX
Runtime version (1.26.0 vs 1.24.4), which confounds the thing being compared.

Measured on `units_M_480x352.onnx`, both sessions built in one process and
**alternated frame by frame** — sequential runs are not comparable here,
because BlueStacks' own load drifts by nearly 10× and already produced one
bogus result:

| condition | CPU EP | DirectML | ratio |
|---|---|---|---|
| idle, BlueStacks running | 536.1 ms | 77.0 ms | **6.97×** |
| under 8 competing processes | 1430.1 ms | 205.4 ms | **6.96×** |

The ratio is the least interesting part. DirectML **under heavy load beats the
CPU provider idle**, and its p25 moves only 71 → 78 ms between the two
conditions: the GPU path is largely immune to the CPU contention that inflates
everything else on this box. That is why it is the right lever here — not
because 7× is a large number, but because the live problem is contention and
this is the only stage that can be removed from the contended resource.

---

## ClashRoyaleBuildABot, evaluated 2026-07-31

`clashroyalebuildabot/` is a vendored copy of an open-source bot, kept for two
things this module does not have: a **trained 97-class ONNX unit detector** and
an **actuation path** (`adb shell input tap`). Everything measured below comes
from a real BlueStacks device over ADB, not from the desktop recordings.

**It coexists with the engine.** Its `requires-python = "==3.12.*"` is a soft
pin — no 3.12-only syntax anywhere — and every dependency has a `cp311` wheel.
One Python 3.11 process hosts CRBAB, `clash_royale_env.pyd` and torch together.
The training venv is already on numpy 2.4.6, exactly CRBAB's pin.

**What it reads correctly**, verified against frames read by eye:

| signal | result |
|---|---|
| hand, 4 slots + next | exact on every ground-truth frame |
| elixir | exact (integer only; `readers/elixir.py` is finer) |
| screen state | correct — `in_game` / `bypass_end_of_game` / `lobby` |
| Princess tower HP | correct, and already a **fraction** |
| units, in-game | 4.6/frame, **0% impossible classes** over 71 ladder frames |

**Three defects, two of them fixed here in `live/`:**

1. **31% of in-game detections are off-board phantoms.** Over 71 ladder frames,
   102 of 328 detections fell outside the 18x32 board and **every one was
   `knight`** — the two player avatar icons at `(19,5)` x68 and `(-2,13)` x33.
   `knight` drops from 111 to 9 once they are removed. Fixed by
   `live/board_filter.py`; upstream tracks it as issue #244.
2. **King HP is never read.** `Numbers` has only the four Princess fields, so
   extra scalars 3 and 6 could not be filled. Fixed by `live/king_hp.py`.
3. **Side classification can invert.** Confirmed on a ladder frame: our own
   Valkyrie — blue badge, attacking the enemy King — was reported as
   `enemy valkyrie`. Not fixed. `side.onnx` is a learned classifier where the
   badge colour beside every unit is an exact answer; that is the obvious
   repair and it reuses the badge work in *Findings* 5.

**Latency is the binding constraint.** Full `Detector.run` is **812 ms** on an
idle frame and **932 ms** on real battle frames, against a 1 Hz decision budget.
The ONNX model is **86%** of it (`_infer` 692 ms vs 43 ms preprocess + 14 ms
postprocess), so there is no Python overhead worth optimising. DirectML on the
integrated GPU gives **438 ms vs 499 ms** on identical tensors — real, but only
~12%, and ORT 1.24.4 and 1.26.0 measured equivalent.

**UI overlays cross the playfield** at exactly the high-stakes moments — the
"30 Seconds Left" stopwatch, the "You"+crown award, `x2`, the red tint on tower
loss. Their effect on recall is not yet measured.

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

1. **RESOLVED 2026-07-30 — the engine's board geometry now matches the real
   arena.** All three offsets have been fixed upstream: the river band
   (2026-07-29), and the left Princess towers `x = 3.0 → 4.0` plus the Kings
   `x = 8.5 → 9.0` (commit `dd99991`, `.pyd` rebuilt after it). `engine_tiles()`
   and the old hypothetical `corrected_tiles()` were verified identical, so the
   two-column comparison in `tools/calibrate.py` was deleted as that code's own
   comment said it should be.

   The profile was re-solved against the corrected geometry. Landmark residual
   went **0.63 → 0.31** in-sample, matching the prediction in
   `UPSTREAM_REQUESTS.md`'s measured table exactly.

   **What the investigation actually changed is which number to trust.** All
   landmark metrics are self-referential, and two plausible "improvements"
   were tested and disproved:

   - anchoring on the tower **base** rather than the blob centroid, as
     `calib/homography.py`'s docstring advises in general — leave-one-out went
     0.54 → **4.68**. The detector finds the flat stone *platform*, which is
     already a ground-plane feature, so its centroid is the footprint centre;
     the bottom edge is half a platform too far forward.
   - **dropping** the two hardcoded `own_princess` constants, which score worst
     on every landmark metric and were never detected in any recording —
     improves every landmark metric (in-sample 0.31 → 0.17, LOO 0.78 → 0.54)
     and makes the mapping **3.4× worse** in our own half (mean |dy| 0.105 →
     0.357). They are the only near-side `y` constraint in the fit.

   Acceptance is therefore read off `tools/validate_grid.py`, which scores the
   mapping against the arena's **own rendered tile seams** — independent of the
   landmark set, because the fit never saw them. **0.105–0.224 tiles, PASS.**
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
   machinery. Everything works without it. **Partly landed** — `inject()` and
   `getHandForTeam()` appear in commit `0ab809b`.
5. **Our own hand identity is measurably wrong on real video** — a perception
   bug, not an engine one, recorded here because it invalidates the stage-2
   "logic done" claim. Over 8 recordings and 328 in-match candidate plays the
   icon template agrees with the elixir ledger on card cost only **33.8%** of
   the time, while the observed elixir drop lands within 0.6 of a real cost
   **74.7%** of the time. The template over-predicts Giant at 35% against a
   12.5% prior. Direct proof: Giant is played out of slot 1 at t=21.0s and one
   second later reads as present in slot 2, holding for 10 seconds — a played
   card goes to the back of an 8-card queue, so this is impossible, and no
   debouncer can catch a *stably* wrong classifier (raising `hold` from 2 to 8
   leaves 7 flip-flops per match in two recordings).

   Two prerequisites surfaced with it. The **first ~18 s of every recording has
   no match**: the elixir bar is not yet drawn so its ROI reads a *confident*
   0.00, and the hand slots read garbage like `[2,2,2,2]`. There is no
   match-state machine anywhere in the module, and its absence poisoned every
   measurement taken before it was gated. And **phase is directly observable** —
   the clock ROI reads `Overtime` and a large `x2` badge is drawn on screen — so
   the phase schedule in "Open questions" 1 below is no longer needed.
6. **Tower HP: the engine models level 9, the recordings are levels 4-5.** Read
   off a clean frame at t=20 s, before anything is damaged, so the on-screen
   numbers are the true maxima: our Princess **1750** (badge 4), the opponent's
   **1890** (badge 5), against the engine's 2534 for both. Those are exactly the
   real game's level-4 and level-5 values, and 2534 is its level 9. So absolute
   HP is not comparable and `hp / MAX_BUILDING_HP` would be wrong by ~30% and by
   a *different* factor per player. Handled entirely on this side: report a
   **fraction**, and take the maximum from the first undamaged reading rather
   than a supplied table. Also note **King HP is not rendered until the King is
   activated** — an absent numeral means full HP.

7. **Per-unit HP, fitted against 60 hand-labelled crops (2026-07-31).**
   ClashRoyaleBuildABot reports no unit HP at all, but observation channels 0-7
   store `hp / MAX` per cell — 4,896 of the 13,606 floats. Scored as an "is this
   unit damaged" detector over 226 detections, reweighted from the stratified
   sample to the population: precision **0.79 → 0.98**, recall **0.34 → 0.56**.

   Three faults, found in this order because each hid the next:

   - the **association window was inverted** — it rejected badges more than
     12 px *below* the box top, which is where the correct ones are (measured
     dy −14.5 to +1.5), while accepting badges up to 104 px *above*, which
     belong to other units. Every false positive came in that way;
   - the **fill test was a brightness threshold**, i.e. an accidental ally
     detector: ally fill (111,208,252) means 190.3 and passed by 3 points,
     enemy fill (224,35,93) means 117.3 and never could. Every enemy unit
     measured 0% fill;
   - **two units could claim the same badge**, and two vertically stacked
     widgets merged into one blob the size filter discarded whole.

   The remaining gap is badge **detection**, not association — 8 of the 10
   surviving misses have no badge found near the unit at all.

   **The badge-hue side oracle did not survive measurement.** An earlier
   reading put its disagreement with CRBAB's `side.onnx` at 31% with the badge
   right every time checked; that was measured with the broken matcher and was
   largely counting bad association. Re-measured it is **10%** (7 of 67), and
   of five checked by eye the badge was right twice and wrong twice. Neither
   source dominates, so nothing is overridden — `adapter.py` carries both
   readings and lowers confidence when they differ.

8. **The detector misnames ~15% of the units it finds.** The same labels
   returned a second finding: 9 of 60 carried the wrong card name (`knight` ×6,
   `valkyrie` ×2, `archer`) and 3 were not units at all (`minipekka` ×3, one
   confirmed to be the enemy Princess tower). `knight` is the same class that
   produced all 102 off-board phantoms. Not fixable here — the model is
   upstream's, vendored unchanged. Recorded in `BOT_REQUESTS.md` item 6,
   because a wrong `card_sim_id` drives channels 11-20 with a confidently wrong
   attribute row, which is worse than a dropped detection.

9. **Tower HP is now read as the ABSOLUTE printed numeral (2026-08-09).**
   `readers/tower_numerals.py` plus a tower-specific digit set at
   `config/templates/tower_549x976`. This replaces the bar fraction for the
   timing work: CRBAB's `_calculate_hp` returns 0.0 both when a bar reads empty
   and when it cannot match the colours at all, which over 77 consecutive
   frames put both of our live Princess towers at 0.00 the whole time. A reader
   that cannot tell a live tower from an unreadable one is unusable as a clock,
   and it is what blocked measuring the real game's spell delay.

   **The clock's templates measurably do not transfer** — confidence 0.08–0.18
   against a 0.35 threshold, "3" read as "1" — so the set was cut fresh from the
   8 recordings by `tools/build_tower_digit_templates.py`. 5,389 crops, 4,100
   usable, clustered into 30 groups and labelled by cluster rather than by cell
   (`tools/labels_549x976.json`); 138–614 samples per digit.

   Two results:

   - **Acceptance**: both enemy Princess towers on the known live frame read
     **2030**, end to end through `TowerNumeralReader`. Pinned in
     `tests/test_tower_numerals.py`.
   - **Sequence**: read back over 4 full matches, **542 steps, 12 upward jumps,
     97.8% consistent** with the fact that tower HP never rises. That is free
     ground truth needing no labels, in the same spirit as the clock builder's
     read-back-against-the-arithmetic check, and it exercises HP values no
     labelled cluster ever showed as a whole number. The 12 failures are
     0↔8 and 5↔6 confusions, mostly at confidence below 0.5.

   **The key invariant, found by two failed attempts:** the numeral is drawn
   near-white with a dark outline, and that is the only property that does not
   vary with the arena skin. `ink_channel` + Otsu — which the reader still uses
   — separates the glyph from *saturated* backgrounds like grass (65–75 against
   the glyph's 218–230) but **not** from the light-tan floor all 8 recordings
   are played on (116–185), where Otsu is forced to fit three levels with two
   classes, puts the floor on the ink side, and merges adjacent digits: 588 of
   2,711 cells came out as merged pairs, including every "90" in the batch.
   Thresholding the per-channel minimum above 200 separates all of them.

   **Two things are still open**, both measured and neither yet fixed:

   - `split_digits` should segment on the near-white mask rather than Otsu.
     The reader is currently correct on grass and unreliable on other skins.
   - **`NUMERAL_OFFSET` is enemy-only.** The ally numeral is drawn *on* its
     bar, not above it, so the derived ROI lands on the tower roof and returns
     masonry. Only `BAR_ENEMY_*` was ever exercised. Ally and King towers need
     their own offsets before all six tower scalars can be filled.

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
