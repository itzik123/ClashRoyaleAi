# Session handoff — 2026-08-13 (second session)

Read this before touching the live loop or the expert-iteration pipeline.
Everything here is measured unless it says otherwise.

---

## 0. RESOLVED: the placements were landing off the board

The previous session reported **"0 illegal placements out of 32 live"** off a
circular test — it validated placements against `is_valid_placement`, the same
predicate the mask was built from. That warning stands as a method lesson.

**The cause is now found, fixed and verified live, and it was not legality at
all. It was the tile grid.**

The 2026-08-05 refit of `TILE_WIDTH/HEIGHT/INIT_X/INIT_Y` was wrong in *both*
axes, so the actuator computed tap pixels for a board 28 px too wide and 23 px
too tall. Engine row 1 — the back row the policy uses constantly — was tapped
at y=989 against an arena that stops at y=981. The tap landed below the board,
the game ignored it, and the elixir ledger reported "issued but never
confirmed".

Measured, live, with the elixir bar at its cap so acceptance is unambiguous:

| | before | after |
|---|---|---|
| engine row 1 accepted (controlled probe) | **0/6** | **6/6** |
| engine rows 2, 5, 15 accepted | 12/12 | 12/12 |
| live placements tapping outside the board | **13/34 (38%)** | **0/28 (0%)** |
| live engine-row-1 placements ledger-confirmed | **0/11** | **7/9** |

Fisher exact on that last row: **p = 4.6e-4**.

**The oracle that broke it open: Clash Royale draws the answer itself.**
Selecting a card tints the region you may NOT deploy into red, so differencing
a selected frame against an unselected one is a dense per-pixel readout of the
real rule — 288 cells at once, zero elixir, no policy and no detector in the
loop. Redness delta is ~60 inside the forbidden region and ~0.1 outside it, so
the threshold is not a tuning parameter. `tools/deploy_zone.py`.

**Why the bad refit passed its own cross-check, which is the transferable
lesson.** It scaled each axis off a landmark *separation* whose tile count was
assumed rather than measured, and both counts were short by one (the river gaps
are 11 tiles apart, not 10; the princess HP bars 22, not 21). Both fits were
anchored on the board centre, so the error is zero in the middle and grows
outward — and the symmetry check it congratulated itself on ("engine x 9.0
lands on display centre 360") is *structurally incapable* of seeing a scale
error anchored at the centre. Both the wrong grid and the right one pass it.
The arena's **edges** discriminate; the centre cannot.

That is the fifth instance of this project's recurring failure: an aggregate
that shares an assumption with the thing it is checking. `test_tile_grid.py`
now pins the board's four edges against the game's own rectangle.

---

## 1. The wins (these are solid)

### 1.1 Decision-time search — the large, replicated effect

`python_ai/search_ab_test.py`. 160 PAIRED trials (both arms handed a
bit-identical opening via `env.snapshot()`), 1.5x opponent elixir, ep-64k
checkpoint:

| | |
|---|---|
| greedy policy | 0.625 |
| + 1-ply search | **0.944** |
| paired delta | **+0.319**, 95% CI [+0.237, +0.401] |
| exact McNemar | p = 5.6e-12 |
| deviation rate | 13.8% |
| cost | 2.2x wall clock |

Replicated across five independent collections (expert win rate 0.944 / 0.950 /
0.887 / 0.910 / 0.940). **This is the strongest result the project has.**

Search is scored by the network's OWN critic, and greedy is always candidate 0,
so search only deviates when the critic disagrees with the action head. That
means the value head is substantially better than the action head at exploiting
it — which is the gap expert iteration exists to close.

### 1.2 Expert iteration — real but modest

`python_ai/expert_iteration.py`. Paired greedy-vs-greedy, no search either arm:

| net | n | original | distilled | delta | p |
|---|---|---|---|---|---|
| hard-label argmax | 800 | 0.634 | 0.650 | +0.016 | 0.553 |
| distribution, 4 ep | 800 | 0.629 | 0.666 | +0.038 | 0.095 |
| **distribution + DAgger** | **1600** | **0.649** | **0.693** | **+0.045 [+0.013, +0.077]** | **0.0074** |

Significant, survives Bonferroni for the three evals (alpha = 0.0167). Result
net: **`python_ai/model_weights_dist_e3.pth`**.

Read the size honestly: **+4.5 points is ~14% of what search itself buys**. The
last two nets are statistically indistinguishable — the jump in significance
came from doubling n, NOT from DAgger making a better policy. Do not claim
DAgger raised win rate; it raised conditional lift.

**What works:** distilling the search's VALUE DISTRIBUTION (softmax over
candidate values, T=0.05) with the trunk FROZEN, plus one DAgger round.
**What does not:** hard-label argmax distillation (+0.016, p=0.55).

### 1.3 The metric that made this legible: `conditional_lift`

87% of the expert's overrides are "wait where greedy plays", so a policy that
merely no-ops more scores on naive agreement metrics for free. A 4-config
ablation produced a clean-looking ladder (0.235 / 0.246 / 0.372 / 0.655) that
was **entirely an artifact** — perfectly monotonic in each config's no-op rate.

`conditional_lift` = P(policy waits | greedy plays, expert waited)
                   − P(policy waits | greedy plays, expert played).
Null is exactly 0. Only a state-conditional rule separates the two terms.

| config | lift | p1/p0 | no-op |
|---|---|---|---|
| null | +0.0000 | — | 0.802 |
| dist 4ep/80 eps | +0.1027 | 3.07 | 0.822 |
| E1 16ep/80 eps | +0.1268 | 2.31 | 0.834 |
| E2 16ep/180 eps | +0.1535 | 2.17 | 0.843 |
| **E3 DAgger/180 eps** | **+0.1733** | **2.45** | 0.843 |

**More DATA is actively harmful.** Coverage kept rising while lift stalled,
because p0 rose faster than p1 — selectivity fell 3.07 → 2.31 → 2.17 and the
no-op rate walked toward the expert's 0.896. That is marginal drift, the same
failure hard labels produced. DAgger at a FIXED 180-episode budget was the only
lever that reversed it.

Temperature must be chosen from the TARGET's entropy, never tuned on the
outcome: at T=0.25 the target sits at 94% of max entropy (near-uniform, no
signal) while looking like it is training. `--target-entropy` prints the table.

A no-op duplication in `_build_candidates` silently destroyed the first
attempt: `(NOOP, gx, gy)` and `(NOOP, 0, 0)` are the same action but differ as
tuples, so 99.1% of rows had a target with median value spread of EXACTLY 0.0.
Fixed; 32.6% of rows now carry a real distribution, and search got ~2x faster.

### 1.4 Engine: snapshot / deepCopy

`Board::deepCopy()`, `GameManager::snapshot()`, `ClashEnv::snapshot()`, bound as
`env.snapshot()`. Purely additive — zero deletions in every C++ file except
`MatchStatistics.h` (8, extracting a shared subscribe list). Not
gameplay-affecting; `observation_size()` unchanged at 13606; no checkpoint
invalidated. **530 tests / 5246 assertions**, verified across 200 consecutive
runs (one flaky test found and fixed that way).

Two silent aliasing hazards closed: `Projectile::target` (the only
entity-pointer member; repaired by a virtual remap pass, since `Board.h` cannot
include `Projectile.h`) and `Board::statsEvents` (stateful collectors — copying
the subscriber list would post rollout damage into the live match's stats,
which feed reward shaping).

### 1.5 DirectML — 3.1x on the live detector

The live venv had **CPU-only `onnxruntime`**, which I installed by mistake. The
pipeline was built around DirectML (`action_gate.py`'s docstring records the
producer at 3.99 Hz with it).

```
perception/.venv/Scripts/python.exe -m pip uninstall -y onnxruntime
perception/.venv/Scripts/python.exe -m pip install onnxruntime-directml
```

| | CPU | DirectML |
|---|---|---|
| `detector.run` median | 556 ms | **181 ms** |
| producer | 1.19 Hz | **3.15 Hz** |
| board age mean | 2508 ms | **433 ms** |
| actions blocked stale | **32 of 39** | **0** |

`MAX_STALENESS_MS = 2000` in `live/action_gate.py` was gating out 82% of the
bot's actions. **Check `execution provider:` on line 1 of every live run.**

### 1.6 Row-0 coordinate fix (partial — see Open Problem 1)

`TILE_Y_OFFSET = 1`, so engine row 0 converts to DETECTOR row -1, and
`engine_tile_centre(9, 0)` taps pixel y=1018 against an arena bottom edge of
`DISPLAY_HEIGHT - TILE_INIT_Y = 1003.81`. The tap lands in the dead strip above
the card tray; the card is deselected and nothing deploys.

`engine_row_is_tappable()` (in `live/actuator.py`) masks those rows in the LIVE
path only — deliberately not in `model.py`, since narrowing the training action
space would change what the policy learns to fix a display artifact.

This is real and it helped (the human's run went 18 → 4 unconfirmed). It is
**not** the whole story — see below.

---

## 2. RESOLVED: what the real game's deploy rule actually is

Measured directly off the tint, for every card in the deck, at 288 cells:

**The real game allows engine rows 1..15 × all 18 columns — 270 cells.**
Engine row 0 has no arena row at all (the engine's board is 34 rows, the arena
is 32) and stays masked by `engine_row_is_tappable`.

Two things follow, and the second is a live problem worth its own entry:

- **`TILE_Y_OFFSET = 1` is correct.** It was carried for months flagged as
  "derived, not verified". It is now verified: rows 1..15 accept at 100% and
  row 16 is across the river.
- **The engine's `is_valid_placement` is STRICTER than the real game, not
  looser** — it refuses 34–65 cells the game permits (Cannon 208/288 vs 270,
  troops 242/288 vs 270). So the third live mask is throwing away legal
  placements, most of them the tower-adjacent defensive cells a Cannon most
  wants. It never caused a refusal; it costs action space. Removing it is a
  live-path change with no engine impact — untested, so it stays for now.

Also confirmed: the real game does **not** restrict the back row to the middle
six columns. CRBAB's own `ALLY_TILES` says it does (`y=0` limited to x 6..11)
and that is simply wrong — x=0 and x=17 at engine row 1 both accept.

---

## 3. OPEN PROBLEM 2: the ledger / `unconfirmed` mismatch

Live Training Camp run, 200 s, `dist_e3`, `--act`:

```
32 placements issued
 0 engine-illegal   (but see section 0 -- circular)
 0 on row 0
actions: 32 allowed, none blocked
our elixir spent: 70 over 21 cards, residual -18, 17 issued plays never confirmed
```

**The in-flight elixir deduction ALREADY EXISTS and is correct.** I proposed
implementing it, then found it at `live/mvp_loop.py:516`:

```python
owed = ledger.unconfirmed_cost          # @property -> float, correctly called
if owed:
    gs = replace(gs, my_elixir=max(0.0, gs.my_elixir - owed))
```

It runs in `perceive()`, before `my_elixir` reaches `affordability_mask`. So the
double-commit theory is **wrong** — that protection is in place and the 17 still
happen. Do not re-implement it.

What `rejected` actually counts: `_expire()` increments it for plays whose cost
was never matched to an observed elixir drop within
`PLAY_CONFIRM_WINDOW_S = 4.0`. That is a **confirmation** failure, not
necessarily a placement failure. `_confirm` matches drops by subset-sum over
pending costs (3/4/5), which with a noisy integer bar reading and an opponent
also spending is entirely capable of failing to reconcile a play that landed
fine.

`residual -18`, by the ledger's own definition, "means spend was invented".

**So `unconfirmed` conflates at least three things** and cannot currently
distinguish them: the game refused the tap; the tap never arrived; the ledger
could not reconcile the elixir trace.

**Now instrumented and largely explained.** `live/placement_confirm.py` adds two
oracles independent of the bar (a unit of the expected type appearing, and the
hand slot cycling) and cross-tabulates them against the ledger, keyed per play
via a new `tag` on `ElixirLedger.record_play`. After the grid fix the ledger's
residual went **−19 → −4 → +1** across three runs, i.e. it stopped inventing
spend, because the spend it was recording is now real.

**A caution on the hand oracle, measured.** In the live loop it reported
33/34 placements as "the slot cycled" during the pre-fix run — including 10 of
the 11 engine-row-1 placements that the controlled probe proves never spent a
single elixir (0/6) and that the ledger scored 0/11. The card reader is noisy
enough under load that any misread inside the 3.5 s window trips it. Treat
`hand_changed` as a **lower bound on refusals**, never as confirmation on its
own; the elixir-at-cap probe in `tools/placement_truth.py` is the oracle to
trust when it matters.

---

## 4. DONE: unit-appearance confirmation

The elixir bar is the wrong oracle. **Whether our unit appeared is the right
one**, and `GameState.units` already carries team.

Concrete plan:

1. In `live/mvp_loop.py`, when a placement is issued, record
   `(tick, slot, card_id, tile_x, tile_y, own_unit_count_at_issue)`.
2. On each subsequent perceived board for the next ~2 s (the tap takes ~900 ms
   to land and the resulting board is seen ~700 ms later — `PLAY_CONFIRM_WINDOW_S`'s
   own comment gives ~1.6 s as the legitimate confirmation latency), check
   whether an own-team unit appeared that was not there before, ideally near the
   requested tile.
3. Report a THIRD counter alongside `allowed` and `rejected`:
   `confirmed_by_unit`. Cards that spawn multiple bodies (Minions x3, Archers
   x2) need the count delta, not a boolean.
4. Cross-tabulate `confirmed_by_unit` against the ledger's `rejected`. That
   2x2 is the whole answer:

   | | ledger confirmed | ledger rejected |
   |---|---|---|
   | **unit appeared** | healthy | **ledger metric bug** |
   | **no unit** | impossible-ish | **the game really refused it** |

5. For every "no unit" case, log the exact `(card_id, tile_x, tile_y)` and the
   pixel the actuator tapped. **That list is the ground-truth dataset of what
   the real game rejects** — the thing Open Problem 1 has been missing. Fit the
   real legality rule to it rather than to `is_valid_placement`.

This separates the three causes in section 3 and simultaneously produces the
data needed for section 2. It is the highest-value next action.

Secondary: the sim-vs-live placement distribution gap. In simulation both nets
place at y<=1 about 19-21% of the time; the first live run was 60% (15/25). The
encoder is NOT the cause — `perception_encoder.encode()` and
`getObservationForTeam(0)` agree EXACTLY on a towers-only baseline (0 differing
floats of 12,852 spatial and 754 scalar; pinned by
`perception/tests/test_encoder_matches_engine.py`). So any remaining
discrepancy is in unit DETECTION, not encoding.

---

## 5. What is where

| file | what |
|---|---|
| `perception/tools/deploy_zone.py` | **the real deploy zone, off the game's own tint**; `--fit-grid` refits the tile grid to the arena rectangle |
| `perception/tools/placement_truth.py` | controlled per-cell accept/refuse at capped elixir; `--pixel-scan` for raw-pixel boundary hunting |
| `perception/tools/match_nav.py` | unattended navigation into a Training Camp match from any screen |
| `perception/live/placement_confirm.py` | the unit/hand oracles and the 2x2 against the ledger |
| `perception/tests/test_tile_grid.py` | pins the board's four edges to the measured arena |
| `python_ai/search_ab_test.py` | paired search-vs-policy A/B |
| `python_ai/expert_iteration.py` | collect / distil / eval / ablate / lift / target-entropy |
| `python_ai/make_replays.py` | paired replay JSONs for `web/viewer.html` |
| `python_ai/model_weights_dist_e3.pth` | **the +0.045 result net** |
| `python_ai/model_weights_selfplay.pth` | original, untouched |
| `perception/tools/enter_training_camp.py` | automated lobby → Training Camp over adb |
| `perception/tests/test_tappable_rows.py` | row-0 regression |
| `perception/tests/test_encoder_matches_engine.py` | encoder/engine equivalence |
| `tests/core/test_board_deepcopy.cpp` | snapshot divergence + negative cases |
| `tests/core/test_game_manager_snapshot.cpp` | GameManager-level snapshot |

Label datasets (`*.npz`) and weights (`*.pth`) are gitignored — regenerate with
`expert_iteration.py --collect` (~3 s/episode).

Test commands:

```bash
perception/.venv/Scripts/python.exe -m pytest perception/tests -q
```

```bash
./build_test/Release/ClashRoyaleTests.exe
```

337 perception tests (1 skipped) and 530 C++ cases pass as of this commit.
No C++ or `python_ai/` file was touched this session — every change is inside
`perception/`, so no checkpoint or win-rate history is affected.

Ground-truth probes (need the emulator; they place real cards):

```bash
perception/.venv/Scripts/python.exe perception/tools/deploy_zone.py --ensure-match --slots 0,1,2,3
```

```bash
perception/.venv/Scripts/python.exe perception/tools/placement_truth.py --rows 1,2,15 --xs 0,9,17
```

Live run (queue a Training Camp match first, or use the tool above):

```bash
perception/.venv/Scripts/python.exe -m perception.live.mvp_loop --policy neural --checkpoint python_ai/model_weights_dist_e3.pth --act --seconds 180 --window "BlueStacks App Player"
```

---

## 5b. Still open, and now the largest live problem

**The agent issues placements faster than its elixir can support.** Measured on
the pre-fix run: 34 placements in 160 s, one every 4.7 s, against a sustainable
rate of one per ~9.5 s at 2.8 s/elixir and a 3.4 average cost. It sat at 0–3
elixir for the whole match.

The grid fix improved this a lot without addressing it directly — once taps
actually deploy, they actually cost elixir, so the affordability mask throttles
properly: 34 → 20 placements, spend 82 → 55 (against ~62 available), residual
−19 → −4. But 5 of 28 post-fix placements still go unconfirmed by the ledger,
and the remaining suspects are the ~1.6 s round trip between deciding and
landing, and the integer elixir reading.

Worth measuring next with `tools/placement_truth.py`'s oracle rather than the
ledger, since the ledger is the thing under suspicion.

**Do not re-open "the game refuses our placements" without new live evidence.**
It was measured at 0/28 this session across two matches.

---

## 6. Discipline notes worth keeping

- **Never validate a mask against the predicate that generated it.** Section 0.
- **A cross-check anchored where the error is zero proves nothing.** The bad
  tile grid passed "engine x 9.0 lands on display centre 360" because both fits
  were centre-anchored and the scale error grows outward. Check the edges of
  the range you care about, not its middle.
- **Every failure mode of the deploy-zone probe is "everything is legal"** — an
  unaffordable card selects nothing, a stale baseline already holds the tint, a
  tap pixel off the board is untinted because there is no board there. When a
  measurement's failure mode is maximal permissiveness, it needs an internal
  control that must fire (here: a band deep in the enemy half that MUST be
  tinted whenever a card is really selected).
- **The game is a better oracle than the simulator, and it is free.** The
  deploy-zone tint answers in one screenshot what a season of `is_valid_placement`
  comparisons could not.
- **The control arm's own variance is large**: original greedy measured 0.625,
  0.570, 0.700, 0.634 across four runs at 1.5x elixir. Any comparison under a
  few hundred paired trials measures that, not the treatment.
- **Screen on a different metric than you confirm on.** Selection here was on
  conditional lift; the win-rate eval stayed a single confirmatory test. An
  exploratory n=200 run gave +0.105 at p=0.044 and it was NOISE — the
  confirmatory run at 4x the power killed it. Do not resurrect that number.
- **A p-value from "run until significant" is worthless.** Three win-rate evals
  were run in total and all three are reported, including two nulls.
