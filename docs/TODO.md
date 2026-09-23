# TODO — the single list of genuinely pending work

Consolidated 2026-08-19 from `CLAUDE.md`, three session handoffs,
`PLACEMENT_COLLAPSE.md`, two executed plans, `perception/README.md`,
`perception/UPSTREAM_REQUESTS.md` and `perception/BOT_REQUESTS.md`.

**Everything here was verified against the source tree, not copied from a
document's own status field.** Items whose work turned out to be already merged
were deleted rather than carried forward — three of them had been sitting in
`UPSTREAM_REQUESTS.md` marked OPEN while the exact code they proposed was in the
tree.

Rules that apply to every item below:

- `include/`, `src/` and `python_ai/` are READ-ONLY unless explicitly asked.
  C++ changes need the exact diagnosis and the exact edit confirmed **first**,
  and go to `perception/UPSTREAM_REQUESTS.md` as a proposal. **That file is
  deliberately EMPTY as of 2026-08-24** — the backlog was worked to zero and its
  33 items are archived in `DECISIONS.md` under "ARCHIVE", headings intact so a
  citation by number still resolves. An empty file means no open proposals, not
  a lost one; start numbering new items at 26 so archived references stay
  unambiguous.
- **Confirm a diagnosis before implementing it, even when the item states one.**
  UPSTREAM item 20 asserted "spawned entities carry no cardId"; they carry
  distinct negative ids and the right name, and the fix it proposed would not
  have named them. The item had been read three times without that being caught.
- Any gameplay-affecting change invalidates the win-rate history. Say so.
- Python is 3.11 only: `python_ai/venv/Scripts/python.exe`.

---

## 00. Pre-launch audit follow-ups (2026-09-15) — open, none blocks the run

The audit and its fixes are in `.claude/CLAUDE.md` ("2026-09-15: the pre-launch audit");
the operator's steps are `docs/runbooks/FINAL_RUN_RUNBOOK.md`. These were found and left open
on purpose, each with the reason.

1. **Champion / Hero ability training: IMPLEMENTED 2026-09-16, with two gaps.**
   `rl/abilities.py`, sampled and scored in the joint action, phase-2 opponent and
   a heuristic mirror teacher included. What is still missing:
   (a) **readiness is not in the OBSERVATION** -- it rides in `info` and is used
   only as the action mask, so the policy cannot see "my ability is ready" and
   must infer it; putting it in the extra scalars is an engine change and would
   move `observation_size()`.
   (b) **`trainers/exploiter.py` has its own rollout loop and does not sample
   abilities** -- harmless while `EXPLOITER_ENABLED = False`, a silent asymmetry
   if it is ever turned on with a Champion deck. **Made LOUD 2026-09-23:**
   `run_exploiter_burst` now raises NotImplementedError for a deck with ability
   slots, before any file is written. Porting `rl/abilities.py` into its loop is
   still the real fix, needed only if the exploiter is re-enabled.

2. **Does the aux anti-alignment persist after the warm-up?** Measured only over
   the first 12 updates from init (LSTM 1.67x the other terms at cosine -0.84).
   Re-measure with `scratchpad`-style two-pass differencing at ~2k, 10k and 30k
   episodes of the real run. If it persists, the candidates are a smaller
   coefficient or a PCGrad-style projection of the aux gradient on the shared
   modules (costs a second backward per minibatch -- price it first).

3. ~~**The Fireball-keyed shaping terms**~~ **DONE 2026-09-23.** Both spell terms
   follow `card_probes.damage_spell(deck)` -- the deck's finishing spell, ranked
   by MEASURED Crown Tower damage -- published by both envs as `spell_*` keys
   through one builder (`gym_wrapper.deck_spell_info`). The 2.6 deck's reward is
   bit-identical (12 seeded matches, 0 of 2,280 steps differ); a Rocket deck's
   terms go from exactly zero to live. The same pass fixed the probe cutting
   damage-over-time spells off halfway (Poison 368 -> 736) and the teacher aiming
   every spell with Fireball's disc (Rocket +38% value killed). See `CLAUDE.md`,
   "2026-09-23".

4. **The tower PBRS term is not policy-invariant.** It telescopes exactly, but
   Phi(terminal) is never zeroed (-0.465 per episode at init), so it carries an
   implicit terminal tower-HP bonus. Aligned with TimeoutRules' tiebreak and
   probably benign; documented as invariant, which it is not. Decide, then fix
   either the code or the documentation.
   **Documentation fixed 2026-09-23; the CODE decision is still yours.** Measured
   on 16 seeded rung-3 mirror matches: tower residue +0.107 on wins / -0.169 on
   losses; the lethal and solvency potentials leave residues too (2 of 16
   matches, up to +0.125 / -0.019) -- the lethal docstring had claimed "EXACTLY
   ZERO". `weights.py`, `shaping.py` and `CLAUDE.md` now say what is true. The
   code option is one line: `(1 - done)` on the `gamma*Phi(s')` half.

5. ~~**The teacher has no air-defence concept**~~ **DONE for the rung-0
   rules gate, 2026-09-23** (rung 0 is the only rules-only rung).
   `card_probes.damages_air` (behavioural: does the card
   hurt a held Balloon) and `tactics.air_siege_map` (flying BUILDING-targeters --
   the case only anti-air answers; Minions/Baby Dragon are deliberately not in it,
   a ground unit still distracts them). With one on our half, anti-air cards
   outrank everything and ground-only cards play only against a ground threat;
   anti-air cards are aimed at the air threat. 36 seeds paired, lone push, 40 s:
   Balloon 1481 -> 1131 HP lost (9 better / 2 worse / 25 tied), Lava Hound 877 ->
   807 (17 / 7 / 12), pooled sign test p ~ 0.006; Hog control bit-identical.
   **Still open:** rungs >= 2 rank by rollout, and a 2-4 s horizon cannot see a
   Balloon arrive -- rung 2 defends ANY lone push worse than rung 0 (Balloon 1587
   vs 1322, Hog 1294 vs 687 over 12 seeds). A property of short lookahead, not of
   air; noted rather than changed.
   Found on the way (C++, `UPSTREAM_REQUESTS.md` item 30): Goblin Gang's and
   Goblin Hut's Spear Goblins and the Rascal Girls cannot hit air -- their
   helpers lack `.withTargetsAir()`.

6. ~~**Spawner huts measure as tower threats**~~ **DONE 2026-09-23**, and it was
   live: a Splashyard control deck (Graveyard, Poison, Baby Dragon, Bowler, Ice
   Wizard, Tornado, The Log, Tombstone) resolved TOMBSTONE as its win condition,
   and a Barbarian Hut outranked the Giant beside it. `teacher.siege_building`
   (fires at the tower itself, spawns no bodies, not deploy-anywhere) is now the
   one definition both the resolver and `card_probes.building_defends` read.
   Over 28 decks only those two resolutions changed; all 16 pool decks are
   identical. Found on the way: a Goblin Drill was measured and PLAYED from the
   own siege row, where it does 0 tower damage in 300 ticks (2654 beside the
   tower) -- it is deploy-anywhere and is handled as such now. Still open: the
   resolver compares buildings over a 1200-tick window against troops' 300, which
   no measured deck currently turns on.

7. **Engine stat inexactness, not fixed:** last-hit overkill is booked as tower
   damage (a Giant "deals" 3,795 to remove 3,546); `ElixirValueKilledCollector`
   credits zero elixir for a spawned body, so a Fireball clearing a Goblin Gang
   earns no value.

8. ~~**Phase 2's per-opponent PFSP win rates are not checkpointed**~~ **DONE
   2026-09-23**, with the phase-1 deck pool's merge-and-reseed pattern: workers
   now count games per opponent, the trainer stores a COUNT-WEIGHTED pooled
   estimate (a worker's untouched 0.5 is a prior and is not averaged in), and a
   resume seeds every worker with it. `refresh_pfsp_pool` only fills missing
   entries, so the seeded estimates survive it.

9. ~~**`CLASH_*` settings are not stamped in the checkpoint**~~ **DONE
   2026-09-23.** `checkpointing.clash_settings()` goes into every checkpoint;
   `restore_common` prints each differing setting on resume (operational ones --
   paths, cadence, workers, seed -- listed separately). A warning, not an error,
   so the runbook's resume path survives a deliberate change. Legacy checkpoints
   say they predate the stamp.

10. **Two curriculum clocks** (audit 04 C7): the patience constants count scenario
    episodes, the windows they gate do not, so every patience is ~30% shorter
    than the run it was calibrated on. Documented; not changed on a control loop
    already changed several times.

11. **Spells hit Crown Towers for 100% of their damage** (real game: 15-30%;
    Fireball 689 vs 159, Rocket 1485 vs 371, The Log 269 vs 41). Proposed as
    `perception/UPSTREAM_REQUESTS.md` item 29 with the exact edit -- C++, so it
    waits for a yes. A from-scratch run is the cheap moment. The Python side
    already follows whichever the engine does (`card_probes.spell_tower_damage`).

12. **Spawned Spear Goblins / Rascal Girls cannot hit air** --
    `perception/UPSTREAM_REQUESTS.md` item 30, proposed with the exact edit and a
    generic test. Two pool decks field Goblin Gang. (Night Witch was suspected
    too and CLEARED: her bats hit air; the zero was a probe flaw, recorded there.)

13. **Rung 1 is probably WEAKER than rung 0, and rungs 3 and 9 add nothing.**
    Measured 2026-09-23, 2.6 mirror, seat-swapped teacher-vs-teacher matches:

    | rung | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
    |---|---|---|---|---|---|---|---|---|---|---|
    | vs rung 0 (24 each) | 0.38 | 0.58 | 0.79 | 0.79 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
    | vs rung r-1 (32 each) | 0.38 | 0.78 | 0.47 | 0.59 | 0.69 | 0.88 | 0.56 | 0.66 | 0.50 | 0.72 |

    Rung 1 scored 0.375 against rung 0 in BOTH independent measurements (21 of 56
    pooled, ~[0.25, 0.50]) -- suggestive, at the edge of significance. The likely
    mechanism is the one item 5 found: rung 0 is the only RULES-only rung (the
    gate defends reflexively), and rung 1 swaps that for a 1 s rollout that
    cannot see a push arrive (a lone Hog costs rung 2 1294 tower HP vs rung 0's
    687). Not harmful to learning -- an agent past rung 0 clears rung 1 fast --
    but it is a rung of negative difficulty and two rungs of none. Options, NOT
    applied (the curriculum has been changed several times already): keep the
    rules gate for DEFENCE below some horizon and let the rollout rank offence
    only; or drop rungs 1/3/9. Measure a candidate with
    `scratchpad`-style rung-vs-rung matches before and after.

---

## 0c. ~~The overflow test lost its regime to the match-end rules~~ DONE 2026-09-23

**Closed by construction, as proposed below -- and it was worse than a skip.**
Measured on untouched main, 20 runs of the test's own sampler: 11 skip, 6 pass,
**3 FAIL**. Its flag marked an episode contaminated the moment the bar TOUCHED
10.0, which discards nothing unless the bar stays there (failing runs: tainted
MAE 0.007 vs clean 0.0078), and a fit on the tainted subset alone can match a
bar pinned at 10 with a constant anyway. The test now builds two boards with
IDENTICAL observed scalars (tick, both spends) whose opponent differs only in
how long it sat on a full bar: elixir 7.00 vs 0.45, gap 6.55 = 187 ticks at the
cap x 0.035, deterministic. No setters were needed -- real plays, so nothing is
confounded by an external write. (Found on the way: a hand slot is locked for
20 ticks after it cycles, `PlayerState::handCooldownTicks`.)

The original entry, kept for the reasoning:

`test_aux_task_is_not_a_memory_probe.py::test_overflow_is_what_makes_this_task_non_trivial`
**skips** as of 2026-09-06 instead of running.

Its sampler plays ordinary matches and needs some of them to reach the elixir
cap. The match-end rules added the same day end a match at 3:00 whenever the
crowns differ, and TRIPLE elixir begins at exactly 3:00 -- so the opponent now
reaches triple elixir only in OVERTIME, and overflow went from "roughly half of
episodes" (measured 2026-09-02, old rules) to none in the sample.

**The finding is not refuted and the skip must not be read as one.** A cap still
discards elixir no scalar records, so `Aux/OppElixir_MAE` would still be an
overflow detector rather than a memory diagnostic. What is gone is this
sampler's ability to REACH that regime by playing matches.

**The fix is to construct the state rather than fish for it.** `set_elixir_for_team`
and `set_current_tick` (UPSTREAM item 22) can put the opponent at the cap
directly, which tests the actual claim -- reconstruction fails once the cap has
discarded income -- without depending on how long a match happens to run, and
makes the test deterministic into the bargain. Roughly an hour's work.

**Do not "fix" it by lengthening matches or reverting the rules.** The new
behaviour is the real game's, and overflow becoming rare is a genuine and
desirable consequence of it.

---

## 0d. PLATEAU_IMPROVEMENT is calibrated faster than the agent learns

**Observed 2026-09-06, not fixed, and deliberately so.**

The plateau valve refreshes its clock only when the progress signal gains
`PLATEAU_IMPROVEMENT` (0.02) over the best seen at that rung. Measured on the
live run at rung 3, the agent improved steadily and was promoted anyway:

    ep 105,600  0.574   worst 0.07
    ep 107,800  0.624   worst 0.15   <- last refresh of best_rung_episode
    ep 109,200  0.637   worst 0.17   <- +0.013 since, so no refresh
    ep ~109,300 PLATEAU -> rung 4

+0.013 per 1,500 episodes is real learning -- the worst deck more than doubled
across that window -- but it is under the 0.02 the valve wants, so the clock ran
out on an improving agent. The threshold is a RATE and it is set faster than
this agent's actual rate.

**WHY IT IS NOT BEING CHANGED.** Lowering it invites noise to refresh the clock
forever, which is the stall the valve exists to end; raising the patience only
slows it. Both are guesses at a constant, and this file already records three
changes to this control loop in one session, each of which exposed the next
problem. A fourth under the same time pressure is how a control loop gets worse.

**And it is no longer costly, which is the actual argument.** The regression
detector added the same day means a premature promotion self-corrects: if the
new rung makes the agent worse it is demoted on measured evidence rather than on
a constant. The loop is closed even when the threshold is wrong.

**If it is to be fixed, fix the SHAPE, not the number.** Fit a slope over the
rung's progress history and advance when the slope is indistinguishable from
zero, which needs no rate constant at all. Measure it against both real series
already recorded here: the rung-3 plateau (0.641/0.640/0.641/0.640/0.637/0.639)
must advance, and this rung-3 climb (0.574 -> 0.637) must not.

---

## 0e. Search: an opponent model halves the damage and does not rescue it

**Measured 2026-09-06 on ep-111k, `eval/search_vs_greedy_pool_ab.py`.**

### The rollout opponent was the C++ heuristic, not nobody

`SearchCfg`'s docstring claimed "a candidate rollout assumes BOTH SIDES NO-OP"
and that sentence is WRONG -- it cost a whole wrong diagnosis before anyone
looked at the board. `sim.step` runs the C++ HeuristicOpponent, so only OUR side
no-ops. Verified: a 400-tick rollout with our side idle put 2 enemy bodies out
and took 1302 of our tower hp. The docstring is corrected.

So the real mismatch was HEURISTIC vs TEACHER: search optimised against the
opponent every historical result was measured against, while phase 1's opponent
had become the UtilityTeacher.

### Giving the rollout the right opponent helps, and is not enough

`search.rollout(sim, card, x, y, horizon, opponent=None)` takes any object with
`act(env, obs_own)`, which is `UtilityTeacher`'s own interface. Frontier decks,
horizon 4, widening off, n=30, **greedy 0.467 in BOTH arms** -- the control that
makes the pairing trustworthy:

| rollout opponent | search | delta |
|---|---|---|
| C++ HeuristicOpponent | 0.300 | -0.167 [-0.400, +0.067] |
| UtilityTeacher rules-only | 0.400 | **-0.067** [-0.233, +0.100] |

Harmful -> break-even. Search still beats greedy on NO deck: three ties, two
losses. **Expert iteration stays closed** -- there is no expert better than the
student, and CLAUDE.md records that distilling a weak one degrades selectivity.

### The model helps UNIFORMLY, and the horizon problem is separate

Read at n=24 this looked "unchanged at horizon 12" and that was premature. The
completed runs show the model worth about the same at both horizons:

| horizon | heuristic rollout | teacher rollout | gain |
|---|---|---|---|
| 4 | -0.167 | -0.067 | +0.100 |
| 12 | -0.469 | **-0.333** [-0.600, -0.067], p=0.041 | +0.136 |

So the opponent model is worth roughly **+0.12 win rate to search, at any
horizon** -- a real and consistent effect. It does not rescue search because
search starts further behind than that: -0.167 at h4 becomes break-even, -0.469
at h12 stays clearly negative.

**The horizon degradation is therefore a SEPARATE problem and survives the fix.**
Something makes a 12-step rollout worse than a 4-step one by ~0.27 even with the
right opponent, and it is not the opponent.

### The live hypothesis, not yet measured

**Our own side no-ops for the whole rollout.** Every candidate is scored as "I
play this, then stand still while a competent opponent answers", which is mildly
pessimistic over 4 s and grossly so over 12 -- fitting the shape exactly, since
the opponent model helped at h4 and did nothing at h12.

**The obstacle is structural, not a knob.** Scoring is ONE batched network
forward over all candidates' final observations. Making our side act needs a
forward per candidate per step -- 52 at K=13/h=4 against 1 today. Any attempt
should first check whether a CHEAP proxy for our own continuation (the advisor
in `advisors/tactics.py`, which needs no network) closes the gap, because the
full version may simply be unaffordable.

### shipping.py

`USE_SEARCH = False`; the deployable agent runs greedy. The case is now weaker
than when it was set -- search is break-even at h4 rather than harmful -- but
greedy is still >= search and costs ~1.5x less, so there is no reason to enable
it. Pinned by `test_shipping_does_not_use_search_until_it_is_re_validated`.

### The methodology note that cost two wrong conclusions

`PoolTeacherEnv` seeds the ENGINE but `UtilityTeacher` holds its own numpy RNG,
built unseeded by gym_wrapper; below rung 10 its epsilon is non-zero, so the two
arms faced opponents making different random choices and the pairing was only
partial. Caught by the greedy control -- which cannot be affected by search --
reading 0.750 in one run and 0.875 in another on identical seeds. On that bad
harness h4-without-widening read -0.031 and prompted the wrong calls that
"search is neutral at h4" and "widening is the culprit"; seeded, the same cell
reads -0.313.

**A paired harness needs a control that MUST be constant, and it has to be read
every run.** Every result above quotes its greedy control for that reason.

**Measured 2026-09-06 on ep-111k, `eval/search_vs_greedy_pool_ab.py`. This
supersedes the +0.319 that item 2 and `shipping.py` are built on.**

Paired, same seed and pool deck per trial, UtilityTeacher rung 3, widening off,
n=32 each. The greedy control reads **0.844 in all three**, which is the proof
the pairing is real:

| horizon | greedy | search | delta |
|---|---|---|---|
| 4 | 0.844 | 0.531 | **-0.313** [-0.531, -0.125] |
| 8 | 0.844 | 0.312 | **-0.531** [-0.719, -0.313] |
| 12 | 0.844 | 0.375 | **-0.469** [-0.688, -0.250] |

Deviation rate 23-27%, so search really is choosing differently; this is not a
vacuous null. It is negative at every horizon, on the frontier decks as well as
the pool (-0.267), and with widening on or off (-0.433 vs -0.469 at h12, CIs
fully overlapping).

**THE CAUSE IS THE OPPONENT MODEL, NOT A BUG.** A candidate rollout assumes BOTH
SIDES NO-OP. That is a passable model of the C++ HeuristicOpponent, which the
+0.319 and the +0.4025 horizon sweep were both measured against, and a bad model
of the UtilityTeacher, which forward-simulates. CLAUDE.md attached exactly this
caveat to the original result -- "says nothing about neural opponents" -- and
this is that caveat coming due. Nothing regressed; the regime changed.

**TWO HYPOTHESES TESTED AND REFUTED**, recorded so they are not re-run:
* *Widening dilutes the candidate set.* At h12, off vs on is -0.469 vs -0.433.
  It costs something at h4 but is not the driver. (`candidates/dec` 2.5 vs 21.9.)
* *The 2026-09-06 match rules broke terminal scoring.* `terminal_weight` 1.0 vs
  10.0 is -0.531 vs -0.469. Indistinguishable.

### What this means for the plan

**Do not run expert iteration.** There is no expert: distilling a policy from
something no better than itself teaches nothing, and CLAUDE.md records that a
weak expert actively DEGRADES selectivity (3.07 -> 2.31 -> 2.17 as data grew).

**`shipping.py` is affected and this is the urgent half.** It ships horizon 12
search, whose measured cost on its own configuration is **-0.433** [-0.633,
-0.233], p=0.00098. Its sweep (0.563 -> 0.963) was measured against the C++
heuristic and has never been re-run against the teacher. The deployable agent
should run its policy greedy until search is re-validated.

**If search is to be rescued, the lever is the ROLLOUT, not the horizon.** Give
the rollout an opponent model -- the cheapest being the UtilityTeacher's own
rules, which are the same code the opponent uses. `teacher.py`'s docstring
already anticipates this: "If the rollout ever gains an opponent model, this is
the candidate that starts paying."

### A methodology note that cost two wrong conclusions here

The first three runs used an UNSEEDED teacher. `PoolTeacherEnv` seeds the engine
but `UtilityTeacher` holds its own numpy RNG, and below rung 10 its epsilon is
non-zero -- 0.12 at rung 3 -- so the two arms faced opponents making different
random choices and the pairing was only partial. It was caught by the greedy
control, which cannot be affected by search, reading 0.750 in one run and 0.875
in another on identical seeds. On that bad harness h4-without-widening read
-0.031 and looked neutral; seeded, the same cell reads -0.313.

**A paired harness needs a control that MUST be constant, and it needs to be
read every run.** Both wrong conclusions here -- "search is neutral at h4" and
"widening is the culprit" -- came from not having looked at it.

---

## 0b. ✅ FIXED (2026-09-06) — the plateau valve was a timer, not a detector

**Measured, then triggered, then fixed and confirmed, all on the live phase-9
run. Commit `d7ab2e2`.**

`curriculum.py`'s plateau valve advances a rung when the 500-episode win rate
has not improved by `PLATEAU_IMPROVEMENT` (0.02) for
`PLATEAU_PATIENCE_EPISODES` (1500) while staying above `PLATEAU_MIN_WIN_RATE`.
Its own docstring records that **PFSP regulates the readable win rate toward the
hard end of the pool** — and then draws that conclusion only for the level GATE
("a level gate can be structurally unreachable"). The same regulation makes the
plateau's IMPROVEMENT test insensitive, which the docstring does not say.

Measured over ep 83,128 -> 87,540, deck pool on, floor off:

| | ep 83,200 | ep 87,400 |
|---|---|---|
| unweighted per-deck mean | **0.301** | **0.534** |
| worst deck | 0.01 | 0.11 |
| decks below 0.20 | 10 | 2 |
| PFSP-weighted readable win rate | ~0.50 | ~0.50 (flat) |

**The agent improved by 23 points on every one of sixteen decks and the readable
signal did not move, so the valve fired TWICE during the fastest learning of the
run** — rung 2 -> 3 at ep 85,340 and 3 -> 4 at ep 87,540, ~2,200 episodes apart,
which is the establishing window plus the patience. Under PFSP that condition is
satisfied by construction, so the valve is a **timer on a ~2,200-episode period**
rather than a convergence detector. Extrapolated, it walks rung 2 -> 10 in
~18,000 episodes regardless of what the agent learns.

**THE FIX IS THE SIGNAL, NOT THE PATIENCE.** Raising
`PLATEAU_PATIENCE_EPISODES` only slows the timer; it stays blind. The detector
should test improvement on a quantity PFSP does not regulate — the **unweighted
mean of the per-deck estimates**, which moved 0.301 -> 0.534 over exactly the
window the valve called flat. Keep the `PLATEAU_MIN_WIN_RATE` floor on the
readable rate (it is a competitiveness floor and is correct as it stands); change
only what the improvement test reads. Additive, with the current behaviour as the
fallback when no deck estimates exist, so the mirror path is untouched.

**THE TRIGGER FIRED, HARDER THAN ITS OWN THRESHOLD.** It was stated in advance
as "fails to improve by >= 0.02 over the 2,000 episodes after an advance". What
actually happened after the rung 3 -> 4 promotion at ep 87,540 is that the mean
FELL, inside 600 episodes:

| ep | unweighted mean | worst deck | |
|---|---|---|---|
| 87,600 | 0.535 | 0.11 | rung 4 |
| 88,200 | 0.522 | 0.09 | rung 4 |
| 88,600 | 0.515 | 0.08 | demoted to rung 3 here |
| 89,200 | 0.539 | 0.10 | |
| 90,200 | **0.563** | **0.11** | |

The readable win rate collapsed 0.50 -> 0.19 over the same span. **Neither
existing valve would have ended it**: the stall valve needs <= 0.10 and the
backstop needs 4,000 episodes, so the run would have sat there ~4.7 hours.

**THE FIX, AND THE CONFIRMATION.** `CurriculumManager.note_progress` takes an
improvement signal the caller owns; phase 1 passes the unweighted per-deck mean.
The floor test still reads the readable rate — that one is a competitiveness
question and is right as it stands. Demoting to rung 3 reversed the decline
immediately and learning resumed at **+0.030/1,000 episodes**, against rung 3's
own earlier +0.035/1,000 — so the rung, not the policy, was the cause.

**What is NOT established:** whether the ladder can now reach the top rungs at
all. The valve no longer advances on a flat readable rate, so from here a rung
ends only on the 0.65 gate or on the progress signal genuinely flattening. If a
run later sits at one rung for many thousands of episodes with the progress
signal still creeping, that is this change's failure mode and the thing to
watch — it is the 2026-08-28 stall arriving by a third road.

---

## 0a. LAUNCH AND WATCH: the meta-deck pool + the eleven-rung ladder

**Built and tested 2026-09-03, not yet run at length.** Branch
`meta-deck-pool-and-finer-curriculum`. Everything below is the WATCH LIST for
the first real run, because the two changes are gameplay-affecting and every
curriculum gate is now calibrated against a different opponent.

What changed: the phase-1 opponent plays a pool of 16 real meta decks
(`opponents/decks/meta_decks.json`) sampled per episode by PFSP weight instead
of our own deck; the teacher ladder went 6 rungs -> 11 with one knob per rung;
the advance gate went 0.80 -> 0.65; and two new exits (plateau, backstop)
guarantee no rung can hold a run that has stopped improving.

**Watch, in priority order:**

1. **`Decks/WinRate_Min` and the `Decks(win):` console row.** The pool's whole
   risk is a matchup the agent never gets off the floor on. The MINIMUM is the
   number that shows it; the pool average cannot, for the same reason the
   placement head's aggregate entropy could not see a per-card collapse.
2. **P(play | in hand) for Cannon / The Log / Fireball**, via
   `eval/probe_card_usage.py`. **This is the prediction the whole deck-pool
   change rests on and it is NOT yet demonstrated.** The 2026-09-03 sweep shows
   the pool offers 1.5-2.9x the opportunity the mirror did; whether a policy
   retrained on it actually picks those cards up is unmeasured. Baseline to beat
   is the mirror-era 0.0053 / 0.0091 / 0.0011.
3. **How many rungs are cleared by `[PLATEAU]` rather than by the gate.** A run
   that plateaued up every rung is at the top having beaten nothing, which is a
   materially weaker claim than the stage number suggests. Both are printed and
   `Training/Curriculum_PlateauAdvances` logs the count.
4. **Time per rung.** The failure being fixed was 23,040 episodes at one rung;
   the backstop caps it near 4,000 by construction, so anything above that means
   a valve is not firing and the guarantee is broken, not merely slow.

**RESUME, do not restart.** Measured on the 16-deck pool, `model_weights_phase5.pth`
(ep 32,484) averages 0.527 unweighted, clears 0.40 against 11 of 16 decks, and
scores a PFSP-weighted **0.422** -- just above `PLATEAU_MIN_WIN_RATE`, so every
valve is live from episode 1. A FRESH net by contrast won 0 of its first 100
episodes against the pool (see CLAUDE.md, "THE COLD START IS REAL"). Nothing in
the observation or action space moved, so the checkpoint loads unchanged.

**Do NOT compare any win rate across this change**, in either direction. A
mirror win rate and a PFSP-weighted pool win rate are different quantities --
PFSP deliberately regulates the second toward the agent's worst matchups, which
is why `PLATEAU_MIN_WIN_RATE` is 0.40 and is not a "the agent got worse" signal.

**If the deck axis turns out to be too much at once**, the cheap fallback is
`CLASH_PHASE1_DECK_POOL=0` (restores the mirror exactly) or trimming the pool
via `enabled: false` in the JSON -- neither needs a code change or invalidates
a checkpoint.

---

## 0. NEXT UP — Stage 2: the live teacher-as-agent loop

**The engine half landed 2026-08-24** (`UPSTREAM_REQUESTS.md` item 22,
APPLIED). **Only Python remains, and all of it is in `perception/`, which is
freely editable.**

Design: `docs/design/specs/2026-08-24-live-teacher-play-design.md`.
The measured deploy-time result behind it: `DECISIONS.md`, "2026-08-24: the
live-mirror state setters". Stage 1 (a passive gap-logger) was **deliberately
skipped** by the human once item 22 was approved.

**Goal:** `UtilityTeacher` plays as US (team 0) in a real match. Perception
reads the screen → mirror env → teacher proposes AND ranks by 5-10 s rollouts
→ the best placement is tapped.

**Two new modules, neither of which exists yet:**

1. `perception/live/mirror.py` — `MirrorBuilder.build(snapshot) -> (env, MirrorGaps)`.
   `reset()` → `inject(..., hp=frac*full, deploy_ticks=0)` per unit →
   `set_elixir_for_team` both sides → `set_hand_for_team(0, hand)` **and CHECK
   ITS BOOL** → `set_tower_hp` / `destroy_tower` per tower → `set_current_tick`.
2. `perception/live/teacher_policy.py` — `TeacherPolicy.decide(gs, ready, now)
   -> Decision`, holding ONE persistent `UtilityTeacher(deck, team=0)` (it
   carries `self.pending` across decisions) while the ENV is rebuilt per
   decision. Wired as `--policy teacher` beside `ScriptedPolicy`/`NeuralPolicy`.
   **`--act` stays opt-in; dry run is the default.**

**Do not add threads.** The pipeline is already three (producer in
`live/pipeline.py`, decision loop at `DECISION_HZ = 1.0`, actuator with a
depth-1 drop queue). `src/bindings.cpp` releases the GIL **nowhere**, so a
rollout thread would BLOCK perception rather than run beside it, and a
concurrent writer would score candidates against different worlds. Budget is
not the issue: teacher **14.1 ms** + rebuild ~1 ms against a **1000 ms**
period. "Streaming" = rebuild the mirror from the latest Snapshot at each
decision — which also deletes `forecast.py`'s entity-removal gap for free.

**The three joints where this goes silently wrong:**

- **Ordering.** REUSE `forecast.py`'s `bodies_per_card` and its
  group-by-`(card_sim_id, team)`. It **resets the env itself**, so every body
  count must be resolved BEFORE the build's `reset()`. Getting it wrong gives
  3x the Skeletons — which reads as "the teacher panics", not as a bug.
- **Hand-slot alignment.** The teacher's `slot` indexes the mirror's hand; the
  actuator taps the real one. Same number ONLY if `set_hand_for_team` returned
  True. On refusal **drop the decision** — a placement against a misaligned
  hand plays a card nobody chose.
- **Cadence.** `pending_ticks -= COMBO_FOLLOWUP_DELAY_TICKS` (10) runs **once
  per `act()` call**, so the teacher assumes exactly 1 decision/second. A
  DROPPED decision advances the plan 1 s in teacher-time while 2 s of wall time
  passed, skewing every combo gap. Log decision-interval drift. Any fix belongs
  in a proposal — `python_ai/` is read-only.

Bias throughout: **no-op over a wrong action.** A mistimed card is worse than a
skipped decision.

**Known limits, already measured — do not rediscover.** Unit-HP **recall
0.34-0.56** at precision 0.98, so ~half of damaged units still arrive at full
health. The opponent's hand is unobservable, so team 1's stays fabricated. The
engine models **no double elixir** (`ELIXIR_REGEN_RATE` is constant), so late
rollouts stay mispriced even with a correct clock — filed as its own item, NOT
folded into 22.

---

## 1. ✅ DONE (2026-08-20/21) — Utility Teacher evaluates MULTI-CARD COMBO placements, and can now afford them

**Built, tested and measured.** `UtilityTeacher` candidates are now SEQUENCES of
placements rather than single cells, and the teacher plans, commits to and
executes two-card combos. Full write-up with every number is in
`DECISIONS.md`, "2026-08-20 (later): the MULTI-CARD teacher".

The short version:

- `Candidate` carries a tuple of `PlacementStep`s; `.slot/.x/.y` still mean the
  first step, so nothing downstream changed. Six curated families
  (`supported_push`, `counter_push`, `defensive_stack`, `cheap_defence`,
  `spell_then_push`, `push_then_spell`), round-robin under a `max_combos`
  budget that is a new competence axis in `TEACHER_STAGES` (0/0/2/3/4/4).
- **No C++ change was needed.** A 0-tick `step_self_play` places nothing and
  `gym_wrapper` gives the teacher one action per decision, so a combo is a
  sequence across consecutive decisions -- which is the shape the +448.5 tower
  HP "supported push" was measured at anyway. Per-entity deploy time verified
  from Python.
- **The generator alone was not enough**, and that is the transferable result: a
  pair priced at 5-6 elixir was affordable in 2 of ~2,400 decisions. What fixed
  it was making the follow-up GAP a searched axis (1/3/5 s), because the pair's
  cost is paid across the gap. A flat savings charge was tried first and is
  measured DEAD -- do not re-propose it.
- **Win rate: SUPERSEDED. Pooled over five paired runs (200 openings) the combo
  machinery costs about 4 points against a mirror** -- deltas +0.000/-0.031/
  -0.081/-0.031/-0.056, pooled -0.0398, CI [-0.076, -0.004]. It was called a
  null at n=40 and that was a power limit, not a result. Kept on as a
  REPERTOIRE choice (the engine rewards escorted pushes; the teacher still beats
  the C++ heuristic 1.000 and the old teacher 95-5), with `max_combos = 0` and
  `combo_families` as one-line off switches. **The lever is completion: 60% of
  chosen combos leave a first card down for a plan that never finishes.**
- **A control failed and the reason is reusable:** `--seed` did not make
  `prove_combos.py` reproducible, because `ClashEnv::reset()`'s opening shuffle
  was unseeded. Within a run the snapshot pairing is sound; ACROSS runs only the
  deltas were comparable, never the arm levels. **The engine half was fixed on
  2026-08-21** — `ClashEnv::seed()` seeds both generators and re-deals — **but
  `prove_combos.py` still never calls it**, so the caveat holds for the harness
  exactly as it stands today. That migration is item 8 below.

### The economy follow-up: DONE 2026-08-21

`play_margin` 0.05 -> 3.0, plus an overflow taper and a follow-up exemption.
Combo share of plays 2.2% -> 14.4%, the old teacher loses ~94% head to head, and
stage 5 still scores 1.000 against the C++ heuristic. Full write-up in
`DECISIONS.md`, "2026-08-21: the teacher's ECONOMY".

**`w_pos` was the obvious lever and is MEASURED WRONG -- do not re-propose it.**
Swept 20 -> 8 it raises elixir (1.83 -> 2.67) and drives combo share to
1.1% -> 0.0/0.0/0.2/0.0/0.2%. It prunes plays by HP-per-elixir, and an escorted
push (388 HP/elixir) sits below a naked Hog (424), so it kills the combo before
the cheap cards it was meant to replace.

**What is still open, and it is narrow.** The combos that actually complete are
DEFENSIVE (`cheap_defence`, `spell_then_push`, `defensive_stack`); the escorted
win-condition push is 3 of 47. That is a hand-co-occurrence and price problem,
not a scoring one -- the tank and the win condition are both in hand on ~4-9% of
decisions and the pair is the deck's most expensive. Worth knowing before
anyone reads "14.4% of plays are combos" as "the Ice Golem + Hog push is now
standard".

---

## 2. Wire decision-time search into the live loop

**The largest measured gap between the benchmark and real play**, and plausibly
small.

Search is **not** in `perception/live/mvp_loop.py` — verified 2026-08-19: the
neural path builds the observation through `perception_encoder` and calls the
policy head directly. There is no candidate rollout and no engine snapshot in
that path.

So the configuration measured at **0.9225** cannot be run live. Live play gets
the net **greedy**, which measured **0.5200** on the same opponent.

The loop already owns a simulator mirror — `perception/forecast.py` holds a real
`ClashRoyaleEnv` and calls `get_observation_for_team(0)` — which is exactly the
object `_search_action` needs to snapshot.

**Caveat:** both those numbers were measured on the retired Giant-deck net. The
*mechanism* is what transfers; re-measure the magnitude before quoting it.

**Files:** `perception/live/mvp_loop.py`, `perception/forecast.py`,
`python_ai/search/realtime_search.py`, `python_ai/shipping.py`.

---

## 3. Human-replay imitation — extraction is the blocker

The recordings exist (8 matches in `perception/assets/recordings/`). The
`bc_pretrain` `.npz` pipeline is built and verified end to end. The missing step
is `perception/` → that schema.

Self-play discovers strategies but not *the distribution humans play*;
AlphaStar's supervised stage was load-bearing, not optional.

**Blocked by:** the recordings use the Giant deck (Valkyrie, Archers, Minions,
Cannon, Fireball, Giant, Musketeer, Mini P.E.K.K.A) and `DEFAULT_DECK` is now
2.6 Hog Cycle. This tie was knowingly given up on 2026-08-16. Either record new
matches on 2.6, or accept cross-deck transfer as a separate question.

**Files:** `python_ai/trainers/bc_pretrain.py` (`DATASET_SCHEMA`), `perception/`.

---

## 4. Remaining observation gaps

Both are real, both are cheap to state and not cheap to fix.

- **Cells overwrite rather than accumulate** in channels 0–7
  (`obs[idx] = normalizedHp`), so a Skeleton Army collapses to one body.
  `CH_COUNT` mitigates but does not fix it.
- **No card-cycle tracking** — verified 2026-08-19: `model.py` has no cycle
  channel at all, while `teacher.py` carries a working `CycleTracker`. This is
  the single most deck-specific gap, because **2.6 is *defined* by cycling back
  to the Hog faster than the opponent cycles their answer**, and the net can see
  only the 4 cards currently in hand.

Both are engine-side (`extractObservationForTeam`) → propose in
`UPSTREAM_REQUESTS.md` first. Changing `NUM_CHANNELS` **invalidates every
checkpoint**.

---

## 5. `skip_frames = 10` — one decision per second

A hard ceiling on tactical precision. Pulling a Hog with a Cannon and timing an
Ice Spirit are sub-second decisions, so this binds harder for 2.6 Hog Cycle than
it did for Giant beatdown.

Left alone so far because changing it is an **unmeasured throughput/precision
trade**, not because it is fine. The 2026-08-07 movement-speed fix shrank the
damage by ~5× for free (one second now covers ~5× less board).

---

## 6. One deck, mirror matchups

"A great player" implies arbitrary matchups. Phase 1's `random_opponent` and the
scripted bots' randomised decks are partial; phase 2's neural opponents all play
`DEFAULT_DECK`.

---

## 7. Shaping is a hand-designed proxy and caps the ceiling

Non-PBRS terms (`W_TOWER_DESTROYED`, `W_FLAWLESS_DEFENSE`, the spell-value term)
bias the optimum by construction — deliberately, and each is documented with
why. Troop *damage* is also the wrong metric: Clash is decided by kills and
elixir advantage, not HP chipped. Long term these should anneal toward zero.

---

## 8. ✅ DONE (2026-08-24) — engine seeding is applied, VERIFIED, and the last unseedable path is closed

Engine request 7 landed 2026-08-21 (`ClashEnv::seed`). Two Python consumers were
migrated on 2026-08-24, a third path needed C++, and **the verification that was
the whole point had never been run** — the machine it was done on had no Python
3.11, no venv and no built `.pyd`, so nothing importing the engine executed there
at all. It was run on 2026-08-24 on a box that has all three. Both checks pass.

**Migrated consumers** (in the read-only tree, at explicit instruction):

- `python_ai/envs/gym_wrapper.py` — `reset(seed=...)` accepted a seed and
  dropped it. `super().reset(seed=seed)` seeds the *wrapper's* `np_random`,
  which this env reads nowhere. Now forwards to `self.game.seed(seed)`, guarded
  on `is not None` so a run does not collapse to one repeated episode.
- `python_ai/eval/prove_combos.py` — five harnesses built each opening with
  `CE(...).reset()` and never called `seed()`. All five now use
  `.seed(args.seed + ENGINE_SEED_OFFSET + i)`; `seed()` ends in `reset()`, so it
  is a drop-in.

**VERIFIED 2026-08-24 — both checks, with their controls:**

```
two envs, reset(seed=7):   observations identical        True
                           hand team 0 identical         [72, 7, 15, 40]
                           hand team 1 identical          [25, 7, 15, 33]
a third env, reset(seed=8): differs from seed 7           True   <- non-vacuous
```

`prove_combos --combo-ab --n 6 --seed 300`, run twice, is **bit-identical**:

| | run 1 | run 2 |
|---|---|---|
| combos OFF | dec 2099, elixir 3.39/p90 6.70, plays 282 | identical |
| combos ON | dec 2160, plays 314, proposed 824, chosen 58, completed 27 | identical |
| levels | off 0.667 → on 0.667 | identical |
| split | 2 better / 3 worse / 1 tied | identical |

Only `ms/dec` moved, which is wall-clock noise. **The OFF arms agree on LEVEL,
not merely on delta** — the acceptance criterion this item was written around.

**So the old rule is LIFTED.** Across runs at a fixed seed, arm LEVELS are now
comparable, not only deltas. That rule existed because the 2026-08-20 combo
ablation's untreated arm moved between runs; it cannot now.

**The third path is closed too — `sample_random_deck(seed=...)` (was UPSTREAM
item 23C, applied 2026-08-24).** It drew from a function-local
`static std::mt19937` that `ClashEnv::seed` could not reach, so a run with
`randomize_opp_deck=True` had a reproducible hand, cycle and heuristic roll and
a still-random opponent deck — the largest of the four variance sources. A
seeded call now builds a PRIVATE generator instead of seeding the static,
because the static is process-global and at `num_envs = 8` every env interleaves
draws from it: seeding it would only reproduce for a fixed construction and call
order. Zero-argument calls are bit-for-bit unchanged.
`python_ai/tests/test_engine_seeding.py` pins reproducibility, a distinct-seeds
control, order-independence, and no-seed compatibility.

---

## Engine requests — the backlog is CLOSED (2026-08-24)

`perception/UPSTREAM_REQUESTS.md` and `perception/BOT_REQUESTS.md` were worked
to empty on 2026-08-24 and are now 0-byte files. All 25 upstream items and all
8 bot items were verified applied, decided, or implemented. **Their full text is
archived in `DECISIONS.md` under "ARCHIVE"**, with headings kept intact so the
112 references that cite them by number still resolve — "UPSTREAM_REQUESTS.md
item 13" resolves to "UPSTREAM item 13" there.

Do not re-open a numbered item without reading its archived entry first: several
were re-proposed in the past after their status line went stale, which is the
specific failure that audit note existed to stop.

### What was implemented during the sweep

- **UPSTREAM item 17 — `CombatEntity::getTicksOnTarget()`.** One const accessor,
  resolved minimally: of the eleven the item tabled, only `ticksOnTarget` has a
  genuinely lossy behavioural proxy (`getDamagePerTick()` collapses it into ≤4
  ramp buckets, and `rangeFalloff` makes the stage inseparable). Everything else
  stays behavioural on purpose. `tests/core/test_snapshot_timing_state.cpp`.
  Additive and const — NOT gameplay-affecting.
- **UPSTREAM item 20 — the replay now writes each entity's `name`.** The item's
  own diagnosis ("spawned entities carry no cardId") was FALSE: spawn-children
  carry distinct negative ids (−1…−48) and already carried the right name; the
  logger simply never wrote it. Fixed in `GameLogger.h` plus a viewer branch,
  which is far smaller than the `spawnedByCardId` the item proposed and actually
  names the child rather than its parent. `cardId` untouched, so stats
  attribution and reward shaping are bit-identical.
  `tests/core/test_game_logger_entity_names.cpp`.
- **UPSTREAM item 23C — `sample_random_deck(seed=...)`.** The third `mt19937`,
  which `ClashEnv::seed` cannot reach. A seeded call now gets a PRIVATE
  generator rather than seeding the process-global static, because at
  `num_envs = 8` a shared stream is only reproducible for a fixed call order.
  Zero-argument calls are bit-for-bit unchanged.
- **`tools/audit/verify_pyd.py` — a false negative that could never pass.** It
  checked `hasattr(cre, "step_self_play_fast")` against the MODULE when that is
  a method on the `ClashRoyaleEnv` CLASS, so the post-build gate CLAUDE.md tells
  you to run before every training run had reported FAILED on healthy builds
  since 2026-08-23. The worst failure mode a gate has: it teaches you to ignore
  it.

### The live residue — this is the part that is still WORK

- **Channels 0-7 assign rather than accumulate** (was BOT item 3). Agreed by
  both sides, one line in `ClashEnv.h:267`:

  ```cpp
  obs[getIndex(channel, y, x)] = normalizedHp;                       // now
  obs[getIndex(channel, y, x)] = std::max(obs[...], normalizedHp);   // proposed
  ```

  Makes channels 0-7 mean "the strongest unit in this cell" and consistent with
  `CH_DPS`/`CH_RANGE`/`CH_SPEED`, which already take the max. Summing would be
  wrong — three Skeletons would read like a PEKKA. No layout or size change, so
  checkpoints still LOAD, but observation SEMANTICS shift, so it is
  **deliberately queued for a clean restart / the next intentional observation
  change** (decision reaffirmed 2026-08-24). Also CLAUDE.md open problem 3.
- **`live/adapter.py` still reads tower HP from the BAR** (was BOT item 7).
  `readers/tower_numerals.py` is built, tested and template-backed — acceptance
  2030 on both enemy Princesses, 542-step read-back at 97.8% — but the live
  adapter is not wired to it. Perception-side work, freely editable.
- **Perception-shaped observation noise stays DEFERRED** (was BOT item 1),
  behind the sensor's own error: ~15% of detected units are misnamed (`knight`
  worst; one `minipekka` was the enemy Princess tower) and card identity agrees
  with the elixir ledger only 33.8% of the time. Training a policy to tolerate
  that would cost real capability and buy nothing once the sensor improves.
- **UPSTREAM item 18 stays ACCEPTED / will-not-fix** — troops deadlock in the
  concave pocket between two buildings, 7 stalls per 342,563 unit-ticks, none on
  a bridge, pinned `[!shouldfail]` in `tests/core/test_navigation_wedge.cpp`.
  Reopen if the rate rises, a stall appears near a bridge, or rollout throughput
  stops being the binding constraint on path planning.
- **UPSTREAM item 8 stays "change nothing"** — Fireball 689 vs Musketeer 721 is
  faithful to the real game, where Fireball needs chip damage on top to kill a
  Musketeer. Under the current 2.6 Hog Cycle deck the awkward EV is if anything
  sharper than the item described (its table lists the retired Giant deck): the
  only clean Fireball kills in the mirror are Skeletons and Ice Spirit, both
  1 elixir. Reach the behaviour through shaping, not by editing the card.

---

## Perception-side open work

From `perception/README.md` and `perception/BOT_REQUESTS.md`.

1. **Stage 3 — opponent placement detection is blocked on data.**
   `detect/placements.py` raises. Needs the next recording batch.
2. **Card identity is measurably wrong.** The icon template agrees with the
   elixir ledger on card cost only **33.8%** of the time (328 in-match plays)
   and over-predicts Giant at 35% against a 12.5% prior.
3. **Phase boundaries unknown.** `clock/match_clock.py` raises
   `PhaseScheduleUnknownError` until real single/double/triple/overtime
   boundaries are supplied.
4. **Card map review — 52 rows.** 41 Evolution pairings and 11 ids ≥ 165 need
   confirming against the build being recorded.
5. **The ledger/`unconfirmed` mismatch.** 5 of 28 post-fix placements still go
   unconfirmed. Remaining suspects: the ~1.6 s decide→land round trip, and the
   integer elixir reading. Measure with `tools/placement_truth.py`'s oracle
   rather than the ledger, since the ledger is the thing under suspicion.
   **Do not re-open "the game refuses our placements"** without new live
   evidence — measured at 0/28 across two matches.
6. **Train against perception-shaped observation noise** (`BOT_REQUESTS` item 1)
   — reasoned, not measured. Deferred: harness yes, training change not yet.

---

## Deliberately NOT on this list

Recorded so they are not re-proposed. Each was measured, not assumed.

- **Observation channels 0-7 max-instead-of-assign** (the queued "clean restart"
  change). Measured 2026-09-15 against true per-tick entity HP: the 2.6 deck hides
  0.8% of troop HP, a control deck 1.7%, a swarm deck 21.3% -- but a max rule would
  recover only ~2.3 points of that (78.7% -> 81.0% represented), because the loss
  is one value per cell, not the write order. Not worth changing every HP estimate
  the teacher and the advisor read. A real fix is a different encoding (a summed
  HP channel next to CH_COUNT), which is an observation change to price on its own.
- **A fifth Hog-specific policy mechanism.** Four returned null. The open
  question is the engine's offence/defence cost balance, and deploy time was the
  answer to it — item 1 is the follow-through, not a fifth mechanism.
- **Re-distilling search into a cured seed.** Refuted three ways on 2026-08-16
  (conditional lift −0.0032, selectivity 0.92, paired A/B p = 0.608). Search and
  the placement cure repair the same weakness and do not stack.
- **Scaling the distillation dataset.** Actively harmful — selectivity falls
  3.07 → 2.31 → 2.17 as coverage rises.
- **Re-adding heuristic exposure to the PFSP pool.** Already 20.8% realized, and
  it regressed anyway.
- **A from-scratch AlphaZero loop with the current search module.** Inert at
  init — search scores candidates with the net's own critic, so at random init
  the expert is not an expert (21.5% deviation, zero gain).
- **The soft/neighbourhood advisor target (A1).** Measured neutral-to-worse.
  Resolution was the binding constraint, not target softness.
- **Re-enabling the exploiter.** Off since 2026-08-11. Its documented re-enable
  precondition is an engine/shaping change, not a Python one.
- **Penalising back-row or low-shot structures.** `P(shots ≤ 3)` was **100.0%**,
  so the predicate has zero variance and no spatial gradient. *A penalty cannot
  move a distribution with no mass to move.*
- **Leading a spell target.** `lead=0` captures 75.5% of achievable value,
  `lead=10` only 52.4%. `tactics.py` defaults to 0; do not "fix" it.
