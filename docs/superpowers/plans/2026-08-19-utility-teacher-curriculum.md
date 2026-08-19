# Utility-Teacher Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the opponent-elixir-multiplier curriculum with a deterministic
utility-search teacher at a symmetric 1.0x economy, so the environment prices the
win condition correctly and Phase 1 has a real opponent instead of a handicapped
one.

**Architecture:** A hand-written bot (`teacher.py`) generates candidate placements
from the engine-validated rules in `tactics.py`, ranks them by rolling each one
forward on an `env.snapshot()` and reading the engine's own statistics, and plays
either side through `step_self_play`. Difficulty is dialed on the bot's lookahead
horizon, candidate width and epsilon -- never on elixir.

**Tech Stack:** Python 3.11 (`python_ai/venv/Scripts/python.exe` -- the `.pyd` is
3.11-only), numpy, torch 2.13.0+cpu, pybind11 bindings in `clash_royale_env`.

## Global Constraints

- **No C++ changes.** Everything is expressible through existing bindings. Any
  perceived need for one goes to `perception/UPSTREAM_REQUESTS.md` as a proposal,
  not into `include/` or `src/`.
- **Engine constants are read from the bindings, never re-typed.** Where a value
  is genuinely not bound, hardcode it with a comment naming the header, matching
  `tactics.py`'s existing pattern.
- **Python interpreter:** every command uses `python_ai/venv/Scripts/python.exe`.
  The default `python` on this box is 3.14 and fails with `ImportError: DLL load
  failed`, which reads like a corrupt build and is only a version mismatch.
- **Coordinate frames** (verified empirically 2026-08-19, not assumed):
  - `step_self_play(c0, x0, y0, c1, x1, y1)` -- team 0's `y0` is **absolute**;
    team 1's `y1` is in team 1's **own mirrored frame** and the engine converts
    it as `y_abs = (BOARD_HEIGHT - 1) - y1 = 33 - y1`.
  - `is_valid_placement(card_id, x, y, team)` takes **absolute** y for both teams.
  - Therefore a teacher on team 1 must convert `y_abs = 33 - y_own` before
    checking legality. Measured: own-frame `y=15.0` -> absolute `18.0` -> legal
    for team 1; the unmirrored `15.0` is illegal for team 1.
  - x is never mirrored.
- **Observation channels** (`ClashEnv.h`): 0-3 ally melee/ranged/building-targeter/
  building, 4-7 enemy same, 8 river/bridges, 9-20 attribute channels with ally at
  `CH_*` and enemy at `CH_* + 1`.
- **Training economy is pinned at 1.0x.** `set_opponent_elixir_multiplier` stays
  bound and callable for measurement harnesses; nothing in the training path may
  raise it.
- **No reward-function changes in this plan.** Changing shaping at the same time
  would make the smoke run uninterpretable.

---

## File Structure

| file | responsibility |
|---|---|
| `python_ai/teacher.py` (create) | `UtilityTeacher`, `card_roles`, `TEACHER_STAGES`, weight profiles. Side-agnostic; reads observations only. |
| `python_ai/scenario_offense.py` (create) | Offensive (punish-window / counter-push) scenario constructors. Default-off. |
| `python_ai/prove_environment.py` (create) | The zero-training falsifier: Teacher vs Teacher, win-condition banned in one arm, swept over multiplier. |
| `python_ai/prove_teacher.py` (create) | Teacher strength: vs C++ heuristic, vs shipping net + search. |
| `python_ai/gym_wrapper.py` (modify) | Optional teacher opponent; routes `step()` through `step_self_play`. |
| `python_ai/train.py` (modify) | `CURRICULUM_STAGES` -> competence stages at 1.0x; wire the teacher; optional offensive scenarios. |
| `python_ai/test_python_ai.py` (modify) | Regression tests for frames, legality, no-op baseline, the 1.0x pin. |
| `CLAUDE.md` (modify) | Record the pivot, its consequences, and the stale probe filenames. |

---

### Task 1: Card role table

**Files:**
- Create: `python_ai/teacher.py`
- Test: `python_ai/test_python_ai.py`

**Interfaces:**
- Produces: `card_roles(deck: list[int]) -> dict[int, str]` where the role is one
  of `"wincon" | "spell" | "building" | "ranged" | "melee"`.

Roles are **derived by injection**, the same technique
`gym_wrapper._find_win_condition` already uses: inject the card on an empty board,
step one tick, and read which of channels 0-3 lights up. `get_card_info` exposes
`is_spell`/`is_building` but no archetype, so a hardcoded id list would silently
mean "Hog Rider forever" and break on the next deck change.

`_find_win_condition` is **left untouched** -- it answers a narrower question, it
sits in the reward path, and changing it is unnecessary risk.

- [ ] **Step 1: Write the failing test**

```python
def test_card_roles_derives_roles_from_the_engine_not_a_hardcoded_list():
    from teacher import card_roles
    from gym_wrapper import DEFAULT_DECK
    roles = card_roles(DEFAULT_DECK)
    assert roles[15] == "wincon"     # Hog Rider, the only real building-targeter
    assert roles[7] == "spell"       # Fireball
    assert roles[33] == "spell"      # The Log
    assert roles[25] == "building"   # Cannon
    assert roles[6] == "ranged"      # Musketeer
    # Ice Golem also targets buildings but is a 2-cost shield, not a win
    # condition -- the highest-cost building-targeter wins, same tiebreak
    # gym_wrapper._find_win_condition uses.
    assert roles[40] != "wincon"
    assert set(roles) == set(DEFAULT_DECK)
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `python_ai/venv/Scripts/python.exe -m pytest python_ai/test_python_ai.py -k card_roles -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'teacher'`

- [ ] **Step 3: Implement `card_roles` in `teacher.py`**

One `ClashRoyaleEnv` per card, `inject(cid, 9.0, 8.0, 0)`, one tick, read
`obs[ch*plane:(ch+1)*plane].max()` for ch in 0..3. Building-targeters collected
and the highest-cost one promoted to `"wincon"`; the rest fall back to
`"melee"`/`"ranged"` by their lit channel. Spells and buildings short-circuit off
`get_card_info` without an injection.

- [ ] **Step 4: Run the test and confirm it passes**

- [ ] **Step 5: Commit** -- `feat(teacher): derive per-card roles from the engine`

---

### Task 2: Candidate generation (rules only, horizon 0)

**Files:**
- Modify: `python_ai/teacher.py`
- Test: `python_ai/test_python_ai.py`

**Interfaces:**
- Produces:
  - `class Candidate` with fields `slot: int`, `card_id: int`, `x: float`,
    `y: float` (own frame), `role: str`. The no-op is `slot == HAND_SIZE`.
  - `UtilityTeacher.__init__(self, deck, team, profile=None, horizon_ticks=30, k_cells=2, epsilon=0.0, seed=None)`
  - `UtilityTeacher.candidates(self, obs_own, hand, elixir) -> list[Candidate]`
  - `UtilityTeacher.reset(self, rng=None) -> None`
  - `UtilityTeacher.to_absolute_y(self, y_own) -> float` --
    `y_own` if `team == 0` else `33.0 - y_own`.

Cells per role, all from `tactics.py`: wincon -> both bridge cells at
`BRIDGE_ROW`; spell -> `best_spell_cell` plus the catch map's next-best cell;
building -> `best_building_cell` plus the centre-pull cell; other troops ->
defensive meet-forward cell on `threat_lane`, plus a support cell behind our own
most advanced unit. Every candidate is filtered through
`env.is_valid_placement(card_id, x, to_absolute_y(y), team)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_teacher_candidates_are_all_legal_for_either_team():
    import numpy as np, clash_royale_env as E
    from teacher import UtilityTeacher
    from gym_wrapper import DEFAULT_DECK
    CE = E.ClashRoyaleEnv
    for team in (0, 1):
        env = CE(DEFAULT_DECK, DEFAULT_DECK, 3600); env.reset()
        for _ in range(40):
            env.step_self_play(-1, 0, 0, -1, 0, 0, 10)
        t = UtilityTeacher(DEFAULT_DECK, team=team, horizon_ticks=0)
        t.reset()
        obs = np.asarray(env.get_observation_for_team(team), np.float32)
        cands = t.candidates(obs, env.get_hand_for_team(team),
                             env.get_elixir_for_team(team))
        assert len(cands) >= 1                       # the no-op is always there
        for c in cands:
            if c.slot == CE.HAND_SIZE:
                continue
            assert env.is_valid_placement(c.card_id, c.x,
                                          t.to_absolute_y(c.y), team), \
                f"team {team} proposed an illegal cell: {c}"


def test_teacher_never_proposes_an_unaffordable_card():
    import numpy as np, clash_royale_env as E
    from teacher import UtilityTeacher
    from gym_wrapper import DEFAULT_DECK
    CE = E.ClashRoyaleEnv
    env = CE(DEFAULT_DECK, DEFAULT_DECK, 3600); env.reset()
    t = UtilityTeacher(DEFAULT_DECK, team=0, horizon_ticks=0); t.reset()
    obs = np.asarray(env.get_observation_for_team(0), np.float32)
    hand = env.get_hand_for_team(0)
    cands = t.candidates(obs, hand, 1.0)          # only the 1-cost cards fit
    for c in cands:
        if c.slot == CE.HAND_SIZE:
            continue
        assert E.get_card_info(c.card_id)["cost"] <= 1.0
```

- [ ] **Step 2: Run and confirm both fail**
- [ ] **Step 3: Implement `Candidate`, `__init__`, `reset`, `to_absolute_y`, `candidates`**
- [ ] **Step 4: Run and confirm both pass**
- [ ] **Step 5: Commit** -- `feat(teacher): rule-based candidate generation`

---

### Task 3: Forward-simulation ranking

**Files:**
- Modify: `python_ai/teacher.py`
- Test: `python_ai/test_python_ai.py`

**Interfaces:**
- Produces:
  - `UtilityTeacher.score(self, env, cand, baseline) -> float`
  - `UtilityTeacher.rollout_stats(self, env, cand) -> dict[str, float]` with keys
    `tower_dealt`, `tower_taken`, `killed`, `lost`, `crowns`.
  - `UtilityTeacher.act(self, env, obs_own) -> tuple[int, float, float]`
  - `PROFILES: dict[str, dict[str, float]]` with keys
    `w_twr`, `w_def`, `w_trade`, `w_crown`, `w_cycle`.

Each candidate is rolled `horizon_ticks` forward on `env.snapshot()` with both
sides no-op after the play. Score is **differenced against the no-op rollout**, so
no-op scores exactly 0 and every candidate carries a marginal value. That is the
whole of the blueprint's opportunity-cost term: a baseline, not a fifth weight
that would need calibrating against terms measured in different units.

At `horizon_ticks == 0` no rollout runs at all and the ranking falls back to a
rules-only priority order -- this is what makes curriculum stages 0-1 free.

- [ ] **Step 1: Write the failing tests**

```python
def test_noop_scores_exactly_zero_so_the_teacher_can_hold_elixir():
    """The no-op baseline is what makes every other score a MARGINAL value.
    If this drifts, the bot either dumps elixir on sight or freezes."""
    import numpy as np, clash_royale_env as E
    from teacher import UtilityTeacher
    from gym_wrapper import DEFAULT_DECK
    CE = E.ClashRoyaleEnv
    env = CE(DEFAULT_DECK, DEFAULT_DECK, 3600); env.reset()
    for _ in range(40):
        env.step_self_play(-1, 0, 0, -1, 0, 0, 10)
    t = UtilityTeacher(DEFAULT_DECK, team=0, horizon_ticks=30); t.reset()
    obs = np.asarray(env.get_observation_for_team(0), np.float32)
    noop = [c for c in t.candidates(obs, env.get_hand_for_team(0),
                                    env.get_elixir_for_team(0))
            if c.slot == CE.HAND_SIZE][0]
    base = t.rollout_stats(env, noop)
    assert t.score(env, noop, base) == 0.0


def test_rollout_does_not_touch_the_live_match():
    """snapshot() deep-copies the stats collectors; if that ever regresses,
    every hypothetical hit would be posted into the real match's statistics,
    and those feed the reward shaping."""
    import numpy as np, clash_royale_env as E
    from teacher import UtilityTeacher
    from gym_wrapper import DEFAULT_DECK
    CE = E.ClashRoyaleEnv
    env = CE(DEFAULT_DECK, DEFAULT_DECK, 3600); env.reset()
    for _ in range(40):
        env.step_self_play(-1, 0, 0, -1, 0, 0, 10)
    before = (env.get_tower_damage_dealt(0), env.get_tower_damage_dealt(1),
              env.get_elixir_spent(0), env.get_elixir_for_team(0))
    t = UtilityTeacher(DEFAULT_DECK, team=0, horizon_ticks=30); t.reset()
    obs = np.asarray(env.get_observation_for_team(0), np.float32)
    t.act(env, obs)
    after = (env.get_tower_damage_dealt(0), env.get_tower_damage_dealt(1),
             env.get_elixir_spent(0), env.get_elixir_for_team(0))
    assert before == after


def test_teacher_is_side_agnostic():
    """One class plays both sides. If team 1's frame conversion is wrong the
    bot silently no-ops forever, which looks like a weak bot, not a bug."""
    import numpy as np, clash_royale_env as E
    from teacher import UtilityTeacher
    from gym_wrapper import DEFAULT_DECK
    CE = E.ClashRoyaleEnv
    env = CE(DEFAULT_DECK, DEFAULT_DECK, 3600); env.reset()
    t1 = UtilityTeacher(DEFAULT_DECK, team=1, horizon_ticks=30); t1.reset()
    played = 0
    for _ in range(120):
        obs1 = np.asarray(env.get_observation_for_team(1), np.float32)
        slot, x, y = t1.act(env, obs1)
        spent_before = env.get_elixir_spent(1)
        env.step_self_play(-1, 0, 0, slot, x, y, 10)
        if env.get_elixir_spent(1) > spent_before:
            played += 1
        if env.is_game_over():
            break
    assert played >= 5, f"team-1 teacher only landed {played} cards in 120 steps"
```

- [ ] **Step 2: Run and confirm all three fail**
- [ ] **Step 3: Implement `PROFILES`, `rollout_stats`, `score`, `act`**
- [ ] **Step 4: Run and confirm all three pass**
- [ ] **Step 5: Commit** -- `feat(teacher): forward-simulation ranking against a no-op baseline`

---

### Task 4: Cycle tracking

**Files:**
- Modify: `python_ai/teacher.py`
- Test: `python_ai/test_python_ai.py`

**Interfaces:**
- Produces:
  - `class CycleTracker` with `observe(hand: list[int]) -> None`,
    `distance_to(card_id: int) -> int` (cards until it returns; `0` = in hand),
    `reset() -> None`.
  - `UtilityTeacher.cycle_value(self, card_id) -> float`

The hand is a deterministic rotation, so our own cycle is exact and free from
observed hand transitions -- no engine change and no new binding. It enters as a
**scored term, never a hard rule**: a wrong estimate then degrades the ranking
slightly instead of freezing the bot into a wrong line.

- [ ] **Step 1: Write the failing test**

```python
def test_cycle_tracker_counts_distance_to_the_win_condition():
    import clash_royale_env as E
    from teacher import CycleTracker
    from gym_wrapper import DEFAULT_DECK
    CE = E.ClashRoyaleEnv
    env = CE(DEFAULT_DECK, DEFAULT_DECK, 3600); env.reset()
    ct = CycleTracker(DEFAULT_DECK); ct.reset()
    for _ in range(400):
        hand = env.get_hand_for_team(0)
        ct.observe(hand)
        d = ct.distance_to(15)
        if 15 in hand:
            assert d == 0, f"Hog is in hand {hand} but distance_to said {d}"
        else:
            assert d > 0
        # cheapest affordable, to keep the hand rotating
        slot = 0
        env.step_self_play(slot, 9.0, 10.0, -1, 0, 0, 10)
        if env.is_game_over():
            break
```

- [ ] **Step 2: Run and confirm it fails**
- [ ] **Step 3: Implement `CycleTracker` and `cycle_value`**
- [ ] **Step 4: Run and confirm it passes**
- [ ] **Step 5: Commit** -- `feat(teacher): exact own-cycle tracking as a scored term`

---

### Task 5: Difficulty ladder and profile randomization

**Files:**
- Modify: `python_ai/teacher.py`
- Test: `python_ai/test_python_ai.py`

**Interfaces:**
- Produces:
  - `TEACHER_STAGES: list[dict]`, each `{"horizon_ticks", "epsilon", "k_cells"}`,
    six entries matching the spec's table.
  - `UtilityTeacher.set_stage(self, stage: int) -> None`

`epsilon` substitutes a uniformly random **legal** action for the argmax. Profile
and lane bias are drawn in `reset()` so the decision rule stays deterministic
given the profile while the opponent is not memorizable across episodes.

- [ ] **Step 1: Write the failing test**

```python
def test_teacher_stages_are_monotone_in_competence_and_never_touch_elixir():
    from teacher import TEACHER_STAGES
    assert len(TEACHER_STAGES) == 6
    for s in TEACHER_STAGES:
        assert set(s) == {"horizon_ticks", "epsilon", "k_cells"}
        assert "opp_elixir_multiplier" not in s
    eps = [s["epsilon"] for s in TEACHER_STAGES]
    hor = [s["horizon_ticks"] for s in TEACHER_STAGES]
    assert eps == sorted(eps, reverse=True)   # noise falls monotonically
    assert hor == sorted(hor)                 # lookahead rises monotonically


def test_epsilon_one_still_only_emits_legal_actions():
    import numpy as np, clash_royale_env as E
    from teacher import UtilityTeacher
    from gym_wrapper import DEFAULT_DECK
    CE = E.ClashRoyaleEnv
    env = CE(DEFAULT_DECK, DEFAULT_DECK, 3600); env.reset()
    for _ in range(40):
        env.step_self_play(-1, 0, 0, -1, 0, 0, 10)
    t = UtilityTeacher(DEFAULT_DECK, team=0, horizon_ticks=0, epsilon=1.0, seed=0)
    t.reset()
    for _ in range(50):
        obs = np.asarray(env.get_observation_for_team(0), np.float32)
        slot, x, y = t.act(env, obs)
        if slot != CE.HAND_SIZE:
            cid = env.get_hand_for_team(0)[slot]
            assert env.is_valid_placement(cid, x, t.to_absolute_y(y), 0)
        env.step_self_play(-1, 0, 0, -1, 0, 0, 10)
```

- [ ] **Step 2: Run and confirm both fail**
- [ ] **Step 3: Implement `TEACHER_STAGES`, `set_stage`, epsilon and profile draw**
- [ ] **Step 4: Run and confirm both pass**
- [ ] **Step 5: Commit** -- `feat(teacher): competence ladder replaces the elixir ladder`

---

### Task 6: `MicroRoyaleEnv` teacher hook

**Files:**
- Modify: `python_ai/gym_wrapper.py`
- Test: `python_ai/test_python_ai.py`

**Interfaces:**
- Produces:
  - `env_config["opponent"] = "teacher"` (default `"builtin"`, i.e. today's
    behaviour) and `env_config["teacher_stage"] = int`.
  - `MicroRoyaleEnv.set_teacher_stage(self, stage: int) -> None`
  - `MicroRoyaleEnv.teacher` -- `None` when not configured.

`MicroRoyaleEnv.step` has exactly one engine call site. With a teacher configured
it routes through `step_self_play`, which deliberately never calls
`opponentTurn()` -- so the C++ `HeuristicOpponent` is absent on that path, which
is exactly what is wanted. Both entry points accumulate `calculateReward()`
identically, so the reward stream is unchanged. The `info` dict is built from the
same accessors either way and must stay byte-identical in shape.

- [ ] **Step 1: Write the failing tests**

```python
def test_default_env_is_unchanged_and_still_uses_the_cpp_heuristic():
    from gym_wrapper import MicroRoyaleEnv
    env = MicroRoyaleEnv()
    assert env.teacher is None


def test_teacher_env_produces_the_same_info_keys_as_the_builtin_env():
    import numpy as np
    from gym_wrapper import MicroRoyaleEnv
    a = MicroRoyaleEnv(); a.reset()
    b = MicroRoyaleEnv({"opponent": "teacher", "teacher_stage": 2}); b.reset()
    assert b.teacher is not None
    act = {"card_index": np.array([4]), "target_x": np.array([9.0]),
           "target_y": np.array([10.0])}
    _, _, _, _, ia = a.step(act)
    _, _, _, _, ib = b.step(act)
    assert set(ia) == set(ib)


def test_teacher_opponent_actually_plays_cards():
    import numpy as np
    from gym_wrapper import MicroRoyaleEnv
    env = MicroRoyaleEnv({"opponent": "teacher", "teacher_stage": 3})
    env.reset()
    act = {"card_index": np.array([4]), "target_x": np.array([9.0]),
           "target_y": np.array([10.0])}
    for _ in range(120):
        _, _, term, _, info = env.step(act)
        if term:
            break
    assert info["team1_elixir_spent"] > 0.0
```

- [ ] **Step 2: Run and confirm they fail**
- [ ] **Step 3: Implement the hook**
- [ ] **Step 4: Run and confirm they pass**
- [ ] **Step 5: Commit** -- `feat(env): optional utility-teacher opponent`

---

### Task 7: `train.py` curriculum rewrite

**Files:**
- Modify: `python_ai/train.py`
- Test: `python_ai/test_python_ai.py`

`CURRICULUM_STAGES` loses `opp_elixir_multiplier` and gains `teacher_stage`. The
entropy-reset, stage-snapshot and phase-transition machinery around it is
untouched -- in particular the phase transition is still evaluated **before**
stage advancement, because the stage gate calls `outcome_history.clear()`.

`PHASE2_ENTRY_WIN_RATE` keeps its current meaning: it gates `mirror` ->
`random_opponent`, **not** the pipeline handoff, despite the name.

- [ ] **Step 1: Write the failing test**

```python
def test_training_never_raises_the_opponent_elixir_multiplier():
    """The 1.5x handicap is what priced the win condition negatively. The API
    survives for measurement harnesses; the TRAINING path must not use it."""
    import re, pathlib
    src = pathlib.Path("python_ai/train.py").read_text(encoding="utf-8")
    for m in re.finditer(r"set_opponent_elixir_multiplier\(([^)]*)\)", src):
        arg = m.group(1)
        assert "1.0" in arg or "1.0)" in arg, \
            f"train.py sets a non-1.0 multiplier: {m.group(0)}"


def test_curriculum_stages_are_competence_not_economy():
    import sys; sys.path.insert(0, "python_ai")
    import train
    for s in train.CURRICULUM_STAGES:
        assert "opp_elixir_multiplier" not in s
        assert "teacher_stage" in s
    assert [s["teacher_stage"] for s in train.CURRICULUM_STAGES] == [0, 1, 2, 3, 4, 5]
```

- [ ] **Step 2: Run and confirm both fail**
- [ ] **Step 3: Rewrite the curriculum and wire `set_teacher_stage` where the
      multiplier was set**
- [ ] **Step 4: Run and confirm both pass, and that `train.py --help` still runs**
- [ ] **Step 5: Commit** -- `feat(train): competence curriculum at a symmetric 1.0x economy`

---

### Task 8: `prove_environment.py` -- the zero-training falsifier

**Files:**
- Create: `python_ai/prove_environment.py`

Teacher vs Teacher. Arm A: full deck. Arm B: identical, except the win condition
is removed from the candidate set (never played, still cycled). Paired via
`snapshot()` so both arms play a bit-identical opening. Swept over opponent
multiplier in `{1.0, 1.25, 1.5}`.

Reports per multiplier: arm A win rate, arm B win rate, paired delta, bootstrap
CI, exact sign test, and an explicit **VOID** marker when arm A >= 0.95 (the
ceiling caveat that voided the original test's 1.00x row).

- [ ] **Step 1: Write the harness with `--n`, `--multipliers`, `--stage`, `--seed`**
- [ ] **Step 2: Run at small n (20) to check it completes and the pairing holds**
- [ ] **Step 3: Run the real measurement at n >= 120 per multiplier**
- [ ] **Step 4: Record the numbers verbatim in the plan's Results section below**
- [ ] **Step 5: Commit** -- `test(env): zero-training falsifier for win-condition pricing`

---

### Task 9: `prove_teacher.py` -- teacher strength

**Files:**
- Create: `python_ai/prove_teacher.py`

Two bars, **both** required:

1. **Strong enough** -- beats the C++ `HeuristicOpponent` decisively at 1.0x.
   Otherwise it is not an upgrade on what Phase 1 already had.
2. **Not a wall** -- still beatable by `shipping.py`'s config (cured weights +
   search h=12). An opponent nothing can beat produces a flat reward signal,
   which is the zero-gradient failure in a new costume.

Also reports teacher vs C++ heuristic at 1.5x, to place it alongside every
historical number in `CLAUDE.md`.

- [ ] **Step 1: Write the harness with `--n`, `--stage`, `--vs {heuristic,shipping}`**
- [ ] **Step 2: Run teacher vs heuristic at 1.0x and 1.5x, per stage**
- [ ] **Step 3: Run shipping net + search vs teacher at the top stage**
- [ ] **Step 4: Record the numbers in the Results section**
- [ ] **Step 5: Commit** -- `test(teacher): strength bars against the heuristic and the shipping agent`

---

### Task 10: Offensive scenario injection (default-off)

**Files:**
- Create: `python_ai/scenario_offense.py`
- Modify: `python_ai/train.py`

`punish_window` (we hold ~8 elixir with the win condition in hand; the opponent
has just committed and sits near 1) and `counter_push` (our defence has just
survived with units alive to support). Built on the same `inject` +
`set_elixir_for_team` + `set_hand_for_team` primitives the existing phase-2
scenarios use. `set_hand_for_team` returns `False` on a hand that is not a valid
permutation of the deck -- **check it**, because a silently rejected setup makes
the scenario a no-op that still counts as an injected episode.

Controlled by `OFFENSIVE_SCENARIO_PROB`, default **0.0**. Turned on only if Task 8
passes.

- [ ] **Step 1: Write a test that the constructors produce legal, applied states**
- [ ] **Step 2: Run and confirm it fails**
- [ ] **Step 3: Implement the constructors and the (default-off) train.py hook**
- [ ] **Step 4: Run and confirm it passes**
- [ ] **Step 5: Commit** -- `feat(train): offensive scenario injection, default-off`

---

### Task 11: Full regression

- [ ] **Step 1:** `python_ai/venv/Scripts/python.exe -m pytest python_ai/test_python_ai.py -q`
- [ ] **Step 2:** `perception/.venv/Scripts/python.exe -m pytest perception/tests -q` (344 tests; the teacher shares `tactics.py`)
- [ ] **Step 3:** `python_ai/venv/Scripts/python.exe python_ai/validate_pipeline.py`
- [ ] **Step 4:** Commit any fixes

---

### Task 12: From-scratch smoke run

- [ ] **Step 1:** Wipe to Episode 0 into a scratch checkpoint path (never over
      `model_weights.pth` until the gate passes)
- [ ] **Step 2:** Run ~2-4k episodes on the new curriculum
- [ ] **Step 3:** `probe_card_usage.py --greedy` and `probe_perfect_defense.py`
- [ ] **Step 4:** Compare against the pre-registered table (spec section 6.3)
- [ ] **Step 5:** Record results

---

### Task 13: Documentation

**Files:** `CLAUDE.md`

- [ ] Record the pivot, the decision, and every measured number from Tasks 8, 9, 12
- [ ] State plainly that this is **gameplay-affecting**: every win rate earned
      against `heuristic@{1.0..1.5}x` is now historical
- [ ] Note that checkpoints are **not** invalidated (no observation/action/
      architecture change) -- weights are wiped by choice, not necessity
- [ ] Fix the stale filenames: `probe_defense.py` -> `probe_perfect_defense.py`,
      `probe_entropy_norm.py` -> `probe_card_usage.py`
- [ ] Commit

---

## Results

All measured 2026-08-19 on the 2.6 Hog Cycle deck, teacher stage 5 unless noted.

### Task 9 -- teacher strength (both bars PASS)

| | result |
|---|---|
| bar 1: vs C++ `HeuristicOpponent` @1.0x, stage 5 (n=12) | **1.000** |
| ...stages 0-4 | 0.54 / 0.29 / 0.71 / 0.96 / 0.92 |
| bar 2: ep-31312 2.6 net (greedy), sides swapped (n=20) | net **0.775**, CI [0.663, 0.875] |
| cost | 3.47 ms/decision, ~1 s/episode at stage 5 |

Beatable but not free -- and materially harder than the C++ heuristic, which
that same net beats ~100%. Stage 1 scoring below stage 0 is a real
non-monotonicity in the rules-only rungs and is n=12 noise-dominated.

**The teacher is not a turtle**, which was the first thing that could have gone
wrong. Win-condition share of its own plays RISES with competence:

| stage | Hog share of plays | P(play) |
|---|---|---|
| 0 | 7.1% | 0.182 |
| 3 | 13.4% | 0.160 |
| 5 | **16.5%** | 0.165 |

against the RL agent's 0.8%. It uses all eight cards.

### Task 8 -- the falsifier, and it did NOT go as predicted

**Symmetric, teacher vs teacher, n=100 paired per row:**

| opp elixir | attack | cycle | delta | p |
|---|---|---|---|---|
| 1.00 | 0.510 | 0.840 | **-0.330** | 5.7e-08 |
| 1.10 | 0.145 | 0.455 | -0.310 | 4.5e-07 |
| 1.20 | 0.015 | 0.130 | -0.115 | 7.6e-05 |

Re-run at 1.0x with the continuous readout and the (confounded) ban arm:

| arm | win | enemy tower dmg dealt | tower dmg TAKEN |
|---|---|---|---|
| attack | 0.540 | 6326.3 | **5772.7** |
| cycle | 0.875 | 6938.4 | **3049.3** |
| ban | 0.910 | -- | -- |

Committing the win condition does not even raise our own tower damage
(-612, p=0.13) and nearly DOUBLES the damage we take. Never playing it at all
(0.910) beats attacking with it by 0.37.

**Dose-response vs the C++ heuristic** (the historical setup, policy confound
removed), n=80: 1.00x VOID at ceiling; 1.25x **-0.1875** (p=0.0096); 1.50x
**-0.1062** (p=0.00049). So the historical finding REPLICATES with a
deterministic bot -- it was never a policy failure.

**Marginal value of one commitment** at the advisor's own gate, both sides
playing on, n=220 states: **-298.2 tower HP, CI [-494.3, -107.1]**
(sign test 87/105, p=0.22 -- the CI and the sign test disagree, so the effect is
size-driven, not frequency-driven).

### The mechanism, isolated

An unopposed lone Hog is **not** weak: injected on an empty board it deals
**2536 tower damage** (a full Princess Tower) and dies at tick 190. So neither
the card nor the enemy King firing from tick 0 is what suppresses it.

`prove_wincon_trade.py`, n=60 paired, defender playing on for 30 s:

| defender elixir | arm | they spend | tower dmg | dmg/elixir committed |
|---|---|---|---|---|
| match state | hog (4) | 1.07 | 634.0 | 158.5 |
| match state | supported (6) | 1.72 | 978.2 | 163.0 |
| **forced to 1.0** | hog (4) | 1.32 | **1025.0** | **256.2** |
| forced to 1.0 | supported (6) | 0.90 | 1162.3 | 193.7 |

**The punish window is real and large: +391 tower HP (+62%) for hitting a
bankrupt defender.** Escorting helps against a solvent defender (+344 HP, CI
[+186, +524]) and is wasted against a broke one (dmg/elixir 256 -> 194), which
is a correct and fairly subtle strategic result.

So the card works and the window exists. What fails is SELECTION: the advisor
commits whenever estimated opponent elixir is <= `HOG_MAX_OPP_ELIXIR = 7.0`,
which is not a punish window at all.

### A flaw found in the teacher itself

`UtilityTeacher.rollout_stats` rolls candidates forward with **both sides
no-oping**. That is fine for ranking a defensive placement a second ahead and
wrong for an attack: the entire cost of committing a win condition is the
counter-push that lands while our half is empty, and an opponent frozen on no-op
never counter-pushes. **The teacher's own scorer therefore cannot see the cost
of attacking**, which is consistent with it over-committing at a gate of 7.0.
Stated as a known limitation rather than silently fixed, because the fix
(letting the opponent act inside every rollout) costs ~20x per decision and the
cheap alternative -- an explicit solvency term -- needs its threshold chosen by
the gate sweep now running.
