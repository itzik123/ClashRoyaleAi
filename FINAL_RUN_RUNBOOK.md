# Final run runbook — from scratch, new deck

Written 2026-09-16 at the end of the pre-launch audit (the audit's reports and
probes are summarised in `CLAUDE.md`, "2026-09-15: the pre-launch audit"). Every
command here was checked against the current source. Run everything from the
repo root with the 3.11 venv.

---

## 1. Choose and check the deck

The deck is an environment variable now, not a code edit:

```powershell
$env:CLASH_DECK = "hog rider,musketeer,cannon,ice golem,skeletons,ice spirit,the log,fireball"
```

Names or ids, comma-separated, ignoring case/dots/spaces (`pekka` is P.E.K.K.A.,
`minipekka` is Mini PEKKA). An Evolution needs `evo:` (`evo:archers`), because an
Evolution has the same name as its base card. Unset = the 2.6 Hog Cycle.

**Preview what the deck turns on and off before committing days to it:**

```powershell
python_ai\venv\Scripts\python.exe -c "import python_ai; from python_ai.deck import DEFAULT_DECK; from python_ai.envs.deck_contract import validate_deck, print_report; print_report(validate_deck(DEFAULT_DECK, strict=False))"
```

Read the output:

| line | meaning |
|---|---|
| `WARN ... Champion/Hero` | Trainable since 2026-09-16 (the ability is sampled and scored), but the path is NEW and the mirror teacher activates by a plain heuristic. Watch `Policy/Entropy_Ability` and a replay early on. |
| `win condition: X (N tower HP per elixir)` | the card the reward's win-condition term and the mirror teacher both build around. `WEAK` below 200 is a warning, not a blocker. |
| `WARN no win condition resolves` | the win-condition reward term is off for this deck. |
| `advisor target speaks for K/8 cards` | the placement head gets a rule-based target for these K cards and only an entropy bonus for the rest. 0/8 is a warning. |
| `spell reward terms follow X` | the lethal-spell and value-spell shaping terms (small: ~1.5% of the objective at init) use this deck's own damage spell, with its measured tower damage and cost. |
| `WARN no damaging area spell` | those two terms are off for this deck (The Log alone does not count: it is a roller). |

Engine slot rules the parser enforces: a Champion only in deck slots 1-2, an
Evolution only in slots 0 or 2. Order in the list matters for those.

**If the deck is unusual** (siege, spell win condition, air-heavy), run the
teacher's passive-opponent probe from `CLAUDE.md` ("THE TEACHER COULD NOT PILOT")
on it — the mirror opponent plays your deck, and a teacher that cannot pilot it
makes the mirror rung meaningless. The teacher has no air-defence concept
(audit 05), so an air deck will face a weak early-rung mirror.

---

## 2. Pre-flight

```powershell
Set-Location C:\Users\itzik\source\repos\ClashRoyaleEnv

# nothing to silently resume from
Test-Path python_ai\model_weights.pth             # must be False
Test-Path python_ai\model_weights_selfplay.pth    # must be False

# no trainer already alive -- read the command lines before killing anything
Get-CimInstance Win32_Process -Filter "Name like '%python%'" | Select ProcessId, CommandLine

# the engine and both suites
python_ai\venv\Scripts\python.exe tools\audit\verify_pyd.py
python_ai\venv\Scripts\python.exe -m python_ai.tools.validate_pipeline   # all checks should pass
```

Optional tidying. **No longer required for correctness:** phase 2 now takes
opponents only from snapshots written after this run started (the old
`historical_checkpoints/` held 51 snapshots from the previous lineage). Archiving
still keeps the directories readable:

```powershell
$arch = "C:\Users\itzik\clash_archive\$(Get-Date -Format yyyyMMdd-HHmm)"
New-Item -ItemType Directory -Force $arch | Out-Null
Move-Item historical_checkpoints, stage_checkpoints $arch
```

Use a NEW `CLASH_LOGDIR`: a fresh start deletes the TensorBoard directory it
is pointed at.

---

## 3. Launch (PowerShell, detached)

```powershell
Set-Location C:\Users\itzik\source\repos\ClashRoyaleEnv
$env:CLASH_DECK       = "..."                  # as chosen in step 1
$env:CLASH_LOGDIR     = "runs/final_phase1"    # new directory
$env:CLASH_SAVE_EVERY = "250"                  # ~16 min between saves
$env:PYTHONUNBUFFERED = "1"
Remove-Item Env:CLASH_SEED -ErrorAction SilentlyContinue
$ts = Get-Date -Format yyyyMMdd-HHmmss
$p = Start-Process -FilePath "$PWD\python_ai\venv\Scripts\python.exe" `
       -ArgumentList "-u","-m","python_ai.trainers.train" `
       -WorkingDirectory "$PWD" -WindowStyle Hidden -PassThru `
       -RedirectStandardOutput "$PWD\runs\final_phase1.$ts.out.log" `
       -RedirectStandardError  "$PWD\runs\final_phase1.$ts.err.log"
"trainer PID $($p.Id)"
Get-Content "runs\final_phase1.$ts.out.log" -Wait -Tail 30
```

`CLASH_WEIGHTS` may now be set too — phase 2 follows it. The first lines of the
log are the `[DECK ...]` report; confirm it names the deck you meant.

Closing the terminal should not kill a `Start-Process` child, but verify it once
(`Get-Process -Id <PID>` from a new terminal).

**The handoff.** Phase 1 launches phase 2 and exits once the curriculum clears
the mirror (rung >= 8 at >= 0.60) and then 5,000 random-opponent episodes. Phase 2
logs to `runs\training_selfplay_pfsp.log` (next to your `CLASH_LOGDIR`) and
TensorBoard `<CLASH_LOGDIR>_selfplay`. If phase 2 dies within its first 60 s,
phase 1 now exits non-zero and says why.

---

## 4. Watch

```powershell
python_ai\venv\Scripts\python.exe -m python_ai.tools.monitor_run --dir . --skip-placement
tensorboard --logdir runs
```

`monitor_run` exits non-zero on an ALARM. It now checks win rate, the rung-0
floor alarm, reward trend, rung and plateau advances, worst deck, and per-card
placement modal share.

What to look at, in order:

0. **`Loss/Ratio_Dev_First_Minibatch`** must stay ~0 (float noise, < 1e-3). It is
   the PPO self-check: the policy the rollout acted under and the one the update
   scores must agree.
1. **`Training/Win_Rate_100` and `Training/Avg_Reward_50`.** A random-init policy
   wins ~0 of its first hundred games against the pool; reward moves before win
   rate. A do-nothing policy reads about -8.2 reward per episode at rung 0.
2. **`>>> [FLOOR]` in the log / `Training/Curriculum_FloorAlarm`.** Fires if the
   run is still at <= 0.05 win rate after 1,000 episodes at rung 0 (about an hour).
   Rung 0 has no easier teacher to fall back to, so this does not fix itself:
   check the deck report, the reward and the masks.
3. **`Decks/WinRate_Min`** — the worst matchup; an average hides a total loss.
4. **`Placement/ModalShare_Max`** — above 0.60 a card is landing on one cell
   whatever the board (the placement-collapse signature).
5. **`Training/Curriculum_PlateauAdvances`** against the rung — a ladder climbed
   entirely by plateau is convergence, not mastery.
6. `Aux/NextCard_CE` against ln(8) = 2.08, `Aux/NextCard_Acc` against 0.125.

Throughput to plan with: ~943 episodes/hour, ~77 updates/hour at rung 0,
**falling** as the teacher's lookahead grows up the ladder.

---

## 5. Crash and resume

**Same command, same environment variables, a new stdout log name.** Look for
`Resumed from ...: episode N, curriculum stage S` near the top.

What survives a crash now: weights, Adam, episode count, entropy coefficients,
rung, phase and budget, deck-pool estimates, the plateau/regression tracker, the
lineage start (so the opponent pool stays this run's), and the deck (a different
`CLASH_DECK` on resume prints a loud warning). Each save keeps the previous
checkpoint as `model_weights.pth.prev`. Ctrl-C saves before exiting. Worst case
on a hard kill at `CLASH_SAVE_EVERY=250` is ~16 minutes.

A resume that hits an error now **crashes** instead of moving the checkpoint
aside and deleting the log. If it crashes, the checkpoint and the log are intact;
fix the cause and relaunch.

**Watchdog** (optional, now safe across the handoff):

```powershell
python_ai\venv\Scripts\python.exe python_ai\tools\run_watchdog.py --log "runs\final_phase1.$ts.out.log" --logdir runs/final_phase1 --save-every 250
```

It recognises both phases, relaunches whichever has the most advanced checkpoint,
and refuses to restart more than 4 times an hour.

---

## 6. Known limits going in

- **Champion/Hero ability training is new** (2026-09-16) and has never run at
  length. The mirror teacher uses the ability heuristically, so a Champion deck's
  early rungs are easier than they look.
- The teacher has **no air-defence concept**; air decks meet a weak mirror early.
- **Spells hit Crown Towers for 100% of their damage**, against the real game's
  15-30% (`perception/UPSTREAM_REQUESTS.md` item 29, proposed, not applied). A
  spell-heavy deck will learn chip that does not transfer. If you approve item 29,
  apply it BEFORE launching: this run starts from scratch, so it costs nothing now.
- Spawner huts (Barbarian Hut, Tombstone) measure as tower threats alone, so a
  deck built around one may name the hut as its win condition.
- Observation channels 0-7 still show one unit per cell. Measured: a swarm deck
  hides 21% of troop HP, and the queued max-instead-of-assign change would recover
  only ~2 points, so it was not made.
- Last-hit overkill is still counted as tower damage, and a spawned body killed by
  a spell is credited zero elixir value.
