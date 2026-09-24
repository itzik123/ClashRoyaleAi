# Changelog

Every release of ClashRoyaleEnv, newest first. Version numbers follow the project's
compatibility boundaries: a **major** version means earlier checkpoints no longer load
or a new training lineage starts, a **minor** version adds features or changes
gameplay, and a **patch** fixes something or marks a point in history. A release marked
*gameplay-affecting* changes the simulated game, so win rates from either side of it
are not comparable.

The reasoning behind each change, with its measurements, is in
[`DECISIONS.md`](DECISIONS.md).

## Unreleased

*Gameplay-affecting.*

- A troop knocked off a bridge onto the riverbank no longer vibrates at the bridge
  mouth. It used to overshoot the mouth by a partial step, turn back, and repeat
  indefinitely: 18% of teacher-vs-teacher matches had one, up to 52 s long.
  `Troop::moveTowards` now caps each step at the remaining distance. See
  `perception/UPSTREAM_REQUESTS.md` item 31.

**Compatibility:** Checkpoints load. Troop trajectories change, so win rates are not
comparable across this change.

## v3.1.0 — Deck-derived spell terms, and a public release

*2026-09-23*

The spell reward terms follow whatever damage spell the deck carries, the teacher aims each spell with its own geometry, and the repository is prepared for going public.

- Both spell reward terms follow the deck's own damage spell, chosen by measured Crown Tower damage. `main` could not start a run before this: `compute_shaping` raised `KeyError('spell_in_hand')` on the first step. The shipped 2.6 deck is bit-identical (0 of 2,280 steps differ).
- `card_probes.spell_effect` reads damage-over-time spells to completion; Poison measured 368 of its 736.
- The teacher aims every spell with that spell's own radius and damage instead of Fireball's (Rocket +38% value on 320 boards, Zap +11%, Poison +7%).
- Open audit items closed: the overflow regime is constructed, a spawner building is no longer a siege win condition, the potential-based terms are documented as they are, every `CLASH_*` setting is stamped into the checkpoint, the rules teacher has an air defence, PFSP estimates survive a resume, and the exploiter refuses a Champion deck.
- The Log's damage is measured instead of restated (240 -> 269). A pre-checked table of 28 trainee decks is in the runbook.
- Proposed, not changed: spells hit Crown Towers for 100% of their damage (real game: Fireball 159 of 688). See `perception/UPSTREAM_REQUESTS.md` item 29.
- Public release: MIT license, a README for outside readers with a Simulation View demo, and the loose root documents moved under `docs/` and `.claude/`.

**Compatibility:** Checkpoints from v3.0.0 load. The spell terms change the reward for any deck without Fireball.

## v3.0.0 — Deck-agnostic training, ready for a from-scratch run

*2026-09-16*

A pre-launch audit hunted for anything that would crash, switch off silently, or quietly mis-tune when the trainee plays a deck other than the 2.6 Hog Cycle, before a from-scratch run on a new deck.

- The trainee deck is set by `CLASH_DECK`, validated at startup and recorded in the checkpoint; resuming under a different deck warns.
- Four mechanisms keyed to the 2.6 deck went silent under other decks and now derive from the deck: the advisor target, the win-condition resolver (one resolver, ranked by measured tower damage), scenario injection, and the placement mask. The mask had deleted the enemy half for deploy-anywhere troops and gave every Evolution an all-illegal row.
- Engine: damage to spawned bodies was booked as tower damage (810 for one Goblin Barrel). This fed both the agent's tower reward and the teacher's rollouts.
- Reward: a phantom +0.084 on the first real step of every episode is gone.
- Teacher: the top rung froze against an opponent sitting on full elixir. Failed three-crowns against a passive opponent fell from 30 to 3 of 116 matches.
- Cold start and unattended runs: a rung-0 floor alarm, uniform deck sampling for a fresh net, crash-safe resume and checkpoint writes, per-card placement modal share logged, and the next-card auxiliary loss ramped in over 2,000 episodes.
- Champion and Hero abilities are trained as part of the joint action, masked by readiness.

**Compatibility:** New lineage. The next run starts from scratch on a different deck. C++ suite: 714 cases, 713 pass plus 1 expected failure.

## v2.3.0 — A teacher that pilots every deck; overtime and sudden death

*2026-09-15*

The phase-1 opponent can now actually attack with all sixteen meta decks, matches end the way real ones do, and the curriculum reads a signal that PFSP does not regulate.

- The teacher found no win condition in 5 of the 16 decks (siege buildings, bait and Graveyard), so it never played the card those decks are named for. Siege reach and spawning spells are now detected by injection. Mortar Cycle, for example, now three-crowns a passive opponent in 882 ticks (was 1,769).
- Graveyard dealt zero tower damage: its spawn cadence exactly matched a tower's fire rate. It now spawns 12 Skeletons at 0.5 s, like the real card.
- Match end: a crown lead at 3:00 wins, and the first crown in overtime wins (sudden death).
- The curriculum's plateau valve, backstop and deck floor were reading a PFSP-regulated win rate; they were redirected, and the deck floor is off.
- Decision-time search is net-negative against the teacher (-0.31 to -0.53 win rate), so the shipped agent runs its policy greedy. Adding an opponent model brings search to break-even.
- Live loop: the neural policy crashed on the first in-game frame, and the confirmation oracle read a different hand than the policy.
- Tooling for long unattended runs: a watchdog and a report builder.

**Compatibility:** Gameplay-affecting: five opponents got materially stronger. Per-deck mean win rate over the run: 0.301 -> 0.639.

## v2.2.0 — Real elixir phases and a meta-deck opponent pool

*2026-09-03*

The match economy follows the real game's schedule, and the phase-1 opponent plays sixteen real ladder decks instead of a mirror.

- Elixir runs 1x, then 2x from 2:00 and 3x from 3:00. It had been starved: over 4,229 decisions the agent's mean elixir was 2.16, and it was never at 9 or above.
- The phase-1 opponent samples 16 RoyaleAPI ladder decks by PFSP weight. The 2.6 mirror ranked 16th of 16 on the opportunity it offers Cannon, The Log and Fireball, which explains why those three cards went unplayed.
- Curriculum: 6 rungs at a 0.80 gate became 11 rungs at 0.65, each moving one difficulty axis, plus a plateau valve and a backstop. One stage had consumed 23,040 episodes without advancing.
- The next-card auxiliary head was confidently wrong on unfamiliar decks (cross-entropy 13.76 against a uniform 5.22); its loss is capped and the heads can be reset.
- Measured negative and recorded so they are not re-run: a human placement prior mined from 221 expert replays (-7.3 win-rate points, n = 800), and a deck-coverage floor with and without threat gating.
- Perception reads the hand from deck-specific templates (100% on 120 held-out slot crops).

**Compatibility:** Observation 13,976 -> 13,977. Checkpoints migrate with `python_ai/tools/migrate_checkpoint_elixir_phase.py`. Gameplay-affecting.

## v2.1.0 — Centre-to-centre sight and rolling spells

*2026-08-29*

Targeting, sight and placement follow the real game's geometry, and The Log and Barbarian Barrel roll instead of exploding in place.

- Sight is strict centre-to-centre. Radii had inflated every unit's aggro range; a Hog Rider's 9.5 became 10.9 against a Cannon.
- Placement checks a body's footprint against the arena edge, so buildings lose columns 0 and 17 and troops lose nothing.
- Building-targeters go to the nearest building, towers included, and targets are compared by footprint distance.
- The Log and Barbarian Barrel sweep a forward corridor with lateral knockback (The Log: 3.9 wide, 10.1 long). Rolling spells can be cast only on the own half or the river.
- The auxiliary opponent-elixir head, which was solvable from two present scalars, was replaced by a next-card classifier that needs memory.
- A card-counting probe, and an autopsy showing Cannon, The Log and Fireball were already being avoided by episode 2,018, long before the stage-3 plateau they were blamed on.

**Compatibility:** Gameplay-affecting. The Log's learned placements and the legal building area both changed.

## v2.0.0 — Observation v4, and a network that can count cards

*2026-08-27*

The observation, the objective and the network all changed. Every checkpoint trained before this release is architecturally incompatible.

- The opponent's card cycle is observable: `seen[]` and `recency[]` blocks for all 185 card ids, with a card's identity treated as observable in the same way its elixir cost already was. Observation 13,606 -> 13,976. Sections are located by forward offsets (`EXTRA_SCALARS_START`, `CYCLE_START`); five call sites that subtracted from the end would otherwise have corrupted the reward.
- Discount 0.99 -> 0.999. The horizon was 100 decisions against a 360-decision match, so about 98% of the objective was shaping and one crown was worth 22.4 wins.
- Network: the trunk's receptive field grew from 10x10 to the whole board (dilated context blocks), the scalar encoder was split into four branches, and BPTT chunks went 25 -> 50 so the gradient outlasts a card rotation. 1,907,329 -> 1,846,963 parameters for +14.8% update time.

**Compatibility:** Breaking. Checkpoints from before this release do not load into the new network.

## v1.5.0 — The engine and training audit

*2026-08-27*

A three-part audit of the C++ engine and a full audit of the training stack, fixing defects that were silent rather than loud.

- A living Mortar kept its owner's King alive (both used the symbol 'R'), so King kills became timeouts.
- Ice Golem slowed with its attack instead of its death explosion; Ice Spirit half-slowed instead of stunning.
- A freeze stopped movement one tick short; buildings outlived their lifetime by about a second.
- Golden Knight's dash ran backwards; Hero Giant threw enemy buildings; Cursed Hog and Royal Chef re-armed on every application.
- Nineteen spawned units ran a quarter slower than their own card, and Golemite moved below the slowest real tier.
- Training: checkpoint paths anchored (the launch directory decided whether a run resumed), atomic checkpoint writes, NaN containment in one guarded optimizer step, seeded runs, a side-null harness, and search no longer values a draw at zero.

**Compatibility:** Gameplay-affecting; checkpoints load. C++ suite 669 -> 673 cases.

## v1.4.0 — Real speed tiers and live-mirror state setters

*2026-08-24*

Troops move at the real game's speeds, and the simulator can be set to a position read off the screen.

- Speed tiers: the engine had none, and 104 of 131 troops were at the wrong speed. They now use the real game's five tiers (30/45/60/90/120 tiles per minute), verified against Supercell's exported card table. The largest gameplay change to date.
- State setters for driving the simulator from perception: tower HP and destruction, the match clock, and `inject()` with HP and remaining deploy time.
- `step_self_play_fast` skips building the observations a rollout throws away.
- The placement head stopped computing rows nobody reads: PPO update 27.38 -> 17.99 s (1.52x).
- All 148 sight ranges pinned by a catalogue test; engine seeding wired through.
- The knowledge base split into an operational reference and a decision log; perception dropped ~2k lines of vendored agent code.

**Compatibility:** Gameplay-affecting; checkpoints load, but earlier win rates are void.

## v1.3.0 — A search-based teacher, the 2.6 Hog Cycle, and a faithful arena

*2026-08-22*

The curriculum opponent became a teacher that ranks its plays by simulation, and the arena was rebuilt to match the real game's geometry.

- The default deck is the 2.6 Hog Cycle.
- Curriculum pivot: the phase-1 opponent proposes plays by rules and ranks them by simulating each one forward. Its difficulty is how far ahead it looks. It also plans multi-card combos (tank now, win condition next) and assumes the other side keeps playing.
- Deploy time: units now take a second to deploy, removing a subsidy the defender had been getting on every placement.
- Arena geometry lives in one header, re-centred on the real board (bridges at 2.5 / 14.5). The King Tower sleeps until activated, blind units follow their lane, and two pathing deadlocks are fixed.
- `python_ai/` restructured into a package with one subpackage per responsibility.
- Replay viewer overhaul, including the Teacher Debug Simulation View.

**Compatibility:** Gameplay-affecting.

## v1.2.1 — The neural placement cure

*2026-08-15*

The placement head no longer needs the tactical advisor for Fireball, and beats chance on all three cards it had been failing.

- Fireball elixir killed 0.062 -> 2.565, statistically level with the advisor (2.464).
- Cannon (+141.8 tower HP) and Giant (+145.2 tower damage) beat a random legal cell for the first time.
- The tactical override became a measured null (0.507 vs 0.480, p = 0.56).

**Compatibility:** Gameplay-affecting: the spell-value term is live and annealing.

## v1.2.0 — Live-ready hybrid baseline

*2026-08-14*

The network chooses what and when; a deterministic advisor chooses where for Cannon, Fireball and Giant; a solvency gate vetoes bankrupting spends.

- +11.8 win-rate points (0.584 -> 0.703, p = 1.9e-05, n = 600 paired).
- Verified against the real game: 250 decisions at 0.96 Hz, none over the 1,000 ms budget.

## v1.1.0 — Decision-time search and expert iteration

*2026-08-13*

A snapshot mechanism in the simulator, and the two things it unblocks.

- 1-ply search: +0.319 win rate (0.625 -> 0.944, p = 5.6e-12, 160 paired trials) against the built-in heuristic at 1.5x elixir.
- Distilling the search's value distribution back into the policy: +0.045 (p = 0.0074, 1,600 paired trials), at no inference cost.

## v1.0.1 — State before the decision-time search work

*2026-08-11*

Marks the last commit before the snapshot mechanism, so the before/after boundary of that feature is addressable.

## v1.0.0 — First live match won, screen to card, end to end

*2026-08-06*

The perception -> policy -> actuation loop playing a real match unaided.

- Screen capture, detection, and an encoder verified bit-exact against the engine.
- Decisions at 0.97 Hz with none of 49 over budget; a placement takes 347 ms.

## v0.4.0 — Observation v3 and the perception stack

*2026-07-31*

The observation was rebuilt, both teams see the board correctly, and the first perception pipeline reads real matches.

- Observation rebuilt from 9 to 21 channels (13,606 values); earlier checkpoints are incompatible.
- Team 1 fixes: a river-placement asymmetry, and an observation displaced by one row (the self-play null moved 0.598 -> 0.520).
- `inject()` and `get_hand_for_team()` for driving the engine from outside.
- `perception/`: arena calibration, a vendored Build-A-Bot detector, King Tower HP, a unit-HP reader, the State -> GameState adapter, live capture via Windows.Graphics.Capture, a match recorder, and an elixir ledger.
- The reward system reworked.

**Compatibility:** Observation breaking change (pre-1.0).

## v0.3.0 — Heroes, sight ranges and a richer league

*2026-07-27*

Units see and chase like the real game, the Hero mechanic arrives, and self-play gets scripted opponents and scenarios.

- Sight and aggro ranges: units chase only what they can see, with sourced values for every card that has a non-default range.
- Deck slot rules (Evolution, Heroic, Wild Card), up to two Champions per deck, and two Champion-ability action slots.
- Hero cards: Mini P.E.K.K.A, Musketeer, Knight, Wizard, Giant, Mega Minion, Magic Archer, Ice Golem, Barbarian Barrel and Goblins.
- A random opening hand, the hand-cycle delay, and a slot-aware random-deck generator.
- Scripted opponents in the PFSP pool, and scenario injection in self-play.
- River-crossing fixes for the Prince, Bandit, Mega Knight and Royal Hogs family.

## v0.2.0 — Every card in the game, and a self-play league

*2026-07-22*

The card roster reaches the full live game, including Champions and Evolutions, and training moves to a PFSP self-play league.

- The card roster synced to the live game, with mechanic gaps closed across 35 cards (shields, charge, enrage, parry, hook, invisibility, periodic effects, splash, jump, range falloff).
- All eight Champions on a generic ability framework.
- The Evolutions framework and all 41 Evolutions, cross-checked against a sourced table; Tower Troops (Tower Princess, Cannoneer, Dagger Duchess, Royal Chef); Mirror and Spirit Empress.
- Self-play training, then a prioritized fictitious self-play (PFSP) league.
- Target locking, true radius-based splash, and the rule that buildings never move.

## v0.1.0 — The simulator plays and learns

*2026-07-18*

A working C++ battle engine, Python bindings, a PPO agent training against it, and a replay viewer.

- C++ engine: board, towers, elixir, troops, buildings and spells, collision resolution, ground-versus-air targeting and flying troops, with card stats synced to the live game.
- pybind11 bindings and a PPO agent training against the engine.
- An engine-side match statistics interface (damage, kills, elixir, card plays, outcome).
- A replay viewer and match video output; an entity unit-test suite.
