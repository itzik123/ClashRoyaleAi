"""Rebuild the engine from one perceived frame, so the teacher can search it.

`MirrorBuilder.build(game_state) -> (env, MirrorGaps)`. The env holds the board
perception sees, positioned as closely as the item-22 setters allow; the gaps
record, per build, every field that had to be made up. Design:
docs/design/specs/2026-08-24-live-teacher-play-design.md, sections 3.1 and 5.

Rebuilt from scratch at every decision, never carried forward: an entity
perception stops seeing cannot be deleted from an engine board, and a rebuild
makes that question disappear (the "removal" gap in forecast.py).

The order of the build is load-bearing:

  1. body counts      resolved first, on a SEPARATE env (forecast.py's
                      `bodies_per_card` resets whatever env it measures on)
  2. seed()           a deterministic reset, so the fabricated opponent hand
                      is the same on every build
  3. towers           destroy_tower / set_tower_hp(fraction * engine max)
  4. clock            set_current_tick, so the elixir phase is the engine's own
  5. units            inject(..., hp=fraction * card max, deploy_ticks=0)
  6. elixir, hand     last, so nothing above can move them

What is still fabricated, and why (each is a field of MirrorGaps):

  opponent elixir   no opponent bar exists (contracts.GameState.opp_elixir_spent
                    is always None); left at whatever `assume_opp_elixir` says
  opponent hand     unobservable; whatever the seeded deal gives
  opponent deck     unobservable; defaults to our own
  opponent cycle    inject() records no play, and the teacher does not read it
  unit HP           where no bar was drawn (`hp_measured` False) the unit is
                    full, which perception's recall (0.34-0.56) makes common
  tower HP          where the bar could not be read, the tower is full
"""
from __future__ import annotations

from dataclasses import dataclass, field

from forecast import SimForecaster, group_units, injection_points, tile_centre
from timebase import seconds_to_ticks

#: GameState tower field -> the engine's (team, slot). Slots are BOARD
#: coordinates for both teams (0 King, 1 left Princess, 2 right), the
#: convention `ClashEnv::setTowerHp` documents; the screen is never mirrored,
#: so a "left" tower on screen is the low-x one for either side.
TOWER_SLOTS = (
    ("own_king", 0, 0),
    ("own_princess_left", 0, 1),
    ("own_princess_right", 0, 2),
    ("opp_king", 1, 0),
    ("opp_princess_left", 1, 1),
    ("opp_princess_right", 1, 2),
)

# The engine's fixed match length; set_current_tick clamps to it.
MATCH_TICKS = 3600

# The build's reset. Fixed, so two builds of one frame are bit-identical,
# including the fabricated opponent hand.
DEFAULT_SEED = 0


@dataclass
class MirrorGaps:
    """What one build could not take from perception. Counted, never silent."""

    injected: int = 0
    unmappable_units: int = 0
    """Detected, but the detector's class has no engine card."""

    hp_assumed_full: int = 0
    """Injected units whose bar was not read, so they are at full health."""

    hp_ceiling_unknown: int = 0
    """Injected units with a measured fraction but no known engine maximum (or
    one the observation saturates at), so the fraction could not be applied
    exactly."""

    towers_assumed_full: tuple[str, ...] = ()
    """Towers whose HP was not measured and are left at full."""

    towers_refused: tuple[str, ...] = ()
    """Towers the engine refused to set (e.g. a measured 0.0 not marked
    destroyed: the engine will not hold a 0-hp tower that still stands)."""

    hand_set: bool = False
    hand_reason: str = ""
    """Why the hand could not be set. When `hand_set` is False the mirror's
    slot numbers are NOT the screen's, and nothing may be played from it."""

    opp_elixir_assumed: float | None = None
    opp_hand_fabricated: bool = True
    opp_deck_fabricated: bool = True
    opp_cycle_unrecorded: bool = True

    def fabricated(self) -> list[str]:
        """Names of the fabricated fields, for a one-line log."""
        out = []
        if self.unmappable_units:
            out.append(f"unmappable={self.unmappable_units}")
        if self.hp_assumed_full:
            out.append(f"hp_full={self.hp_assumed_full}")
        if self.hp_ceiling_unknown:
            out.append(f"hp_ceiling={self.hp_ceiling_unknown}")
        if self.towers_assumed_full:
            out.append("towers_full=" + ",".join(self.towers_assumed_full))
        if self.towers_refused:
            out.append("towers_refused=" + ",".join(self.towers_refused))
        if not self.hand_set:
            out.append(f"hand_refused({self.hand_reason})")
        if self.opp_elixir_assumed is not None:
            out.append(f"opp_elixir={self.opp_elixir_assumed:g}")
        if self.opp_hand_fabricated:
            out.append("opp_hand")
        if self.opp_deck_fabricated:
            out.append("opp_deck")
        if self.opp_cycle_unrecorded:
            out.append("opp_cycle")
        return out


@dataclass
class _UnitHp:
    max_hp: float
    saturated: bool


@dataclass
class MirrorBuilder:
    """Owns the mirror env and rebuilds it per decision.

    One owner thread, by construction: the decision thread holds the builder,
    and nothing else touches the env. `src/bindings.cpp` releases the GIL
    nowhere, so a second thread would block perception, not run beside it.
    """

    deck: list
    opp_deck: list | None = None
    assume_opp_elixir: float | None = 5.0
    """Opponent elixir written into the mirror; None leaves the seeded reset's
    value. Either way it is a guess, and recorded as one."""

    seed: int = DEFAULT_SEED
    engine: object = None
    _env: object = field(init=False, default=None)
    _forecaster: object = field(init=False, default=None)
    _hp: dict = field(init=False, default_factory=dict)
    _opp_deck_given: bool = field(init=False, default=False)
    last_opp_bodies: dict = field(init=False, default_factory=dict)
    """Card injections per opponent card in the last build, so a caller can
    tell when the opponent placed something new."""

    def __post_init__(self):
        if self.engine is None:
            import engine as _engine_mod  # noqa: PLC0415
            self.engine = _engine_mod.load()
        self.deck = [int(c) for c in self.deck]
        self._opp_deck_given = self.opp_deck is not None
        self.opp_deck = ([int(c) for c in self.opp_deck]
                         if self.opp_deck is not None else list(self.deck))
        self._env = self.engine.ClashRoyaleEnv(self.deck, self.opp_deck,
                                               MATCH_TICKS)
        # A separate env for body counts: bodies_per_card resets the env it
        # measures on, and must never touch the mirror mid-build.
        self._forecaster = SimForecaster(self.deck, engine=self.engine)
        # Warm the two lazily-measured tables now, not on the first frame that
        # needs them: the encoder's card table costs ~55 ms once, and our own
        # cards' body counts ~0.7 ms each. Paid at startup, not mid-match.
        from python_ai.models import perception_encoder as enc  # noqa: PLC0415
        enc.card_table()
        for card_id in self.deck:
            self._forecaster.bodies_per_card(card_id)

    @property
    def env(self):
        return self._env

    # -- per-card facts, measured once -----------------------------------
    def _is_spell(self, card_id: int) -> bool:
        try:
            return bool(self.engine.get_card_info(int(card_id))["is_spell"])
        except Exception:                                   # noqa: BLE001
            return False

    def _unit_hp(self, card_id: int) -> _UnitHp | None:
        """One body's full engine HP, read off the observation by the encoder's
        own card table. The observation clips at MAX_*_HP, so a card at the
        ceiling is flagged: its true maximum may be higher."""
        if card_id not in self._hp:
            from python_ai.models import perception_encoder as enc  # noqa: PLC0415
            row = enc.card_table().get(card_id)
            if row is None or row.max_hp <= 0.0:
                self._hp[card_id] = None
            else:
                self._hp[card_id] = _UnitHp(
                    max_hp=float(row.max_hp),
                    saturated=row.max_hp >= row.hp_divisor * 0.999)
        return self._hp[card_id]

    # -- the build -------------------------------------------------------
    def build(self, gs):
        gaps = MirrorGaps()
        env = self._env

        # 1. Body counts before anything touches the mirror.
        groups, gaps.unmappable_units = group_units(gs.units)
        # A spell id among perceived UNITS would be re-cast on every rebuild:
        # an explosion seen for one frame would hit the mirror's board again
        # each second. No detector class maps to a spell today; the mapping
        # table is generated, so this is enforced here rather than assumed.
        for key in [k for k in groups if self._is_spell(k[0])]:
            gaps.unmappable_units += len(groups.pop(key))
        bodies = {cid: self._forecaster.bodies_per_card(cid)
                  for cid, _team in groups}

        # 2. A deterministic reset.
        env.seed(int(self.seed))

        # 3. Towers.
        unmeasured, refused = [], []
        for name, team, slot in TOWER_SLOTS:
            tower = getattr(gs, name)
            if tower.destroyed:
                if not env.destroy_tower(team, slot):
                    refused.append(name)
                continue
            if not tower.hp_measured:
                unmeasured.append(name)
                continue
            frac = min(1.0, max(0.0, float(tower.hp_fraction)))
            if frac >= 1.0:
                continue                       # already full after the reset
            hp = frac * float(env.get_tower_max_hp(team, slot))
            if hp < 1.0 or not env.set_tower_hp(team, slot, hp):
                refused.append(name)
        gaps.towers_assumed_full = tuple(unmeasured)
        gaps.towers_refused = tuple(refused)

        # 4. The clock, so the engine runs its own elixir phase.
        env.set_current_tick(min(MATCH_TICKS,
                                 seconds_to_ticks(max(0.0, gs.seconds_elapsed))))

        # 5. Units: one injection per card, visible units already deployed.
        opp_bodies: dict[int, int] = {}
        for (card_id, team), members in groups.items():
            points = injection_points(members, bodies[card_id])
            if team == 1:
                opp_bodies[card_id] = opp_bodies.get(card_id, 0) + len(points)
            for unit in points:
                x, y = tile_centre(unit)
                hp = -1.0
                if not unit.hp_measured:
                    gaps.hp_assumed_full += 1
                elif float(unit.hp_fraction) < 1.0:
                    full = self._unit_hp(card_id)
                    if full is None or full.saturated:
                        gaps.hp_ceiling_unknown += 1
                    if full is not None:
                        hp = max(1.0, float(unit.hp_fraction) * full.max_hp)
                env.inject(card_id, x, y, team, hp=hp, deploy_ticks=0)
                gaps.injected += 1
        self.last_opp_bodies = opp_bodies

        # 6. Elixir and hand last.
        env.set_elixir_for_team(0, float(gs.my_elixir))
        if self.assume_opp_elixir is not None:
            env.set_elixir_for_team(1, float(self.assume_opp_elixir))
        gaps.opp_elixir_assumed = (float(self.assume_opp_elixir)
                                   if self.assume_opp_elixir is not None
                                   else float(env.get_elixir_for_team(1)))
        gaps.opp_deck_fabricated = not self._opp_deck_given

        gaps.hand_set, gaps.hand_reason = self._set_hand(env, gs.my_hand)
        return env, gaps

    @staticmethod
    def _set_hand(env, hand) -> tuple[bool, str]:
        """set_hand_for_team, with its bool CHECKED and read back. A refusal
        means the mirror's slot numbers are not the screen's."""
        hand = [int(c) if c is not None else -1 for c in (hand or ())]
        size = len(env.get_hand_for_team(0))
        if len(hand) != size:
            return False, f"perceived {len(hand)} cards, engine hand is {size}"
        if any(c < 0 for c in hand):
            return False, f"unreadable slot in {hand}"
        if not env.set_hand_for_team(0, hand):
            return False, f"engine refused {hand}"
        if list(env.get_hand_for_team(0)) != hand:
            return False, "engine hand does not read back"
        return True, ""
