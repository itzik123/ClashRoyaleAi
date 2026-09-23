import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import clash_royale_env

# Deliberately small (in the same spirit as AlphaStar's embeddings for choosing
# an action type): it only has to tell apart the few cards a hand slot can hold,
# not encode a card's full strategic meaning; the CNN and the scalar MLP already
# give the LSTM everything else.
CARD_EMBED_DIM = 16

# Width of the placement head's full-resolution branch (see place_hires in
# __init__). Deliberately narrow: the branch runs at the full 34x18, 12.24x the
# cells of the pooled 9x5 map, so every channel there costs about 12x a channel
# on the coarse path. 8 context channels + 8 hidden channels are enough to pick
# a cell inside a block, which is all the branch has to do -- the coarse path
# already knows roughly where.
HIRES_CTX_DIM = 8
HIRES_HIDDEN = 8

# The number of legal placement rows on our half, read live from the engine
# (like every other constant here) instead of a hard-coded copy.
# get_own_half_max_y() returns 15.0 since the river was centred on 16.5 (see
# Board.h; it used to be 15.5 with an off-centre river that gave team 0 one more
# row than team 1), i.e. whole rows 0..15 -- 16 rows either way. It needs an
# instance rather than a static attribute, so one is built here once, at module
# load.
_probe = clash_royale_env.ClashRoyaleEnv(list(range(8)), list(range(8)), 100)
OWN_HALF_MAX_Y = _probe.get_own_half_max_y()
MAX_PLACEMENT_X = _probe.get_max_placement_x()
del _probe
# The last row on our half that troops may use (get_own_half_max_y = 15.0 -> row 15)
OWN_HALF_ROWS = int(OWN_HALF_MAX_Y) + 1
# The placement head spans the WHOLE board, not only our half. The reason: the
# engine exempts spells from the half rule (GameManager::isValidPlacement checks
# only the board bounds and the dead zone for a spell), but the action space used
# to cap target_y at 15.5 for every card -- so Fireball physically could not
# cross the river, and a quarter of the deck could not be used for its purpose at
# any amount of training. The head now produces the whole board, and legality is
# enforced by a per-card mask (see placement_mask), not by shrinking the space.
PLACEMENT_ROWS = clash_royale_env.ClashRoyaleEnv.BOARD_HEIGHT

# is_spell for every card_id, read live from the engine (the get_card_info
# binding) instead of a hard-coded Python list -- exactly the kind of drift this
# project has already been burned by.
_ALL_IDS = clash_royale_env.get_all_card_ids()
NUM_CARD_IDS_LIVE = clash_royale_env.ClashRoyaleEnv.NUM_CARD_IDS


def _registered_card_ids(num_card_ids):
    """Every id the ENGINE can put in a hand -- Evolutions included.

    `get_all_card_ids()` deliberately filters the 41 Evolutions (ids 123-163)
    out, and that is right for sampling a random deck. It is WRONG for anything
    describing what a hand can hold: the engine accepts an Evolution in deck
    slots 0 and 2 and plays it normally (Evolution Archers: 242 legal cells).
    Deriving the legality table and the spell flags from the filtered list left
    every Evolution row all-False, so an Evolution in the deck was a card the
    policy could never place (audit 06, E-2). Probed through `get_card_info`,
    which raises on the gaps (16, 37, 38, ...), rather than restated.
    """
    ids = []
    for cid in range(num_card_ids):
        try:
            clash_royale_env.get_card_info(cid)
        except ValueError:
            continue
        ids.append(cid)
    return ids


_REGISTERED_IDS = _registered_card_ids(NUM_CARD_IDS_LIVE)
_spell_flags = torch.zeros(NUM_CARD_IDS_LIVE)
#: 1.0 where the card may be placed on the ENEMY half as far as the ROW rule is
#: concerned: spells, and deploy-anywhere troops (Miner, Goblin Drill). The
#: engine-derived legality table still decides the exact cells. Kept separate
#: from `_spell_flags` because "is a spell" and "ignores the own-half rule" are
#: different facts that only coincided while no deck held a Miner.
_row_free_flags = torch.zeros(NUM_CARD_IDS_LIVE)
for _cid in _REGISTERED_IDS:
    _info = clash_royale_env.get_card_info(_cid)
    if _info["is_spell"]:
        _spell_flags[_cid] = 1.0
    if _info["is_spell"] or _info.get("deploy_anywhere", False):
        _row_free_flags[_cid] = 1.0



def _build_placement_legality(num_card_ids, placement_rows, board_width):
    """(num_card_ids + 1, rows*width) bool -- what the ENGINE will accept.

    Derived from ClashRoyaleEnv.is_valid_placement, never recomputed here. The
    predicate combines board bounds, Board::isBackRowDeadZone, the per-card
    placementRadius / isSpell / deployAnywhere, and the tower footprint
    clearance; a second copy of that geometry in Python is exactly the drift
    this project has already paid for twice (map geometry, NUM_CARD_IDS).

    Returns None when the binding is absent, and placement_mask then falls
    back to the row-only mask -- i.e. the old, leaky behaviour. That is
    deliberate and LOUD: it warns, because a silent fallback would let a stale
    .pyd quietly restore a defect that was costing 58.7% of the policy's card
    choices.

    Cost is one-off: ~113k pure predicate calls at construction, no per-step
    work at all, because legality does not depend on board state (measured:
    208/288 legal cells on an empty board and 208/288 with six troops down,
    zero cells changed).
    """
    try:
        import clash_royale_env as _env
        if not hasattr(_env.ClashRoyaleEnv, "is_valid_placement"):
            raise AttributeError("is_valid_placement")
    except Exception as exc:  # noqa: BLE001 -- any import/attr failure is the same story
        warnings.warn(
            f"clash_royale_env.is_valid_placement unavailable ({exc}); the "
            "placement mask falls back to own-half rows only. Measured on the "
            "ep~45,800 checkpoint, that let 58.7% of the policy's card choices "
            "be silently rejected by playCard, whose advantages are then pure "
            "noise in the gradient. Rebuild the .pyd -- see "
            "perception/UPSTREAM_REQUESTS.md item 12.",
            RuntimeWarning, stacklevel=2)
        return None, None

    deck = [10, 1, 41, 25, 7, 2, 6, 5]
    probe = _env.ClashRoyaleEnv(deck, deck, 20000)
    probe.reset()

    cells = placement_rows * board_width
    table = torch.zeros(num_card_ids + 1, cells, dtype=torch.bool)
    # Every id a HAND can hold, Evolutions included -- see _registered_card_ids.
    known = set(_registered_card_ids(num_card_ids))
    for cid in range(num_card_ids):
        if cid not in known:
            # Unknown id: leave the row all-False. It can never be the chosen
            # card (it is not in any hand), and an all-True row would be a
            # silent claim about a card the registry does not have.
            continue
        for cell in range(cells):
            y, x = divmod(cell, board_width)
            if probe.is_valid_placement(cid, float(x), float(y), 0):
                table[cid, cell] = True
    table[num_card_ids] = True          # the permissive no-op fallback row

    # --- cells our OWN dead towers hand back (2026-08-27) -----------------
    # The premise above -- "legality does not depend on board state" -- was
    # measured against TROOPS and does NOT hold for a destroyed tower. The
    # tower's 3x3 footprint clears when it dies:
    #
    #   own LEFT princess destroyed   242 -> 251 legal cells (+9)
    #   own RIGHT princess destroyed  242 -> 251 legal cells (+9)
    #   ENEMY princess destroyed      242 -> 242             ( 0)
    #
    # so only our own towers matter, and a table cached on a full board masks
    # those nine cells off forever -- exactly the ground a player defends after
    # losing a tower.
    #
    # Probed over a BOUNDED WINDOW around each tower rather than by re-running
    # the whole board for every card in every tower state: the base pass is
    # ~113k predicate calls and 2.35s, and two more full passes would triple
    # the cost of constructing a net. The window is +/-2 cells, i.e. 25 per
    # tower against an observed 3x3 footprint, and
    # tests/test_placement_mask_after_tower_loss.py does the exhaustive
    # all-cards/all-cells comparison to prove it is wide enough -- an
    # under-sized window would otherwise be a silent mask divergence.
    freed = torch.zeros(2, num_card_ids + 1, cells, dtype=torch.bool)
    centres = _own_princess_centres()
    for slot_i, (cx, cy) in enumerate(centres):
        probe2 = _env.ClashRoyaleEnv(deck, deck, 20000)
        probe2.reset()
        if not probe2.destroy_tower(0, slot_i + 1):   # slots 1=LEFT, 2=RIGHT
            continue
        window = [(x, y)
                  for y in range(max(0, cy - 2), min(placement_rows, cy + 3))
                  for x in range(max(0, cx - 2), min(board_width, cx + 3))]
        for cid in range(num_card_ids):
            if cid not in known:
                continue
            for x, y in window:
                cell = y * board_width + x
                if table[cid, cell]:
                    continue                      # already legal, no delta
                if probe2.is_valid_placement(cid, float(x), float(y), 0):
                    freed[slot_i, cid, cell] = True
    return table, freed


def _own_princess_centres():
    """[(x, y), ...] for team 0's LEFT and RIGHT Princess Towers, as CELLS.

    Derived from the bound ArenaLayout rather than restated -- CLAUDE.md's
    no-second-copies rule, and this geometry has already gone stale twice.
    """
    import clash_royale_env as _env
    y = int(_env.arena_princess_y(0))
    return [(int(_env.ARENA_LEFT_LANE_X), y),
            (int(_env.ARENA_RIGHT_LANE_X), y)]


#: Dilations of the context blocks appended to the CNN trunk, in order.
#:
#: They run on the POOLED 9x5 map, where one cell is worth 4 input cells (two
#: stride-2 pools), so a 3x3 kernel at dilation d reaches 8d further in INPUT
#: rows. The base trunk measures 10x10 by gradient, so 10 + 16 + 16 = 42 >= 34
#: covers the whole board -- bought at the price of two 9x5 convolutions, the
#: cheapest resolution on the board at which to buy reach.
#:
#: WHY REACH WAS NEEDED. A bridge sits at y=16.5 and the enemy King at y=30.5,
#: 14 rows apart; the enemy Princess Towers at y=27.0, 10.5 rows. At a 10-row
#: field no single convolutional feature could relate "their win condition just
#: crossed" to "this is the tower it is walking at" -- and the placement head
#: reads the spatial map DIRECTLY, so that limit reached the ACTION, not just
#: the representation.
#:
#: WHY NOT WIDTH. Measured in test_net_trunk_receptive_field.py: doubling the
#: channels costs 2.28x and moves the receptive field by exactly zero. Width
#: and reach are orthogonal axes, and this problem is on the reach axis.
#:
#: WHY (2,2) AND NOT (2,4) -- MEASURED, AND THE ONE COUNTERINTUITIVE BIT.
#: A single d=4 block reaches as far as two d=2 blocks and has the same
#: parameter count, so it looks like the cheaper way to buy 32 rows. It is not.
#: Isolated fwd+bwd at the real update shape (500, 32, 9, 5):
#:
#:     identity          0.97 ms          bottleneck d=1     7.30 ms
#:     bottleneck d=2    8.26 ms          bottleneck d=4    43.72 ms
#:
#: 5.3x for the SAME parameters. `padding=dilation` on a 9x5 map means d=4 pads
#: to 17x13 -- 221 cells against 45 -- so the convolution spends ~80% of its
#: work on padding. Dilation is cheap only while the dilation is small relative
#: to the map it runs on, and a 9x5 map is small. Two d=2 blocks reach the same
#: 32 rows for 14.6 ms instead of 50.1 ms.
#:
#: Both are inside budget end-to-end (3-5% of an update chunk, against a ~1%
#: noise floor measured arm-against-itself), so this was decided on the
#: isolated measurement, where the 6x difference is far above noise.
CONTEXT_DILATIONS = (2, 2)

#: Bottleneck width inside a context block. 32 -> 16 -> 16 -> 32, so the only
#: 3x3 convolution runs at half the trunk's channel count and the block costs
#: ~2.7k parameters instead of the ~9.2k a plain 32->32 dilated conv would.
CONTEXT_BOTTLENECK = 16


class DilatedContextBlock(nn.Module):
    """Residual dilated bottleneck: `x + expand(relu(spread(relu(reduce(x)))))`.

    THE FINAL 1x1 IS ZERO-INITIALISED, which is the whole design. At
    initialisation the block computes exactly `x`, so appending it to a trained
    trunk leaves every existing feature bit-identical and the change is
    strictly additive -- the net starts where it was and learns to use the new
    reach. `place_hires[-1]` in this same file already uses that device.

    The consequence that trips up measurement, pinned in
    tests/test_net_dilated_context.py: a zero gate emits zero gradient too, so
    a gradient-measured receptive field on a FRESH net truthfully reports the
    identity path's 10x10. Activate the gate before measuring the STRUCTURAL
    field. That is the same class of trap as measuring a receptive field on a
    zeros input and getting whichever ReLU channels happened to be open.

    Shape-preserving by construction (`padding=dilation` on a 3x3 kernel), so
    `cnn_out_dim` and therefore `lstm_input_dim` do not move -- the 1.8M-
    parameter LSTM, 96% of the net, keeps its shape and its checkpoint.
    """

    def __init__(self, channels, bottleneck, dilation):
        super().__init__()
        self.channels = channels
        self.dilation = dilation
        self.reduce = nn.Conv2d(channels, bottleneck, kernel_size=1)
        self.spread = nn.Conv2d(bottleneck, bottleneck, kernel_size=3,
                                padding=dilation, dilation=dilation)
        self.expand = nn.Conv2d(bottleneck, channels, kernel_size=1)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, x):
        h = F.relu(self.reduce(x))
        h = F.relu(self.spread(h))
        return x + self.expand(h)


#: Widths of the scalar encoder's four semantic branches. They MUST sum to
#: SCALAR_FEATURE_DIM: that sum is `lstm_input_dim - cnn_out_dim`, and moving it
#: reshapes `LSTMCell(1504, 256)` -- 1,804,288 parameters, 96% of the net, all
#: discarded on the next load. Retuning the split is free; changing the total is
#: not, and should be a deliberate decision rather than a side effect.
#:
#: `extra` is an EXPANSION (9 -> 12), not a compression. Those nine floats are
#: the match clock, both players' cumulative spend and six tower HPs -- they
#: decide who is winning -- and in the monolithic layer they shared all 64
#: outputs with 740 sparse one-hot dims.
ECON_BRANCH_DIM = 8
HAND_SLOT_EMBED_DIM = 5      # x hand_size (4) = 20
EXTRA_BRANCH_DIM = 12
CYCLE_BRANCH_DIM = 24
SCALAR_FEATURE_DIM = 64

#: Card-identity width inside the cycle branch. `seen` and `recency` index the
#: same card space, so ONE shared projection serves both: `seen @ W` is the sum
#: of the embeddings of the cards the opponent has shown, `recency @ W` the same
#: sum weighted by how recently. 2,960 parameters against a dense 370-wide
#: layer's 8,880, and the two blocks cannot learn two different notions of what
#: a card is.
CYCLE_EMBED_DIM = 16


class ScalarEncoder(nn.Module):
    """The scalar half of the observation, encoded in four independent branches.

    WHAT THIS REPLACES AND WHY. `nn.Linear(1124, 64)`. Every one of its 64
    outputs was a row spanning all 1124 inputs, so the columns carrying the
    opponent's card cycle shared their outputs with the 740 hand one-hot
    columns -- and a gradient step taken to improve hand encoding rewrote the
    same rows the cycle is read through. The cycle was not compressed away so
    much as continuously perturbed by other objectives' learning.

    Note what the defect is NOT. "17.6:1 compression destroys the cycle" does
    not survive the mathematics: a random projection of 370 dims into 64
    preserves pairwise structure well (Johnson-Lindenstrauss), so at
    initialisation the information is largely intact. The problem is an
    optimisation one, which is why the test that pins this takes a real
    optimizer step rather than measuring reconstruction error.

    Branches are concatenated in a FIXED order with the cycle LAST, so
    `cycle_slice` names a contiguous block that other code (and the tests) can
    address. Offsets into the observation are taken from the engine's own
    bound values, never restated -- the last thing to restate this layout read
    card-recency floats where it expected tower HP.
    """

    def __init__(self, hand_size, num_card_ids, num_extra_scalars,
                 cycle_block_size, extra_start, cycle_start):
        super().__init__()
        self.hand_size = hand_size
        self.num_card_ids = num_card_ids
        self.num_extra_scalars = num_extra_scalars
        self.cycle_block_size = cycle_block_size
        self.extra_start = extra_start
        self.cycle_start = cycle_start
        self.onehot_start = 1 + hand_size

        hand_dim = hand_size * HAND_SLOT_EMBED_DIM
        self.out_dim = (ECON_BRANCH_DIM + hand_dim
                        + EXTRA_BRANCH_DIM + CYCLE_BRANCH_DIM)
        if self.out_dim != SCALAR_FEATURE_DIM:
            raise ValueError(
                f"scalar branches sum to {self.out_dim}, not "
                f"{SCALAR_FEATURE_DIM}; that changes lstm_input_dim and "
                "discards the LSTM's 1.8M trained parameters. Retune the "
                "branch widths so they still sum to the total, or change the "
                "total deliberately and say so.")

        # elixir + the hand's costs. Affordability is a JOINT function of the
        # two, so they share a branch rather than being split apart.
        self.econ = nn.Linear(1 + hand_size, ECON_BRANCH_DIM)
        # ONE projection reused across hand slots, not a dense layer over all
        # 740 dims: slot i and slot j hold the same kind of thing (a card), and
        # a dense layer would have to learn that four separate times.
        self.hand_slot = nn.Linear(num_card_ids, HAND_SLOT_EMBED_DIM)
        self.extra = nn.Linear(num_extra_scalars, EXTRA_BRANCH_DIM)
        self.cycle_card = nn.Linear(num_card_ids, CYCLE_EMBED_DIM, bias=False)
        self.cycle_out = nn.Linear(2 * CYCLE_EMBED_DIM, CYCLE_BRANCH_DIM)

        self.cycle_slice = (self.out_dim - CYCLE_BRANCH_DIM, self.out_dim)

    def forward(self, scalar):
        econ = scalar[:, :self.onehot_start]
        onehots = scalar[:, self.onehot_start:
                         self.onehot_start + self.hand_size * self.num_card_ids]
        extra = scalar[:, self.extra_start:
                       self.extra_start + self.num_extra_scalars]
        cycle = scalar[:, self.cycle_start:
                       self.cycle_start + self.cycle_block_size]

        e = F.relu(self.econ(econ))
        h = F.relu(self.hand_slot(
            onehots.reshape(-1, self.hand_size, self.num_card_ids))).flatten(1)
        x = F.relu(self.extra(extra))
        # DETACHED, and this is the whole of the 2026-08-28 fix. See
        # `cycle_branch` below for the measurement; the short version is that
        # PPO was winning a 248:1 gradient fight for these parameters and
        # spending them on a one-dimensional opponent-tempo readout, so the
        # branch's own output decoded the opponent's next card WORSE after
        # training than at random init. Detaching here means the actor and the
        # critic READ the branch and can never RESHAPE it; `cycle_id_head` is
        # now its only gradient, and that one asks for card identity.
        #
        # Note what is NOT detached: `lstm.weight_ih`, `place_ctx`, every head.
        # They keep learning to USE these 24 dims, at full gradient. Only the
        # definition of the 24 dims is protected.
        return torch.cat([e, h, x, self.cycle_branch(cycle).detach()], dim=1)

    def cycle_branch(self, cycle):
        """The cycle branch WITH its gradient attached. (B, CYCLE_BRANCH_DIM).

        Separated from `forward` because the two callers want opposite things:
        `forward` feeds the LSTM and must detach (see above), while the
        auxiliary identity loss needs the live graph. Same parameters, called
        twice -- a recompute of 3,752 parameters on the minibatch, not a second
        copy of the definition.

        WHY THIS EXISTS, measured 2026-08-28 on `model_weights_phase4.pth` at
        ep ~7,200 (24 episodes, 5,366 decisions, linear probes with l2 swept
        per representation), as lift over the marginal on "which card does the
        opponent play NEXT":

            obs cycle_raw (the ceiling)   +0.412
            this branch, PPO-trained      +0.169     <- 41% retained
            this branch, at random init   +0.195     <- training made it WORSE
            hx, PPO-trained               +0.046     <- 11% retained
            hx, at random init            +0.135
            everything EXCEPT the cycle   +0.031     <- not shortcut-solvable

        The cause is that `cycle_card` is a SHARED projection over a SUM-POOLED
        bag of cards, so the only thing the dominant objective can cheaply
        extract is aggregate opponent activity -- which is 1-D, and optimising
        for it pulls the eight deck columns onto a common direction. Measured:
        mean pairwise |cos| between them went 0.191 +- 0.017 at init to 0.461
        (~16 sigma), effective rank 7.48 -> 6.01, and the branch output's PC1
        went 32% -> 54% of variance while its column norms GREW 2.1x. It was
        not neglected. It was re-tasked.

        It is not a capacity limit -- `Linear(370, 8) + ReLU` trained FOR this
        task keeps 92% of the ceiling, and 24 dims keeps 96%, so widening this
        branch is the wrong instinct. And it is fully reversible: handed an
        identity gradient, these exact collapsed weights recover +0.390 of the
        +0.412 ceiling and the columns un-align (|cos| back to 0.202).
        """
        c = cycle.reshape(-1, 2, self.num_card_ids)
        c = F.relu(self.cycle_card(c)).flatten(1)
        return F.relu(self.cycle_out(c))


class MicroRoyaleNet(nn.Module):
    # 21 observation channels: 0-3 our units (melee/ranged/tank/buildings), 4-7 the
    # same for the opponent, 8 river/bridges, 9-20 per-team attribute channels
    # (indexed through the bound CH_* constants).

    # The recurrent width, declared ONCE here because it is the shape every
    # caller needs before it has a net: a fresh (hx, cx) is zeros of this size,
    # and every harness that steps the policy manually builds one. It used to be
    # typed as a bare `256` in five separate files, which is the same duplicated-
    # constant failure CLAUDE.md forbids for engine constants -- `policy_io`
    # re-exports this attribute so nothing has to repeat the literal.
    LSTM_HIDDEN = 256

    def __init__(self, channels=None, board_width=None, board_height=None, hand_size=None, num_card_ids=None,
                 placement_rows=None, num_ability_slots=0,
                 context_dilations=CONTEXT_DILATIONS,
                 branched_scalars=True):
        super(MicroRoyaleNet, self).__init__()

        # Defaults are read live from the compiled engine (not hard-coded), so any
        # change to the board size or the card count on the C++ side propagates
        # here automatically, with no matching manual edit in this file. This is
        # exactly the kind of drift that crashed training runs in this project
        # more than once before this fix.
        channels = channels if channels is not None else clash_royale_env.ClashRoyaleEnv.NUM_CHANNELS
        board_width = board_width if board_width is not None else clash_royale_env.ClashRoyaleEnv.BOARD_WIDTH
        board_height = board_height if board_height is not None else clash_royale_env.ClashRoyaleEnv.BOARD_HEIGHT
        hand_size = hand_size if hand_size is not None else clash_royale_env.ClashRoyaleEnv.HAND_SIZE
        num_card_ids = num_card_ids if num_card_ids is not None else clash_royale_env.ClashRoyaleEnv.NUM_CARD_IDS

        self.channels = channels
        self.board_width = board_width
        self.board_height = board_height
        self.hand_size = hand_size
        self.num_card_ids = num_card_ids

        # The placement head is categorical over whole board cells (see place_ctx /
        # place_up below). placement_rows = the full board height; actual legality
        # is enforced by a per-card mask (placement_mask), because it differs
        # between a troop and a spell.
        self.placement_rows = placement_rows if placement_rows is not None else PLACEMENT_ROWS
        self.placement_cells = self.placement_rows * board_width
        # Rows allowed for an ordinary troop (not a spell, not deploy-anywhere): our half only.
        self.own_half_rows = min(OWN_HALF_ROWS, self.placement_rows)
        # (num_card_ids,) -- 1.0 if the card is a spell. A buffer, not a parameter:
        # it is a fact about the engine, not something learned, but it has to move
        # to the device together with the network.
        self.register_buffer("spell_flags", _spell_flags[:num_card_ids].clone())
        # (num_card_ids,) -- 1.0 where the own-half ROW rule does not apply
        # (spells AND deploy-anywhere troops). This, not spell_flags, is what
        # placement_mask reads. persistent=False for the same reason as the
        # legality tables below: it is a fact about the engine, and a checkpoint
        # copy of it could restore the pre-2026-09-15 values that confined a
        # Miner to its own half.
        self.register_buffer("row_free_flags",
                             _row_free_flags[:num_card_ids].clone(),
                             persistent=False)

        # (num_card_ids + 1, placement_cells) bool -- which cells the engine
        # actually accepts for each card. The last row is a permissive fallback
        # for the no-op.
        #
        # Built once, because it is static: measured on our 288-cell half, 208
        # cells were legal on an empty board and exactly 208 with six troops on
        # it -- zero cells changed. So it is a fixed table rather than a runtime
        # query, and it costs nothing per step. (The one change a match can make,
        # one of our Princess Towers dying, is handled by the variants below.)
        #
        # A buffer rather than a parameter, and persistent=False: it is a fact
        # about the engine, not a learned weight, and saving it in a checkpoint
        # would turn it into a second copy that can go stale against the engine
        # -- exactly the drift the table exists to prevent.
        _legal_table, _freed_table = _build_placement_legality(
            num_card_ids, self.placement_rows, board_width)
        self.register_buffer("_placement_legal", _legal_table,
                             persistent=False)
        #: (2, num_card_ids+1, cells) -- cells that become legal when our own
        #: LEFT/RIGHT Princess dies. Same persistent=False reasoning.
        self.register_buffer("_placement_freed", _freed_table,
                             persistent=False)
        #: Cell index of each own Princess centre, for reading its liveness out
        #: of the observation. Built once; cheap.
        # --- legality folded into ONE table indexed by tower state ----------
        # Four variants -- both Princesses alive / left dead / right dead /
        # both dead -- each already OR-ed with the base table. The mask then
        # does a single gather instead of a base gather plus one gather and one
        # OR per tower, which is what made the first version of this feature
        # 53% more expensive than the base mask. Built by pure tensor ORs, so it
        # costs no extra engine probing; 4 x 186 x 612 bools is ~455 KB.
        if _freed_table is not None:
            _by_state = torch.stack([
                _legal_table,                                          # 0: both alive
                _legal_table | _freed_table[0],                        # 1: LEFT dead
                _legal_table | _freed_table[1],                        # 2: RIGHT dead
                _legal_table | _freed_table[0] | _freed_table[1],      # 3: both dead
            ])
        else:
            _by_state = None
        self.register_buffer("_placement_legal_by_state", _by_state,
                             persistent=False)

        self._own_princess_cells = (
            [y * board_width + x for x, y in _own_princess_centres()]
            if _freed_table is not None else [])
        #: The same two centres as FLAT indices into the observation vector.
        #: The spatial half is channel-major, so channel 3 (ally buildings --
        #: the index rewards.shaping.building_hp_end reads) starts at
        #: 3*H*W. Precomputed so placement_mask can read two scalars instead of
        #: reshaping the whole spatial block on every call.
        _ch_ally_buildings = 3
        self._own_princess_flat = [
            _ch_ally_buildings * self.board_height * board_width + c
            for c in self._own_princess_cells]

        # 0 = the deck has no Champion, so there are no ability heads at all. This
        # is not a cosmetic optimisation: without a Champion those two heads
        # sampled pure noise every tick -- they added variance to the PPO ratio
        # (total_logprob) and contributed up to 2*log(2) = 1.386 to the entropy
        # bonus, i.e. the trainer spent effort keeping a coin flip random.
        # Measured: DEFAULT_DECK has no Champion, and is_champion_ability_ready
        # returned False for both slots for the whole match. The trainer passes
        # the value explicitly, from the deck in use.
        self.num_ability_slots = num_ability_slots

        # Size of the flattened spatial block ClashEnv produces
        self.spatial_size = channels * board_height * board_width
        # The scalar part: elixir + hand_size costs + hand_size one-hots of card
        # identity (num_card_ids values each) + the tail ClashEnv appends
        # (NUM_EXTRA_SCALARS): the time fraction, the elixir spent by each side,
        # the six tower HPs, and the elixir-phase multiplier.
        #
        # The tail is APPENDED, not pushed into the middle, and that is not
        # arbitrary: all the code that reads this vector by offset
        # (affordability_mask, hand_card_ids, placement_mask here, and the
        # scripted opponents) measures from the start of the scalar section.
        # Appending keeps every one of those offsets valid unchanged; inserting
        # in the middle would have broken them all silently -- no exception, just
        # a policy reading the wrong numbers.
        self.num_extra_scalars = clash_royale_env.ClashRoyaleEnv.NUM_EXTRA_SCALARS
        # Two num_card_ids-wide blocks holding the opponent's card cycle (item 24,
        # 2026-08-27): seen[] and recency[]. Derived from the binding rather than
        # written here as a number -- 2*185 in Python would be exactly the second
        # copy the rule at the top of CLAUDE.md forbids. Appended AFTER the tail,
        # so extra_start and every offset before it stay exactly as valid as they
        # were.
        self.cycle_block_size = clash_royale_env.ClashRoyaleEnv.CYCLE_BLOCK_SIZE
        self.scalar_size = (1 + hand_size + hand_size * num_card_ids
                            + self.num_extra_scalars + self.cycle_block_size)
        # The offset (within scalar_obs) where the tail starts -- needed by ScalarEncoder and by diagnostics.
        self.extra_start = 1 + hand_size + hand_size * num_card_ids
        # ...and the start of the cycle blocks, right after the tail. Both offsets
        # are measured FORWARD from the start of the scalar section, never
        # backward from its end: a backward offset breaks silently every time
        # something is appended, which is exactly what happened here.
        self.cycle_start = self.extra_start + self.num_extra_scalars

        # ==========================================
        # 1. Spatial feature extraction (CNN)
        # Input: (Batch, NUM_CHANNELS, 34, 18)
        # ==========================================
        # ceil_mode=True on both MaxPools: board_height=34 does not divide cleanly
        # by 4 (34 -> 17 -> 8 with ordinary floor, which would delete a whole row
        # -- exactly the new back row next to the King Tower, which was the whole
        # point of that change). With ceil_mode no row or column is silently
        # dropped; the spatial map is only slightly larger.
        # Split into trunk + flatten (instead of one Sequential ending in Flatten):
        # the spatial feature map BEFORE flattening is the input of the
        # convolutional placement head (see placement_given_card). It is the same
        # tensor, not an extra computation -- the CNN still runs only once.
        self.cnn_trunk = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )
        # Appended at the END, not in the middle, and that is not style: two
        # callers slice the trunk by index (extract_features_hires and
        # hires_features take [:2] for the full-resolution branch and [2:] for the
        # rest). Appending keeps both slices exactly valid; inserting at the start
        # would hand the full-resolution branch a completely different tensor,
        # with no error anywhere.
        self.context_dilations = tuple(context_dilations)
        for dilation in self.context_dilations:
            self.cnn_trunk.append(
                DilatedContextBlock(32, CONTEXT_BOTTLENECK, dilation))
        self.cnn_flatten = nn.Flatten()

        # Output size of the CNN after pooling (ceil twice, matching ceil_mode=True above)
        pooled_h = math.ceil(math.ceil(board_height / 2) / 2)
        pooled_w = math.ceil(math.ceil(board_width / 2) / 2)
        self.pooled_h, self.pooled_w = pooled_h, pooled_w
        self.cnn_out_dim = 32 * pooled_h * pooled_w

        # ==========================================
        # 2. Scalar feature extraction (MLP)
        # Input: elixir, hand costs and card identities, the extra scalars, and the cycle blocks
        # ==========================================
        # Four semantic branches instead of a single Linear(scalar_size, 64). The
        # name (`scalar_mlp`) is kept deliberately: expert_distill.TRUNK_MODULES
        # finds modules by name, and renaming it would have disconnected it
        # silently. See ScalarEncoder.
        if branched_scalars:
            self.scalar_mlp = ScalarEncoder(
                hand_size, num_card_ids, self.num_extra_scalars,
                self.cycle_block_size, self.extra_start, self.cycle_start)
            scalar_feature_dim = self.scalar_mlp.out_dim
        else:
            # The monolithic layer this replaced. Kept so the A/B can always be
            # run (just like collision_bench.cpp keeps both paths), and above all
            # so the test showing that it FAILS the non-interference check can
            # actually run it rather than describe it.
            self.scalar_mlp = nn.Sequential(
                nn.Linear(self.scalar_size, SCALAR_FEATURE_DIM),
                nn.ReLU()
            )
            scalar_feature_dim = SCALAR_FEATURE_DIM
        self.scalar_feature_dim = scalar_feature_dim

        # --- the cycle skip (2026-08-28) ------------------------------------
        # Width of the block that bypasses the LSTM and reaches the heads
        # directly. 0 on the monolithic `branched_scalars=False` path, which
        # has no cycle branch at all -- and that zero must stay a real zero
        # rather than a slice, because `t[..., -0:]` is the WHOLE tensor, not
        # an empty one. Everything downstream goes through `_split_cycle` /
        # `_head_input` for exactly that reason.
        #
        # WHY the skip: even a healthy branch only reaches the actor and the
        # critic THROUGH the LSTM, where it is 24 of 1504 input dims (1.6%) and
        # is subject to the same pressure that collapsed it. Measured, the
        # recurrence retains 27% of the branch's card signal when trained
        # (+0.169 -> +0.046) against 69% untrained (+0.195 -> +0.135) -- so it
        # is a second lossy stage, in the same direction, for the same reason.
        # Routing the block straight to the heads makes delivery unconditional:
        # +0.169 instead of +0.046 today, and +0.390 instead of +0.046 once
        # `cycle_id_head` has repaired the branch.
        #
        # `lstm_input_dim` is deliberately NOT touched -- the block still
        # occupies its 24 slots in `features`, so LSTMCell(1504, 256) keeps its
        # shape and all 1,804,288 of its trained parameters warm-start.
        self.cycle_feature_dim = (CYCLE_BRANCH_DIM if branched_scalars else 0)
        head_dim = self.LSTM_HIDDEN + self.cycle_feature_dim

        # ==========================================
        # 2b. Card-identity embedding -- for autoregressive placement (see placement_given_card)
        # ==========================================
        # Works directly on the one-hot already in obs, not on the slot index:
        # slots rotate, so the same slot is a different physical card every
        # cycle, and an embedding of the index could never learn "Fireball on
        # their cluster, Hog at the bridge". nn.Linear on a true one-hot vector is
        # mathematically identical to an ordinary embedding lookup, without a
        # detour through argmax.
        self.card_id_embed = nn.Linear(num_card_ids, CARD_EMBED_DIM, bias=False)
        # A dedicated embedding for the no-op (card_index == hand_size, no real
        # card in that slot). The engine ignores target_x/target_y entirely when
        # card_index is outside [0, hand_size) (see the matching comment in
        # gym_wrapper.py), so this exists only to keep a well-defined placement
        # distribution on every tick for the loss -- a separate learned
        # parameter, so gradient from no-op timesteps never contaminates the
        # learning of any real card-conditioned placement.
        self.noop_embed = nn.Parameter(torch.zeros(CARD_EMBED_DIM))

        # ==========================================
        # 3. Memory layer (LSTM)
        # ==========================================
        self.lstm_input_dim = self.cnn_out_dim + scalar_feature_dim
        # LSTMCell rather than LSTM, so the environment loop steps each tick by hand
        self.lstm = nn.LSTMCell(self.lstm_input_dim, self.LSTM_HIDDEN)

        # ==========================================
        # 4. Action heads (actor)
        # ==========================================
        # a. The card-choice head (categorical). Like the value head it reads hx
        # (plus the cycle skip), never the card about to be chosen. hand_size hand
        # slots + one extra action = the no-op (wait / bank elixir). The engine
        # ignores a cardIndex outside [0, hand_size), so no C++ change was needed.
        self.card_head = nn.Linear(head_dim, hand_size + 1)

        # b. The spatial placement head -- categorical over whole board cells, not
        # Gaussian. Autoregressive: conditioned on hx AND on the chosen card's
        # embedding.
        #
        # Why it was replaced: the previous version sampled from a Normal with a
        # learned placement_log_std. A Gaussian's entropy is log(sigma) + const,
        # so the gradient of the entropy bonus with respect to log_std is EXACTLY
        # 1, constant and data-independent -- a constant force pushing sigma up,
        # which the noisy policy gradient loses to. Measured on the checkpoints:
        # log_std started at -2.0 and climbed to -1.86 after 68,515 episodes
        # (sigma GREW instead of shrinking), i.e. sigma = 0.156 in normalised
        # units = 2.66 tiles of standard deviation in x and 2.46 in y. The two
        # bridges are 10 tiles apart, so the agent's own noise was half the
        # distance it had to tell apart -- it simply could not aim for a lane. A
        # categorical distribution fixes this at the root: its entropy is bounded
        # above by log(placement_cells) and falls naturally as the policy
        # sharpens, placement is exact to the cell, and illegal cells can be
        # masked.
        #
        # And what was replaced AFTER that: the previous version was
        #     nn.Linear(256 + CARD_EMBED_DIM, placement_cells)   # 272 -> 612
        # i.e. a single dense layer producing a logit map over the board from a
        # vector that had already been through two MaxPools and been flattened.
        # Such a layer has no spatial structure: it must MEMORISE, with a separate
        # weight, what each of the 612 cells means, and nothing is shared between
        # cell (5,7) and its neighbour (5,8) -- even after the net learns "place
        # near the left bridge", none of it carries over to the adjacent cell.
        #
        # Instead: take the CNN's spatial feature map (B,32,9,5), add the context
        # (hx + card identity) broadcast over every cell, and upsample back to
        # 34x18 (see place_up below). Each cell's logit is then computed by the
        # same shared weights acting on the local features OF THAT BOARD REGION
        # -- exactly the right inductive bias for a game whose decision is
        # "where". The same pattern AlphaStar uses to produce spatial arguments.
        self.place_ctx = nn.Linear(head_dim + CARD_EMBED_DIM, 32)
        # RESIZE + CONV, not ConvTranspose. Fixed on 2026-08-09, after it was
        # measured that the previous version --
        #     ConvTranspose2d(32,32,k=2,s=2) -> ReLU -> ConvTranspose2d(32,16,k=2,s=2)
        # -- produces a FIXED PERIODIC BIAS on the logit map, identical for every
        # card and every game state.
        #
        # Why it happens: ctx is added by broadcast, i.e. it is SPATIALLY UNIFORM.
        # A ConvTranspose2d with kernel_size=2, stride=2 has a different weight for
        # each of the 4 positions in an output block, so a uniform input does NOT
        # produce a uniform output -- it produces a pattern with period 2, and two
        # such layers give period 4. That pattern is a property of the weights
        # alone; the card can only shift the whole map by a constant, it cannot
        # change which cell WITHIN the period wins.
        #
        # Measured on the ep~129k checkpoint: 75.0% of the variance of the logit
        # map is explained by (x mod 4, y mod 4) alone, and in behaviour 73.0% of
        # all placements landed on x = 3 (mod 4) against a null of 22.2%
        # (chi^2 = 94.8, 3 df), 28.9% on the two cells (11,2)/(11,3), and only 91
        # of the 288 legal cells were ever used. This was wrongly blamed on how
        # the reward priced buildings (the "Cannon pathology"), but the
        # concentration was CARD-INDEPENDENT -- Giant 44.4%, Valkyrie 38.9%,
        # Cannon 27.1% -- and a reward asymmetry that concerns buildings cannot
        # explain a Giant walking behind its own King Tower.
        #
        # Upsample(nearest) + Conv2d(stride=1) fixes it at the root: a spatially
        # uniform input stays uniform after both operations (except at the
        # padding edges), because every output cell is computed with exactly the
        # same weights. Odena, Dumoulin & Olah,
        # "Deconvolution and Checkerboard Artifacts" (2016).
        #
        # Channels shrink 32->16->8, not 32->32->16: a 3x3 conv at full resolution
        # costs far more than a ConvTranspose with k=2, and the placement head
        # already takes 41% of update time. The narrowing keeps the cost close to
        # the original -- see the benchmark numbers in this change's git log.
        self.place_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),            # 9x5 -> 18x10
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),            # 18x10 -> 36x20
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),    # -> (B,1,36,20)
        )
        # ==========================================
        # 4c. Residual full-resolution branch (2026-08-14)
        # ==========================================
        # What was missing: place_up reads a 9x5 map for a 34x18 board, so one
        # pooled cell covers ~4x4 board tiles, and the context (hx + card) enters
        # as a SPATIALLY UNIFORM vector. So the head's spatial vocabulary is
        # blocks of 4 tiles: the card can shift the whole map by a constant, but
        # WHICH TILE INSIDE THE BLOCK WINS is fixed by weights shared across all
        # states. That is a REPRESENTATION limit, not a training failure -- and it
        # is exactly what distill_tactics.py measured on 2026-08-14: cross-entropy
        # against the advisor's exact cell fell 180.9 -> 21.4 while the exact
        # (argmax) match stayed at 0.0%. A loss that falls while the argmax never
        # moves is the signature of a target the head cannot express.
        #
        # The branch here reads the trunk's activation BEFORE pooling (16x34x18),
        # at single-tile resolution, conditioned on exactly the same context, and
        # is added to the coarse logits as a residual.
        #
        # Why the last convolution is zero-initialised: the handoff proposed
        # concatenating into place_up, which changes its shape and therefore
        # THROWS AWAY the trained placement head from every checkpoint (exactly
        # the price the 2026-08-09 checkerboard fix had to pay). That price is
        # unnecessary: a zero-initialised residual branch computes an IDENTICAL
        # function at initialisation, so an existing checkpoint loads and behaves
        # bit-identically, the cards that work today keep working, and only the
        # genuinely new parameters start from zero. Gradient still flows: the
        # zeroed layer itself gets a non-zero gradient (it sees live
        # activations), so it leaves zero on the first step and the layer below
        # starts learning on the second. Standard zero-conv behaviour.
        self.place_ctx_hi = nn.Linear(head_dim + CARD_EMBED_DIM, HIRES_CTX_DIM)
        self.place_hires = nn.Sequential(
            nn.Conv2d(16 + HIRES_CTX_DIM, HIRES_HIDDEN,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(HIRES_HIDDEN, 1, kernel_size=3, stride=1, padding=1),
        )
        nn.init.zeros_(self.place_hires[-1].weight)
        nn.init.zeros_(self.place_hires[-1].bias)

        # The upsampling produces 4*pooled_h x 4*pooled_w, which is >= the board
        # size because pooling used ceil_mode. Crop back to the top-left corner:
        # since both pools are stride 2 with ceil, pooled cell i covers originals
        # 2i and 2i+1 at every level, so index j in the upsampled map aligns
        # exactly with row/column j of the original board. The crop is an
        # alignment, not an approximation.
        assert self.placement_rows == board_height, (
            "the convolutional placement head builds a board-sized map and crops it; "
            "a placement_rows different from board_height would break that alignment")

        # ==========================================
        # 5. State-value head (critic) -- reads hx (plus the cycle skip), never the chosen card.
        # ==========================================
        self.value_head = nn.Linear(head_dim, 1)

        # ==========================================
        # 6. Champion ability heads (up to 2 Champions at once -- see
        # activate_ability_slot1/2 in gym_wrapper.py, and
        # CardRegistry::validateDeckSlots on the C++ side). Each has 2 logits
        # (0 = do not activate, 1 = activate).
        #
        # Created ONLY if num_ability_slots > 0 -- see the note on that field
        # above. With a Champion-less deck they were pure noise; now they simply
        # do not exist (not even in the state_dict), so there are no dead
        # parameters and no contribution to logprob/entropy.
        # ==========================================
        self.ability_slot1_head = nn.Linear(256, 2) if num_ability_slots >= 1 else None
        self.ability_slot2_head = nn.Linear(256, 2) if num_ability_slots >= 2 else None

        # ==========================================
        # 7. Auxiliary head: the opponent's next card
        # ==========================================
        # Like the auxiliary tasks in UNREAL and IMPALA, this head plays no part
        # in choosing an action: its only job is its loss, which shapes what the
        # LSTM state keeps rather than what the policy does.
        #
        # AUXILIARY TASK: which card does the OPPONENT play next?
        #
        # This replaced an opponent-ELIXIR regression head on 2026-08-28, and
        # the reason is that the old task was measurably not a task at all.
        # Opponent elixir is an affine function of two scalars the observation
        # ALREADY CARRIES: `elixir(t) = start + rate*t - spent(t)`, where t is
        # extra-scalar 0 and spent(t) is extra-scalar 2. Ordinary least squares
        # on those two -- four parameters, no recurrence -- scores MAE 0.0000
        # over 2,606 samples, while the trained head sat at 0.77. So the head
        # was not modelling the opponent, it was failing at arithmetic on two
        # present inputs, and it exerted essentially no representational
        # pressure on the LSTM (python_ai/tests/test_aux_task_is_not_a_memory_
        # probe.py pins that measurement).
        #
        # Next-card is the task that cannot be solved that way: it needs the
        # opponent's PLAY HISTORY, which lives only in the recurrent state and
        # in the item-24 cycle channels. Measured 2026-08-28 by
        # eval/probe_card_counting.py, the trained hx decoded the next card at
        # +0.013 lift over a random projection at ep 1522 and -0.025 at ep
        # 2054 -- i.e. training was DISCARDING cycle information, because
        # nothing in the objective asked for it. This head is that ask.
        #
        # NUM_CARD_IDS-wide rather than 8-wide over the opponent's deck: the
        # deck is not known in `random_opponent` or in deployment, and the
        # cycle observation block is already NUM_CARD_IDS-wide, so this keeps
        # one card space across the whole net. 256*185 = 47k params, 2.5% of
        # the net.
        #
        # Still deliberately a separate method rather than an extra output of
        # step_lstm_and_card: needed ONLY during training, never when choosing
        # an action.
        self.aux_card_head = nn.Linear(self.LSTM_HIDDEN, NUM_CARD_IDS_LIVE)

        # ==========================================
        # 8. IDENTITY head on the cycle branch (2026-08-28)
        # ==========================================
        # The other half of the detach in ScalarEncoder.forward. Reading the
        # 24-dim branch DIRECTLY rather than hx, it is the only gradient those
        # 3,752 parameters now receive, and it asks for exactly one thing: is
        # the identity of each card the opponent has shown still recoverable
        # from this block?
        #
        # Its coefficient can be O(1) precisely BECAUSE of the detach. The old
        # arrangement had `aux_card_head` fighting the actor and the critic for
        # the same weights and losing 248:1 -- matching that needed a
        # coefficient near 2.5, which puts a 2.0-nat CE against an actor loss
        # of 0.02, i.e. a different objective rather than a tuning knob. With
        # the branch isolated there is no fight to lose and no coefficient to
        # balance.
        #
        # `aux_card_head` STAYS, on plain hx, and the two are not redundant.
        # `recency[]` decays with a 200-tick constant, so this head asks "what
        # has been played lately"; knowing which four cards they HOLD means
        # integrating the play sequence, which only the recurrence can do. This
        # head protects the encoder, that one asks the LSTM to do the
        # integration it currently is not doing. It is also why `aux_card_head`
        # must keep reading hx alone and never `_head_input` -- handed the skip
        # it would answer from the branch and stop asking anything of memory.
        self.cycle_id_head = (nn.Linear(CYCLE_BRANCH_DIM, NUM_CARD_IDS_LIVE)
                              if self.cycle_feature_dim else None)

    def _split_cycle(self, features):
        """The DETACHED cycle block ScalarEncoder placed at the end of `features`.

        None when there is no cycle branch. Never `features[..., -0:]`, which
        would silently be the entire feature vector.
        """
        if not self.cycle_feature_dim:
            return None
        # `.detach()` again, and it is not redundant even though `forward`
        # already detached what it concatenated. The slice comes out of a `cat`
        # whose OTHER inputs carry a graph, so it inherits requires_grad=True
        # and merely happens to route zero gradient to the branch. That makes
        # the isolation an argument about cat's backward rather than a property
        # you can read here. This makes it local and costs nothing.
        return features[..., -self.cycle_feature_dim:].detach()

    def _head_input(self, hx, cycle_feat):
        """hx, plus the cycle skip when there is one. (Batch, head_dim)."""
        if cycle_feat is None:
            return hx
        return torch.cat((hx, cycle_feat), dim=-1)

    def cycle_features(self, obs, detached=False):
        """The cycle branch run on a raw observation, WITH gradient by default.

        The PPO update's entry point to `cycle_id_head`, and the fallback
        `placement_given_card` uses when a caller hands it `obs` but no
        precomputed context. Returns None where there is no branch.
        """
        if not self.cycle_feature_dim:
            return None
        scalar_obs = obs[:, self.spatial_size:]
        cycle = scalar_obs[:, self.cycle_start:
                           self.cycle_start + self.cycle_block_size]
        out = self.scalar_mlp.cycle_branch(cycle)
        return out.detach() if detached else out

    def predict_cycle_card(self, cycle_feat):
        """Logits over card ids, read from the cycle branch. (Batch, C).

        Trained against the same next-card label as `predict_opp_next_card`;
        see `cycle_id_head` for why both exist.
        """
        return self.cycle_id_head(cycle_feat)

    def extract_features_hires(self, obs):
        """
        Feature extraction (CNN + scalar MLP + card-identity embeddings) -- the
        non-recurrent part of the network. It can be called on one huge flattened
        batch (T*N) to run the CNN (and the cheap card-embedding computation)
        once instead of once per tick -- the main speed-up of the PPO update.

        obs: (Batch, obs_dim)
        Returns (combined, card_embeds, spatial_map, hires_map):
          combined: (Batch, lstm_input_dim) -- fed to the LSTM.
          card_embeds: (Batch, hand_size+1, CARD_EMBED_DIM) -- one embedding per
            real hand slot plus one more (the no-op), indexed by card_idx only
            after the card is chosen -- see placement_given_card. Keeping it
            apart from `combined` is exactly what makes autoregressive
            placement possible: the card's identity cannot be mixed into the
            LSTM before the choice without losing the ability to condition on
            it afterwards.
          spatial_map: (Batch, 32, pooled_h, pooled_w) -- the CNN feature map
            BEFORE flattening, the input of the convolutional placement head.
            It is the tensor `combined` is derived from, not an extra
            computation.
          hires_map: (Batch, 16, board_height, board_width) -- the activation
            before the first pooling, the input of the full-resolution branch
            (see place_hires). Also not an extra computation: spatial_map is
            derived from it.
        """
        # Split the flat vector into its spatial and scalar parts, following observationSize() in C++
        spatial_obs = obs[:, :self.spatial_size].view(-1, self.channels, self.board_height, self.board_width)
        scalar_obs = obs[:, self.spatial_size:]

        # The trunk runs in two halves instead of as one Sequential, to expose the
        # pre-pooling activation (16x34x18) to the full-resolution branch. Same
        # modules, same order, same computation -- nothing is computed twice, and
        # test_trunk_split_is_bit_identical_to_the_sequential checks there is not
        # even a one-ulp difference (one would shift the features under every
        # checkpoint).
        hires_map = self.cnn_trunk[:2](spatial_obs)
        spatial_map = self.cnn_trunk[2:](hires_map)
        cnn_features = self.cnn_flatten(spatial_map)
        scalar_features = self.scalar_mlp(scalar_obs)
        combined = torch.cat((cnn_features, scalar_features), dim=1)

        # The layout in which ClashEnv::extractObservationForTeam builds
        # scalar_obs: [elixir(1), costs(hand_size), onehots(hand_size*num_card_ids)].
        onehot_start = 1 + self.hand_size
        card_onehots = scalar_obs[:, onehot_start:onehot_start + self.hand_size * self.num_card_ids]
        card_onehots = card_onehots.view(-1, self.hand_size, self.num_card_ids)
        card_embeds = self.card_id_embed(card_onehots)  # (Batch, hand_size, CARD_EMBED_DIM)

        noop = self.noop_embed.view(1, 1, -1).expand(card_embeds.shape[0], 1, -1)
        card_embeds = torch.cat([card_embeds, noop], dim=1)  # (Batch, hand_size+1, CARD_EMBED_DIM)

        return combined, card_embeds, spatial_map, hires_map

    def extract_features(self, obs):
        """Only the original three return values -- see extract_features_hires.

        Kept as a wrapper instead of widening the signature, because every
        existing caller (both trainers, the exploiter, every measurement script,
        the live loop) unpacks exactly three values. Only the hot callers moved
        to extract_features_hires and pass the map on; everyone else lets
        placement_given_card rebuild it from obs, which is EXACTLY THE SAME
        computation (checked by test_recomputed_hires_equals_the_passed_one), at
        the cost of one extra conv on a cold path.
        """
        combined, card_embeds, spatial_map, _ = self.extract_features_hires(obs)
        return combined, card_embeds, spatial_map

    def hires_features(self, obs):
        """The trunk's activation before pooling: (Batch, 16, 34, 18).

        This is the input of the full-resolution branch. Computed from obs so a
        caller that did not keep the map can rebuild it itself, instead of
        silently getting a different head.
        """
        spatial_obs = obs[:, :self.spatial_size].view(
            -1, self.channels, self.board_height, self.board_width)
        return self.cnn_trunk[:2](spatial_obs)

    def affordability_mask(self, obs):
        """
        Action mask: which hand slots are actually playable NOW, given the
        current elixir and the costs -- both already inside the observation
        vector, so this is computed in Python alone, with no extra engine call.

        This is the single most important fix in the whole pipeline.
        GameManager::playCard returns false silently when there is not enough
        elixir -- no exception, no negative reward, no signal to the agent at
        all. Measured on the trained policy of the time (243 real decision
        steps): 74.9% of all steps tried to play a card it could not afford,
        and only 11.5% actually placed one. So in ~87% of the samples in every
        rollout the action stored in the buffer had no effect on the world --
        the same next_state would have followed any other action -- and the
        advantage attributed to it was pure noise fed straight into the
        gradient. The cause is structural, not transient: elixir regenerates at
        0.035 per tick (at 1x) and skip_frames=10, i.e. 0.35 elixir per
        decision against cards costing 3-5, so on average only 0.55 of the 4
        slots were playable and only 27.2% of steps had even one legal option.

        The mask must be applied IDENTICALLY in rollout collection and in the
        PPO update, or the old/new logprob ratio breaks -- which is why it is
        computed from the observation itself (stored in the buffer anyway)
        rather than stored separately: the same obs necessarily produces the
        same mask in both calls.

        obs: (Batch, obs_dim). Returns a bool tensor (Batch, hand_size+1); the
        last column (no-op) is always True -- waiting is always a legal action.
        """
        scalar_obs = obs[:, self.spatial_size:]
        # The layout in which ClashEnv::extractObservationForTeam builds
        # scalar_obs: [elixir(1), costs(hand_size), onehots(...)]. Both are
        # divided by 10 there, so their ratio is right without scaling back.
        elixir = scalar_obs[:, 0:1]
        costs = scalar_obs[:, 1:1 + self.hand_size]
        # cost <= 0 marks an empty or invalid slot (see ClashEnv: card ? cost/10 : 0)
        # -- never a real free card.
        playable = (costs > 0.0) & (costs <= elixir + 1e-6)
        noop = torch.ones(obs.shape[0], 1, dtype=torch.bool, device=obs.device)
        return torch.cat([playable, noop], dim=1)

    def hand_card_ids(self, obs):
        """
        Which card_id sits in each hand slot, from the one-hot already in obs.
        (Batch, hand_size) long. An empty slot -> -1.

        Needed by the live diagnostics (how many distinct cards the bot really
        plays): a card's identity must be read from the obs the decision was
        made on, not from game.get_hand() after the step -- the hand rotates as
        soon as a card is played, so a late read returns the NEXT card, not the
        one chosen.
        """
        scalar_obs = obs[:, self.spatial_size:]
        onehot_start = 1 + self.hand_size
        onehots = scalar_obs[:, onehot_start:onehot_start + self.hand_size * self.num_card_ids]
        onehots = onehots.view(-1, self.hand_size, self.num_card_ids)
        ids = onehots.argmax(dim=-1)
        # An empty one-hot row (sum 0) is a slot with no card -- argmax would
        # return 0, which is a valid card_id (Knight), so mark it explicitly.
        return torch.where(onehots.sum(dim=-1) > 0.5, ids, torch.full_like(ids, -1))

    def elixir_from_obs(self, obs):
        """Current elixir (0..10) from the obs. ClashEnv divides it by 10 when building it."""
        return obs[:, self.spatial_size] * 10.0

    def hand_costs_from_obs(self, obs):
        """Hand card costs (0..10) from the obs: a (Batch, hand_size) tensor.

        The same undoing of the divide-by-10 that elixir_from_obs does, on the
        scalars right after the elixir -- exactly the offsets affordability_mask
        reads.

        Returns a tensor, not a list, following elixir_from_obs above: a caller
        that wants a flat list for a single row writes `[0].tolist()`
        explicitly, instead of the function silently swallowing the batch
        dimension.

        It lives here because it had three literal copies
        (advisors/hybrid_policy.py, eval/gate_ab.py, perception/live/mvp_loop.py),
        each with the offset and the 10.0 written by hand -- exactly the "second
        copy of an engine constant" pattern CLAUDE.md forbids, which has already
        gone stale twice in this project.

        Not folded into SolvencyGate.mask: the test there is built on
        np.zeros(SPATIAL+1), so reading costs would return an empty slice rather
        than raise -- and the test would pass without checking anything.
        """
        s = self.spatial_size
        return obs[:, s + 1:s + 1 + self.hand_size] * 10.0

    def step_lstm_and_card(self, features, hidden_state, card_mask=None):
        """
        First half of the recurrent step: advances the LSTM and computes the
        card choice and the state value -- neither depends on the card about to
        be chosen. Kept apart from placement so the value-only bootstrap call
        (for GAE) never has to compute a placement it would throw away.
        features: (Batch, lstm_input_dim)
        card_mask: (Batch, hand_size+1) bool from affordability_mask, or None
          (no masking -- the old behaviour, only for where no observation is
          available).
        """
        hx, cx = self.lstm(features, hidden_state)
        # The cycle skip. Taken from `features` rather than recomputed, so it
        # is bit-identical to what the LSTM was just fed and costs nothing --
        # ScalarEncoder already put it at the end of that vector, detached.
        head_in = self._head_input(hx, self._split_cycle(features))
        card_logits = self.card_head(head_in)
        if card_mask is not None:
            # -inf, not a large-but-finite negative: Categorical normalises
            # through log_softmax, and a finite value would still leave a tiny
            # but non-zero probability for an illegal action, i.e. both a rare
            # sample of it and an entropy contribution. -inf gives exactly zero
            # in both. The no-op is always legal (see affordability_mask), so no
            # row can come out all -inf.
            card_logits = card_logits.masked_fill(~card_mask, float("-inf"))
        state_value = self.value_head(head_in)
        # Like card_logits and state_value, the Champion heads never depend on
        # the card about to be chosen, so they are computed here, not in
        # placement_given_card. None when the deck has no Champion (see
        # num_ability_slots), and the trainer skips them entirely.
        ability_slot1_logits = self.ability_slot1_head(hx) if self.ability_slot1_head is not None else None
        ability_slot2_logits = self.ability_slot2_head(hx) if self.ability_slot2_head is not None else None
        return card_logits, ability_slot1_logits, ability_slot2_logits, state_value, (hx, cx)

    def placement_given_card(self, hx, card_embeds, card_idx, obs=None,
                             spatial_map=None, hires_map=None,
                             ctx=None, ctx_hi=None, cycle_feat=None):
        """
        Second half: placement conditioned on card_idx (sampled just now at
        rollout time, or read from the buffer during the PPO update) -- this is
        the autoregressive step itself.
        hx: (Batch, 256). card_embeds: (Batch, hand_size+1, CARD_EMBED_DIM).
        card_idx: (Batch,) long tensor, values in [0, hand_size] inclusive.

        Returns logits (Batch, placement_cells) over whole board cells. The
        conversion to real coordinates is cell_to_xy() below.

        Factorised JOINTLY over all cells (not as a separate product of x and
        y): a factorised p(x)*p(y) cannot represent "either near the left
        bridge or in the back-right corner" without also spreading mass over
        the two mixed combinations, and that is exactly the kind of bimodal
        decision the game demands.

        placement_cells = placement_rows * board_width = 34*18 = **612**, not
        288. Corrected on 2026-08-27: 18*16 = 288 was right only while the
        placement head covered our half alone (16 rows), and it stayed here
        after the head was widened to all 34 rows so a spell could cross the
        river. CLAUDE.md names this line as an example of the "do not keep a
        second copy of an engine constant" rule -- and it is exactly that
        failure: a number that was once measurable, kept after the thing it
        described changed.

        Note that 288 = 16*18 still appears elsewhere in this file (208/288
        legal cells, 91/288 used), and there it is CORRECT: that is the legal
        deploy area for an ordinary troop, our half only. The two numbers live
        side by side and describe different things, which is why mixing them
        up was easy.
        """
        if spatial_map is None:
            raise ValueError(
                "placement_given_card needs the spatial_map from extract_features. "
                "Passing None here would produce different logits from the ones the "
                "rollout produced, and the PPO ratio would break silently -- so this "
                "is an error, not a default.")

        if hires_map is None and obs is None:
            # No silent fallback to the coarse-only head. A different function
            # from the one that ran the rollout would break the PPO ratio
            # silently -- exactly what the spatial_map=None guard already exists
            # for.
            #
            # Checked here, at the top of the function, not where hires_map is
            # built: since the cycle skip was added there are TWO dependencies
            # on obs, and this message is the more precise of the two -- a caller
            # who got the cycle message while hires_map was also missing would
            # fix the wrong thing.
            raise ValueError(
                "placement_given_card needs hires_map, or obs to build it "
                "(see hires_features). None in both would compute a different "
                "head from the one that ran the rollout.")

        batch_idx = torch.arange(card_embeds.shape[0], device=card_embeds.device)
        chosen_embed = card_embeds[batch_idx, card_idx]  # (Batch, CARD_EMBED_DIM)
        # Context -> 32 channels, broadcast over every cell of the spatial map.
        # Addition, not concatenation: "what is the overall situation, and which
        # card" shifts the whole logit map, while the local structure of the
        # board stays in the map itself.
        # ctx/ctx_hi are supplied from outside only by the row-compaction path in
        # forward_sequence, which computes them over the WHOLE batch and then
        # slices. The reason was measured: nn.Linear (GEMM) is NOT independent of
        # batch size on this backend -- Linear(280->32) differs by 4.768e-07 in
        # the forward and by 2.289e-05 in grad_W between batch 500 and 167. These
        # two layers are ~0.5% of the head's cost, so computing in full and
        # slicing is nearly free, and it is what makes the LOGITS of the kept
        # rows bit-identical.
        #
        # **But it does not make the WEIGHTS bit-identical, and it must not be read that way.**
        # Conv2d's grad_W also depends on batch size -- but only for some
        # shapes, and that is exactly the trap: a first check at 18x10 came back
        # "independent" and was recorded here as such, and a sweep over the
        # other shapes refuted it. Measured 500 -> 184, with an exactly-zero
        # incoming gradient on the dropped rows:
        #
        #     place_up.1    Conv2d(32,16) 18x10   bit-identical
        #     place_up.4    Conv2d(16,8)  36x20   differs by 5.814e-03
        #     place_up.6    Conv2d(8,1)   36x20   differs by 2.808e-03
        #     place_hires.0 Conv2d(24,8)  34x18   differs by 4.883e-03
        #
        # This blocks ANY row-compaction scheme from being bit-exact at the level
        # of the weights. What IS guaranteed, and measured: the logits of the
        # kept rows, and the loss itself.
        # The cycle skip, on the same terms as `hires_map` two blocks down:
        # supplied by the hot caller, else rebuilt from `obs`, else an error --
        # never a silent zero. A zero block here would compute a DIFFERENT head
        # from the one that ran the rollout and break the PPO ratio quietly,
        # which is the exact failure the spatial_map guard above already exists
        # to prevent. Skipped entirely when both contexts are precomputed.
        if cycle_feat is None and self.cycle_feature_dim and (
                ctx is None or ctx_hi is None):
            if obs is None:
                raise ValueError(
                    "placement_given_card needs cycle_feat or obs to rebuild "
                    "it (see cycle_features). None in both would compute a "
                    "different head from the one that ran the rollout.")
            cycle_feat = self.cycle_features(obs, detached=True)
        head_in = self._head_input(hx, cycle_feat)

        if ctx is None:
            ctx = self.place_ctx(torch.cat((head_in, chosen_embed), dim=-1))  # (B,32)
        h = spatial_map + ctx.view(-1, 32, 1, 1)
        logit_map = self.place_up(h)                                    # (B, 1, 4*ph, 4*pw)
        logits = logit_map[:, 0, :self.placement_rows, :self.board_width].reshape(
            -1, self.placement_cells)

        # --- the full-resolution branch, as a residual -----------------------
        # The coarse path above knows roughly where; this picks the tile inside
        # the block. Zero-initialised, so at initialisation this line adds an
        # EXACT zero (not "approximately"): the last convolution is zeroed in
        # weight and bias, so its output is an exact zero tensor and the
        # residual sum is exact. That is what lets existing checkpoints load and
        # behave bit-identically.
        if hires_map is None:
            hires_map = self.hires_features(obs)
        if ctx_hi is None:
            ctx_hi = self.place_ctx_hi(torch.cat((head_in, chosen_embed), dim=-1))
        h_hi = torch.cat(
            (hires_map,
             ctx_hi.view(-1, HIRES_CTX_DIM, 1, 1).expand(
                 -1, -1, hires_map.shape[2], hires_map.shape[3])), dim=1)
        fine = self.place_hires(h_hi)                                   # (B,1,H,W)
        logits = logits + fine[:, 0, :self.placement_rows, :self.board_width].reshape(
            -1, self.placement_cells)
        if obs is not None:
            # Per-card legality mask. -inf, not a finite value, for exactly the
            # same reason as in step_lstm_and_card: Categorical normalises
            # through log_softmax, so a finite value would leave a tiny
            # probability on an illegal cell and an entropy contribution. There
            # is always at least one legal cell, so no row comes out all -inf.
            logits = logits.masked_fill(~self.placement_mask(obs, card_idx), float("-inf"))
        return logits

    def forward_sequence(self, feats_seq, card_embeds_seq, spatial_seq, obs_seq,
                         card_mask_seq, card_idx_seq, reset_seq, hidden_state,
                         extra_card_idx_seq=None, hires_seq=None,
                         active_rows=None, with_ability=False):
        """
        A batched replacement for looping forward_from_features over timesteps.
        Mathematically IDENTICAL -- checked by a dedicated bit-identity test.

        Why it exists: only the LSTM is truly recurrent. Everything after it
        (card head, value, aux, placement) is pointwise in time, but the loop
        ran it L times on a batch of B=8. On CPU that is dominated by overhead,
        not compute: measured 35.3 ms for 25 calls of the card/value/aux heads,
        against 0.7 ms for one call on L*B=200 -- 48x. The placement head gains
        1.3x (it is compute-bound, not overhead-bound).

        Overall this saves ~5% of update time, no more -- the CNN and the
        placement convolutions are still 84% of the cost. It is worth it only
        because the equivalence is provable.

        feats_seq/obs_seq/card_mask_seq/card_idx_seq/reset_seq: (L, B, ...).
        Returns card_logits (L,B,hand+1), place_logits (L,B,cells),
        values (L,B), aux_card_logits (L,B,C), the final hidden state, and the
        coverage-pass logits (or None); with_ability=True appends the ability
        logits.
        """
        L, B = feats_seq.shape[0], feats_seq.shape[1]
        hx, cx = hidden_state
        hx_steps = []
        for l in range(L):
            hx, cx = self.lstm(feats_seq[l], (hx, cx))
            # Collected BEFORE the reset, exactly as in the original loop: the
            # heads at step l use the state after the LSTM and before the
            # end-of-episode reset.
            hx_steps.append(hx)
            reset = reset_seq[l].unsqueeze(1)
            hx = hx * reset
            cx = cx * reset

        flat_hx = torch.stack(hx_steps).reshape(L * B, -1)
        # The cycle skip, read off the same feats_seq the LSTM consumed, so it
        # matches step_lstm_and_card's `head_in` exactly.
        flat_cycle = self._split_cycle(feats_seq.reshape(L * B, -1))
        flat_head = self._head_input(flat_hx, flat_cycle)
        card_logits = self.card_head(flat_head)
        mask_flat = card_mask_seq.reshape(L * B, -1)
        card_logits = card_logits.masked_fill(~mask_flat, float("-inf"))
        values = self.value_head(flat_head).squeeze(-1)
        # (L*B, NUM_CARD_IDS) LOGITS, not a scalar -- the caller reshapes to
        # (L, B, C) and feeds cross_entropy. No output scaling: the old head
        # multiplied by 10.0 to put a sigmoid-free regression in elixir units,
        # which has no analogue for a classifier.
        # flat_hx, NOT flat_head: this head exists to pressure the RECURRENCE,
        # and handed the skip it would answer straight off the branch.
        aux = self.aux_card_head(flat_hx)

        # Built once and shared by both calls below. Without it the coverage
        # pass would recompute the trunk's conv1 a second time on exactly the
        # same input.
        flat_obs = obs_seq.reshape(L * B, -1)
        flat_hires = (self.hires_features(flat_obs) if hires_seq is None
                      else hires_seq.reshape(L * B, *hires_seq.shape[2:]))
        flat_embeds = card_embeds_seq.reshape(L * B, *card_embeds_seq.shape[2:])
        flat_spatial = spatial_seq.reshape(L * B, *spatial_seq.shape[2:])

        # --- row compaction (2026-08-24) --------------------------------------
        # The placement head is ~41% of update time and runs here TWICE (the
        # chosen card + the coverage slot). Every consumer of both outputs is
        # masked by decision: actor_loss by mb_decision, placement entropy by
        # mb_placed (a subset of it), clip_frac by mb_decision, and both halves
        # of coverage_terms by decision. Measured on model_weights_selfplay.pth
        # over 1500 steps: decision fires on 0.368 of rows, i.e. **63.2% of these
        # convolutions were multiplied by an exact zero**.
        #
        # active_rows=None reproduces the previous behaviour exactly, so no
        # existing caller is affected.
        #
        # The fill is **zero, not -inf**, and that choice is load-bearing: an all
        # -inf row gives Categorical.entropy() = nan, and nan * 0.0 = nan would
        # poison every masked sum in the update. Any FINITE fill gives
        # finite * 0.0 == 0.0 exactly, which is exactly what the old path
        # produced there -- so every masked reduction keeps its shape, order and
        # values, and the loss comes out bit-identical.
        #
        # The two linear context layers are computed over the WHOLE batch and
        # only then sliced, because GEMM is not batch-size independent. See
        # placement_given_card -- which also documents why the WEIGHTS are
        # nevertheless not bit-identical (Conv2d's grad_W is batch-dependent at
        # the 36x20 and 34x18 shapes), a backend limit no row-compaction
        # implementation can get around.
        def _placement(idx_seq):
            flat_idx = idx_seq.reshape(L * B)
            if active_rows is None:
                return self.placement_given_card(
                    flat_hx, flat_embeds, flat_idx, flat_obs, flat_spatial,
                    hires_map=flat_hires, cycle_feat=flat_cycle)
            out = flat_hx.new_zeros(L * B, self.placement_cells)
            if active_rows.numel() == 0:
                # An empty batch would reach Conv2d; skip it entirely. This is
                # not a theoretical corner case -- a chunk made entirely of
                # forced steps is exactly what a bankrupt agent produces, and
                # P(nothing affordable) measured 73.9%.
                return out
            rows = torch.arange(L * B, device=flat_hx.device)
            joint = torch.cat((flat_head, flat_embeds[rows, flat_idx]), dim=-1)
            sub = self.placement_given_card(
                flat_hx[active_rows], flat_embeds[active_rows],
                flat_idx[active_rows], flat_obs[active_rows],
                flat_spatial[active_rows], hires_map=flat_hires[active_rows],
                ctx=self.place_ctx(joint)[active_rows],
                ctx_hi=self.place_ctx_hi(joint)[active_rows],
                cycle_feat=None if flat_cycle is None else flat_cycle[active_rows])
            return out.index_copy(0, active_rows, sub)

        place_logits = _placement(card_idx_seq)

        # --- placement COVERAGE pass (optional) -----------------------------
        # The placement map of a card OTHER than the chosen one, on exactly the
        # same flat_hx/spatial. It exists to close a coverage hole in the
        # gradient: both actor_loss and the entropy bonus flow only through the
        # placement_given_card of the card that was CHOSEN, so a card the policy
        # stopped playing never gets placement gradient again and its head
        # freezes. Measured: Cannon/Fireball/Giant returned the fixed cell (11,0)
        # in 54%-91% of states, and a Cannon at the policy's cell saved 121 tower
        # HP against 396 for a random legal cell -- worse than random, i.e. a
        # broken function rather than a value judgement.
        #
        # Returns logits only; the caller decides what to do with them (train.py
        # adds entropy). There is no new parameter here -- the head is the same
        # head -- so no checkpoint is invalidated.
        extra_logits = None
        if extra_card_idx_seq is not None:
            extra_logits = _placement(extra_card_idx_seq).view(L, B, -1)

        out = (card_logits.view(L, B, -1), place_logits.view(L, B, -1),
                values.view(L, B), aux.view(L, B, -1), (hx, cx), extra_logits)
        if not with_ability:
            return out
        # One (L, B, 2) logit tensor per Champion head, off flat_hx -- the same
        # input step_lstm_and_card gives the heads at rollout time.
        ability = [h(flat_hx).view(L, B, 2)
                   for h in (self.ability_slot1_head, self.ability_slot2_head)
                   if h is not None]
        return out + (ability,)

    def predict_opp_next_card(self, hx):
        """Logits over card ids for the opponent's NEXT play. (Batch, C).

        Raw logits, not probabilities: the loss is cross_entropy, which wants
        logits, and the only other caller (expert_metrics) reports them the
        same way.

        Called only from the PPO update loop (see `aux_card_coef`). Its
        gradient flows back into the LSTM and the CNN -- that is the entire
        point of the head, and the reason it replaced the elixir regression,
        whose target was solvable from two present scalars and therefore
        shaped nothing. See the head's definition for the measurement.
        """
        return self.aux_card_head(hx)

    def placement_mask(self, obs, card_idx):
        """
        Which board cells are legal for the chosen card. (Batch, placement_cells) bool.

        The engine (GameManager::isValidPlacement) distinguishes two cases:
          * an ordinary troop -- our half only, y <= get_own_half_max_y().
          * a spell           -- the whole board; the half rule is skipped.
        Before this mask, Python forced the stricter case on both (target_y was
        always capped at 15.5), so Fireball could never cross the river. This
        mask puts the distinction back where it belongs.

        Which cards are exempt from the half rule is read from the one-hot
        already in obs times row_free_flags (spells and deploy-anywhere
        troops), with no engine call and no argmax -- so it works on a whole
        batch and reproduces exactly the same mask in the PPO update as in the
        rollout.

        no-op (card_idx == hand_size): the engine ignores the placement
        entirely, so the our-half mask is returned only to keep the
        distribution well-defined and non-empty.
        """
        batch = obs.shape[0]
        scalar_obs = obs[:, self.spatial_size:]
        onehot_start = 1 + self.hand_size
        onehots = scalar_obs[:, onehot_start:onehot_start + self.hand_size * self.num_card_ids]
        onehots = onehots.view(batch, self.hand_size, self.num_card_ids)
        # (Batch, hand_size) -- 1.0 where the slot holds a spell
        # "spell" here means "exempt from the own-half ROW rule", which covers
        # deploy-anywhere troops too (row_free_flags). Reading spell_flags
        # instead confined a Miner to its own half: audit 06, E-1.
        slot_is_spell = (onehots * self.row_free_flags.view(1, 1, -1)).sum(dim=-1)
        # Extend to the no-op slot (never a spell)
        slot_is_spell = torch.cat(
            [slot_is_spell, torch.zeros(batch, 1, device=obs.device, dtype=slot_is_spell.dtype)], dim=1)
        chosen_is_spell = slot_is_spell.gather(1, card_idx.view(-1, 1)).squeeze(1) > 0.5  # (Batch,)

        rows = torch.arange(self.placement_rows, device=obs.device).view(1, -1, 1)
        own_half = rows < self.own_half_rows                       # (1, rows, 1)
        full_board = torch.ones_like(own_half)
        allowed_rows = torch.where(chosen_is_spell.view(-1, 1, 1), full_board, own_half)
        mask = allowed_rows.expand(batch, self.placement_rows, self.board_width)
        mask = mask.reshape(batch, self.placement_cells)

        # ...and now also what the engine actually accepts, not just the half-board.
        #
        # The mask above allows every row on our half. GameManager::isValidPlacement
        # does not: it also rejects Board::isBackRowDeadZone and the towers'
        # footprints. playCard returns false silently, with no exception and no
        # signal, so such an action has the same effect as a no-op and its
        # advantage is pure noise fed into the gradient. This is exactly the
        # disease the affordability mask was built to cure, left open on the
        # placement axis.
        #
        # Measured on the ep~45,800 checkpoint over 1,340 decision steps: 58.7% of
        # the card choices were rejected here, 94.8% of the rejections on row
        # y=0. Zero leaks through the affordability mask.
        #
        # The table is derived from the engine's is_valid_placement, not
        # recomputed in Python: the predicate combines the board bounds, the dead
        # zone, placementRadius/isSpell/deployAnywhere and the towers'
        # footprints. A second copy of that geometry is exactly the drift this
        # project has already paid for twice.
        if self._placement_legal is not None:
            table = self._placement_legal.to(obs.device)
            slot_ids = onehots.argmax(dim=-1)                       # (B, hand)
            # An empty slot: the one-hot is all zero and argmax returns 0, which
            # is a valid card id. Mark it explicitly so it is never read as card 0.
            slot_empty = onehots.sum(dim=-1) <= 0.0                 # (B, hand)
            slot_ids = torch.where(slot_empty, torch.full_like(slot_ids, -1), slot_ids)
            noop = torch.full((batch, 1), -1, device=obs.device, dtype=slot_ids.dtype)
            slot_ids = torch.cat([slot_ids, noop], dim=1)
            chosen_id = slot_ids.gather(1, card_idx.view(-1, 1)).squeeze(1)
            # The table's last row is a permissive fallback for the no-op / an
            # empty slot, so the distribution stays well-defined (no row all -inf).
            chosen_id = torch.where(chosen_id < 0,
                                    torch.full_like(chosen_id, table.shape[0] - 1),
                                    chosen_id)
            # Legality, selected by which of OUR OWN Princess Towers are still
            # standing. A tower's 3x3 footprint clears when it dies (+9 cells
            # each, measured), so a table cached on a full board permanently
            # masks the policy out of the ground it defends after losing a
            # tower. ENEMY towers change nothing, which is why only team 0's two
            # Princesses index this.
            #
            # Liveness comes from the ally-building channel at each tower's own
            # centre cell: the encoder marks a tower at its centre only, at
            # hp/MAX_BUILDING_HP (2534/4008 = 0.632 at full) and 0 once dead,
            # and `hp <= 0` is refused by the engine's setters, so `> 0` is
            # exactly "alive". Read as two scalar columns straight out of the
            # flat vector via the precomputed offsets -- reshaping the whole
            # (channels, H, W) block to slice channel 3 copies 612 floats per
            # row for two numbers.
            #
            # ONE gather, into the state-indexed table built in __init__.
            # Doing it as base-gather + per-tower gather + OR measured 53% more
            # expensive than the base mask; this costs the same as the single
            # lookup it replaces.
            if self._placement_legal_by_state is not None:
                by_state = self._placement_legal_by_state.to(obs.device)
                dead_l = (obs[:, self._own_princess_flat[0]] <= 0.0).long()
                dead_r = (obs[:, self._own_princess_flat[1]] <= 0.0).long()
                legal = by_state[dead_l + 2 * dead_r, chosen_id]
            else:
                legal = table[chosen_id]

            mask = mask & legal
        return mask

    def freed_cells_for(self, tower_slot, card_id):
        """{(x, y)} that our own Princess `tower_slot` (1=LEFT, 2=RIGHT) hands
        back for `card_id` when it dies. Diagnostic/testing accessor."""
        if self._placement_freed is None:
            return set()
        row = self._placement_freed[tower_slot - 1, card_id]
        return {(c % self.board_width, c // self.board_width)
                for c in torch.nonzero(row, as_tuple=True)[0].tolist()}

    def cell_to_xy(self, cell_idx):
        """
        Cell index -> the real board coordinates the engine accepts.
        cell_idx: any long tensor. Returns (x, y) float tensors of the same shape.

        Row-major, matching the placement head's layout. x = column and y = row,
        as exact integers: the largest column is board_width - 1 =
        get_max_placement_x() and the largest row is placement_rows - 1 =
        BOARD_HEIGHT - 1, so every cell falls inside the board the engine
        enforces -- no clamp is needed and no cell silently rolls onto the edge.
        Which cells a given card may use is placement_mask's job.
        """
        row = torch.div(cell_idx, self.board_width, rounding_mode="floor")
        col = cell_idx % self.board_width
        return col.float(), row.float()

    def forward_from_features(self, features, card_embeds, hidden_state, card_idx,
                              card_mask=None, obs=None, spatial_map=None):
        """
        Runs both halves together, for use when card_idx is already known (the
        PPO update, with the action stored in the buffer -- critical: always
        pass the STORED card_idx here, never a fresh sample, or the PPO ratio
        between the old and new logprob breaks). Not useful during rollout
        collection (card_idx is not known there until it is sampled from
        card_logits) -- there, call step_lstm_and_card and then
        placement_given_card separately; see train.py / train_selfplay.py.

        card_mask must be the same mask that was applied at rollout time (in
        practice: recomputed from the same obs stored in the buffer -- see
        affordability_mask).
        """
        (card_logits, ability_slot1_logits, ability_slot2_logits, state_value,
         (hx, cx)) = self.step_lstm_and_card(features, hidden_state, card_mask)
        placement_logits = self.placement_given_card(hx, card_embeds, card_idx, obs, spatial_map)
        return (card_logits, placement_logits, state_value,
                ability_slot1_logits, ability_slot2_logits, (hx, cx))
