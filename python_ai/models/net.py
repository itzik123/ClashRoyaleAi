import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import clash_royale_env

# Small on purpose: it only has to tell apart the few cards a hand slot can
# hold; the CNN and scalar encoder give the LSTM everything else.
CARD_EMBED_DIM = 16

# Width of the placement head's full-resolution branch (place_hires). Narrow
# because it runs at the full 34x18, ~12x the cells of the pooled map; it only
# picks the tile within a coarse block.
HIRES_CTX_DIM = 8
HIRES_HIDDEN = 8

# Placement constants, read from the engine. get_own_half_max_y() needs an
# instance, so one is built once here.
_probe = clash_royale_env.ClashRoyaleEnv(list(range(8)), list(range(8)), 100)
OWN_HALF_MAX_Y = _probe.get_own_half_max_y()
MAX_PLACEMENT_X = _probe.get_max_placement_x()
del _probe
OWN_HALF_ROWS = int(OWN_HALF_MAX_Y) + 1
# The placement head spans the whole board: spells (and deploy-anywhere troops)
# may cross the river, and per-card legality is a mask (placement_mask), not a
# smaller space.
PLACEMENT_ROWS = clash_royale_env.ClashRoyaleEnv.BOARD_HEIGHT

_ALL_IDS = clash_royale_env.get_all_card_ids()
NUM_CARD_IDS_LIVE = clash_royale_env.ClashRoyaleEnv.NUM_CARD_IDS


def _registered_card_ids(num_card_ids):
    """Every id the engine can put in a hand, Evolutions included.

    `get_all_card_ids()` filters out the Evolutions (ids 123-163), which is
    right for sampling a deck but wrong for what a hand can hold. Probed
    through `get_card_info`, which raises on the gaps.
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
#: 1.0 where the own-half row rule does not apply: spells and deploy-anywhere
#: troops (Miner, Goblin Drill). Separate from `_spell_flags` because the two
#: facts differ; the legality table still decides the exact cells.
_row_free_flags = torch.zeros(NUM_CARD_IDS_LIVE)
for _cid in _REGISTERED_IDS:
    _info = clash_royale_env.get_card_info(_cid)
    if _info["is_spell"]:
        _spell_flags[_cid] = 1.0
    if _info["is_spell"] or _info.get("deploy_anywhere", False):
        _row_free_flags[_cid] = 1.0



def _build_placement_legality(num_card_ids, placement_rows, board_width):
    """(num_card_ids + 1, rows*width) bool: what the engine will accept, plus the
    cells each own Princess Tower frees when it dies.

    Derived from ClashRoyaleEnv.is_valid_placement rather than restating its
    geometry (bounds, back-row dead zone, placement radius, spells,
    deploy-anywhere, tower footprints). Legality does not depend on troops on
    the board, so the table is built once (~113k predicate calls).

    Returns (None, None) when the binding is missing, falling back to a
    row-only mask with a loud warning: a stale .pyd would otherwise silently
    let the policy choose placements the engine rejects.
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
    # Every id a hand can hold; see _registered_card_ids.
    known = set(_registered_card_ids(num_card_ids))
    for cid in range(num_card_ids):
        if cid not in known:
            # Unknown id: all-False. It is never in a hand.
            continue
        for cell in range(cells):
            y, x = divmod(cell, board_width)
            if probe.is_valid_placement(cid, float(x), float(y), 0):
                table[cid, cell] = True
    table[num_card_ids] = True          # permissive fallback row for the no-op

    # The one board state that does change legality: our own Princess Tower
    # dying frees its footprint (+9 cells each; enemy towers change nothing).
    # Probed only in a +/-2-cell window around each tower to avoid two more
    # full passes; tests/test_placement_mask_after_tower_loss.py compares
    # exhaustively to prove the window is wide enough.
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
                    continue                      # already legal
                if probe2.is_valid_placement(cid, float(x), float(y), 0):
                    freed[slot_i, cid, cell] = True
    return table, freed


def _own_princess_centres():
    """[(x, y), ...] for team 0's left and right Princess Towers, as cells, from
    ArenaLayout.
    """
    import clash_royale_env as _env
    y = int(_env.arena_princess_y(0))
    return [(int(_env.ARENA_LEFT_LANE_X), y),
            (int(_env.ARENA_RIGHT_LANE_X), y)]


#: Dilations of the context blocks appended to the CNN trunk.
#:
#: They run on the pooled 9x5 map, where a cell spans 4 input rows, so a 3x3
#: kernel at dilation d adds 8d rows of reach: 10 (base trunk) + 16 + 16 = 42
#: >= 34 covers the board. The reach lets a feature relate a unit at the bridge
#: to the tower it is walking at, which the placement head reads directly.
#: Width does not buy reach.
#:
#: (2, 2) rather than one d=4 block: same parameters and reach, but on a 9x5
#: map padding=4 makes the convolution ~5x slower (most of its work is
#: padding).
CONTEXT_DILATIONS = (2, 2)

#: Bottleneck width inside a context block (32 -> 16 -> 16 -> 32).
CONTEXT_BOTTLENECK = 16


class DilatedContextBlock(nn.Module):
    """Residual dilated bottleneck: `x + expand(relu(spread(relu(reduce(x)))))`.

    The final 1x1 is zero-initialised, so the block is exactly the identity at
    init and appending it to a trained trunk changes nothing until it learns.
    The zero gate also emits zero gradient, so a gradient-measured receptive
    field on a fresh net shows only the base trunk
    (tests/test_net_dilated_context.py). Shape-preserving, so the LSTM's input
    width does not change.
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


#: Widths of the scalar encoder's four branches. They must sum to
#: SCALAR_FEATURE_DIM, which is `lstm_input_dim - cnn_out_dim`; changing the
#: total reshapes the 1.8M-parameter LSTM and discards it on the next load.
#: `extra` expands its 9-10 inputs (clock, spends, tower HPs), which decide who
#: is winning.
ECON_BRANCH_DIM = 8
HAND_SLOT_EMBED_DIM = 5      # x hand_size (4) = 20
EXTRA_BRANCH_DIM = 12
CYCLE_BRANCH_DIM = 24
SCALAR_FEATURE_DIM = 64

#: Card-identity width inside the cycle branch. `seen` and `recency` share one
#: projection: they index the same card space, so both read the same notion of
#: a card.
CYCLE_EMBED_DIM = 16


class ScalarEncoder(nn.Module):
    """The scalar half of the observation, encoded in four independent branches
    (economy, hand, extra scalars, opponent cycle).

    It replaced one `Linear(1124, 64)`, where every output spanned every input,
    so gradient steps improving the hand encoding kept rewriting the rows the
    cycle was read through. Branches are concatenated with the cycle last, so
    `cycle_slice` names a contiguous block. Offsets come from the engine's
    bound values.
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

        # Elixir and the hand's costs: affordability is a joint function of the
        # two.
        self.econ = nn.Linear(1 + hand_size, ECON_BRANCH_DIM)
        # One projection shared across hand slots, which all hold the same kind
        # of thing.
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
        # Detached: the actor and critic read the branch but cannot reshape it,
        # which PPO otherwise did (turning it into a 1-D opponent-tempo
        # readout). `cycle_id_head` is its only gradient. Downstream layers
        # still learn to use these dims at full gradient.
        return torch.cat([e, h, x, self.cycle_branch(cycle).detach()], dim=1)

    def cycle_branch(self, cycle):
        """The cycle branch with its gradient attached. (B, CYCLE_BRANCH_DIM).

        `forward` detaches it for the LSTM; the auxiliary identity loss needs
        the live graph. The branch is not capacity-limited: trained for card
        identity, 24 dims keep ~96% of the achievable signal.
        """
        c = cycle.reshape(-1, 2, self.num_card_ids)
        c = F.relu(self.cycle_card(c)).flatten(1)
        return F.relu(self.cycle_out(c))


class MicroRoyaleNet(nn.Module):
    # 21 observation channels: 0-3 our units (melee/ranged/tank/buildings), 4-7
    # the opponent's, 8 river/bridges, 9-20 per-team attributes (the bound CH_*
    # constants).

    # The recurrent width, which callers need before they have a net (a fresh
    # (hx, cx)); policy_io re-exports it.
    LSTM_HIDDEN = 256

    def __init__(self, channels=None, board_width=None, board_height=None, hand_size=None, num_card_ids=None,
                 placement_rows=None, num_ability_slots=0,
                 context_dilations=CONTEXT_DILATIONS,
                 branched_scalars=True):
        super(MicroRoyaleNet, self).__init__()

        # Defaults come from the compiled engine.
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

        # Categorical over whole board cells; per-card legality is
        # placement_mask.
        self.placement_rows = placement_rows if placement_rows is not None else PLACEMENT_ROWS
        self.placement_cells = self.placement_rows * board_width
        # Rows allowed for an ordinary troop: our half.
        self.own_half_rows = min(OWN_HALF_ROWS, self.placement_rows)
        # A buffer rather than a parameter: an engine fact that moves with the
        # net.
        self.register_buffer("spell_flags", _spell_flags[:num_card_ids].clone())
        # 1.0 where the own-half row rule does not apply (spells and
        # deploy-anywhere troops); this is what placement_mask reads. Not
        # persisted, like the legality tables: a checkpoint copy could go stale
        # against the engine.
        self.register_buffer("row_free_flags",
                             _row_free_flags[:num_card_ids].clone(),
                             persistent=False)

        # Which cells the engine accepts per card (last row: permissive no-op
        # fallback). Static, so built once. Not persisted: a saved copy would
        # be a second copy of engine geometry.
        _legal_table, _freed_table = _build_placement_legality(
            num_card_ids, self.placement_rows, board_width)
        self.register_buffer("_placement_legal", _legal_table,
                             persistent=False)
        #: (2, num_card_ids+1, cells): cells freed when our own left/right
        #: Princess dies.
        self.register_buffer("_placement_freed", _freed_table,
                             persistent=False)
        #: Legality for each own-tower state, pre-OR-ed so the mask is a single
        #: gather.
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
        #: The same centres as flat indices into the observation: channel 3
        #: (ally buildings) starts at 3*H*W.
        _ch_ally_buildings = 3
        self._own_princess_flat = [
            _ch_ally_buildings * self.board_height * board_width + c
            for c in self._own_princess_cells]

        # 0 when the deck has no Champion: then there are no ability heads,
        # rather than two heads sampling noise into the PPO ratio and the
        # entropy bonus.
        self.num_ability_slots = num_ability_slots

        self.spatial_size = channels * board_height * board_width
        # Scalars: elixir, hand_size costs, hand_size card one-hots, then the
        # extra scalars (time, both sides' elixir spent, six tower HPs, the
        # elixir-phase multiplier). New sections are appended so existing
        # offsets stay valid.
        self.num_extra_scalars = clash_royale_env.ClashRoyaleEnv.NUM_EXTRA_SCALARS
        # The opponent's card cycle: seen[] and recency[], appended after the
        # extra scalars.
        self.cycle_block_size = clash_royale_env.ClashRoyaleEnv.CYCLE_BLOCK_SIZE
        self.scalar_size = (1 + hand_size + hand_size * num_card_ids
                            + self.num_extra_scalars + self.cycle_block_size)
        self.extra_start = 1 + hand_size + hand_size * num_card_ids
        # Forward offsets, never measured back from the end, which breaks on
        # the next append.
        self.cycle_start = self.extra_start + self.num_extra_scalars

        # 1. CNN over (Batch, NUM_CHANNELS, 34, 18). ceil_mode on both pools:
        #    floor would drop a row (34 -> 17 -> 8). The pre-flatten map also
        #    feeds the placement head.
        self.cnn_trunk = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )
        # Appended at the end: callers slice the trunk by index ([:2] for the
        # full-resolution branch), and inserting earlier would silently change
        # that tensor.
        self.context_dilations = tuple(context_dilations)
        for dilation in self.context_dilations:
            self.cnn_trunk.append(
                DilatedContextBlock(32, CONTEXT_BOTTLENECK, dilation))
        self.cnn_flatten = nn.Flatten()

        pooled_h = math.ceil(math.ceil(board_height / 2) / 2)
        pooled_w = math.ceil(math.ceil(board_width / 2) / 2)
        self.pooled_h, self.pooled_w = pooled_h, pooled_w
        self.cnn_out_dim = 32 * pooled_h * pooled_w

        # 2. Scalar encoder. Keeps the name `scalar_mlp`:
        #    expert_distill.TRUNK_MODULES finds modules by name.
        if branched_scalars:
            self.scalar_mlp = ScalarEncoder(
                hand_size, num_card_ids, self.num_extra_scalars,
                self.cycle_block_size, self.extra_start, self.cycle_start)
            scalar_feature_dim = self.scalar_mlp.out_dim
        else:
            # The monolithic layer it replaced, kept so the A/B and the
            # non-interference test can run it.
            self.scalar_mlp = nn.Sequential(
                nn.Linear(self.scalar_size, SCALAR_FEATURE_DIM),
                nn.ReLU()
            )
            scalar_feature_dim = SCALAR_FEATURE_DIM
        self.scalar_feature_dim = scalar_feature_dim

        # The cycle skip: the (detached) cycle block also bypasses the LSTM and
        # reaches the heads directly, since through the LSTM it is 1.6% of the
        # input and loses most of its card signal. 0 on the monolithic path;
        # `_split_cycle` / `_head_input` handle that, since `t[..., -0:]` is
        # the whole tensor. lstm_input_dim is unchanged.
        self.cycle_feature_dim = (CYCLE_BRANCH_DIM if branched_scalars else 0)
        head_dim = self.LSTM_HIDDEN + self.cycle_feature_dim

        # 2b. Card-identity embedding for autoregressive placement. Reads the
        # one-hot in obs, not the slot index: slots rotate, so an index
        # embedding could never learn a per-card placement. Linear on a one-hot
        # is an embedding lookup.
        self.card_id_embed = nn.Linear(num_card_ids, CARD_EMBED_DIM, bias=False)
        # A separate embedding for the no-op. The engine ignores the placement
        # then; this only keeps the placement distribution well-defined without
        # training any real card's conditioning on no-op steps.
        self.noop_embed = nn.Parameter(torch.zeros(CARD_EMBED_DIM))

        # 3. Memory.
        self.lstm_input_dim = self.cnn_out_dim + scalar_feature_dim
        # LSTMCell, so the loop steps each tick by hand.
        self.lstm = nn.LSTMCell(self.lstm_input_dim, self.LSTM_HIDDEN)

        # 4. Action heads. a. Card choice: hand_size slots plus the no-op (wait
        #    / bank elixir); reads hx and the cycle skip, never the card about
        #    to be chosen.
        self.card_head = nn.Linear(head_dim, hand_size + 1)

        # b. Placement: categorical over board cells, conditioned on hx and the
        # chosen card's embedding. The CNN's spatial map plus that context
        # broadcast over every cell is upsampled back to 34x18, so every cell's
        # logit comes from shared weights acting on its own board region.
        #
        # It replaced a Gaussian (whose entropy bonus pushed sigma up at a
        # constant rate; its noise ended up half the distance between the
        # lanes) and then a dense layer over all 612 cells (no weight sharing
        # between neighbouring cells).
        self.place_ctx = nn.Linear(head_dim + CARD_EMBED_DIM, 32)
        # Upsample + Conv, not ConvTranspose. A stride-2 ConvTranspose turns
        # the spatially uniform context into a fixed period-4 pattern on the
        # logit map, identical for every card and state (Odena et al.,
        # "Deconvolution and Checkerboard Artifacts", 2016). Channels narrow 32
        # -> 16 -> 8 to keep the full-resolution convolutions affordable.
        self.place_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),            # 9x5 -> 18x10
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),            # 18x10 -> 36x20
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),    # -> (B,1,36,20)
        )
        # 4c. Residual full-resolution branch. The coarse head works in
        # ~4x4-tile blocks with a uniform context, so which tile inside a block
        # wins is fixed across states. This branch reads the trunk's pre-pool
        # activation (16x34x18) with the same context and adds its logits. The
        # last conv is zero-initialised, so an existing checkpoint behaves
        # identically until the branch learns.
        self.place_ctx_hi = nn.Linear(head_dim + CARD_EMBED_DIM, HIRES_CTX_DIM)
        self.place_hires = nn.Sequential(
            nn.Conv2d(16 + HIRES_CTX_DIM, HIRES_HIDDEN,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(HIRES_HIDDEN, 1, kernel_size=3, stride=1, padding=1),
        )
        nn.init.zeros_(self.place_hires[-1].weight)
        nn.init.zeros_(self.place_hires[-1].bias)

        # The upsampled map is 4*pooled >= the board (ceil pooling). Cropping
        # the top-left corner is an exact alignment: pooled cell i covers
        # originals 2i and 2i+1 at each level.
        assert self.placement_rows == board_height, (
            "the convolutional placement head builds a board-sized map and crops it; "
            "a placement_rows different from board_height would break that alignment")

        # 5. Critic: reads hx and the cycle skip.
        self.value_head = nn.Linear(head_dim, 1)

        # 6. Champion ability heads (up to two Champions; deck slots 1 and 2),
        #    2 logits each (decline, activate). Only created when the deck has
        #    Champions.
        self.ability_slot1_head = nn.Linear(256, 2) if num_ability_slots >= 1 else None
        self.ability_slot2_head = nn.Linear(256, 2) if num_ability_slots >= 2 else None

        # 7. Auxiliary head: the opponent's next card. Plays no part in acting;
        #    its loss shapes what the LSTM keeps. The next card needs play
        #    history, unlike the opponent-elixir target it replaced (an affine
        #    function of two observed scalars; see
        #    test_aux_task_is_not_a_memory_probe.py). NUM_CARD_IDS-wide, since
        #    the opponent's deck is not known in general.
        self.aux_card_head = nn.Linear(self.LSTM_HIDDEN, NUM_CARD_IDS_LIVE)

        # 8. Identity head on the cycle branch: its only gradient (see
        #    ScalarEncoder.forward), asking whether each shown card's identity
        #    is still recoverable. `aux_card_head` stays on plain hx and must
        #    never read the skip, or it would answer from the branch and ask
        #    nothing of memory.
        self.cycle_id_head = (nn.Linear(CYCLE_BRANCH_DIM, NUM_CARD_IDS_LIVE)
                              if self.cycle_feature_dim else None)

    def _split_cycle(self, features):
        """The detached cycle block at the end of `features`, or None without a
        cycle branch.
        """
        if not self.cycle_feature_dim:
            return None
        # Detach again: the slice comes out of a cat whose other inputs carry a
        # graph. This makes the isolation local.
        return features[..., -self.cycle_feature_dim:].detach()

    def _head_input(self, hx, cycle_feat):
        """hx, plus the cycle skip when there is one. (Batch, head_dim)."""
        if cycle_feat is None:
            return hx
        return torch.cat((hx, cycle_feat), dim=-1)

    def cycle_features(self, obs, detached=False):
        """The cycle branch on a raw observation, with gradient by default; None
        without a branch.
        """
        if not self.cycle_feature_dim:
            return None
        scalar_obs = obs[:, self.spatial_size:]
        cycle = scalar_obs[:, self.cycle_start:
                           self.cycle_start + self.cycle_block_size]
        out = self.scalar_mlp.cycle_branch(cycle)
        return out.detach() if detached else out

    def predict_cycle_card(self, cycle_feat):
        """Logits over card ids from the cycle branch. (Batch, C)."""
        return self.cycle_id_head(cycle_feat)

    def extract_features_hires(self, obs):
        """The non-recurrent part of the network: CNN, scalar encoder and card
        embeddings. Can run on a flattened (T*N) batch at once.

        obs: (Batch, obs_dim). Returns (combined, card_embeds, spatial_map,
        hires_map):
          combined: (Batch, lstm_input_dim), the LSTM input.
          card_embeds: (Batch, hand_size+1, CARD_EMBED_DIM), one per hand slot plus the no-op; indexed only after the card is chosen, which is what allows autoregressive placement.
          spatial_map: (Batch, 32, pooled_h, pooled_w), the CNN map before flattening.
          hires_map: (Batch, 16, board_height, board_width), the activation before the first pool, for the full-resolution branch.
        """
        # Spatial then scalar, following observationSize() in C++.
        spatial_obs = obs[:, :self.spatial_size].view(-1, self.channels, self.board_height, self.board_width)
        scalar_obs = obs[:, self.spatial_size:]

        # The trunk runs in two halves to expose the pre-pool activation;
        # bit-identical to the Sequential
        # (test_trunk_split_is_bit_identical_to_the_sequential).
        hires_map = self.cnn_trunk[:2](spatial_obs)
        spatial_map = self.cnn_trunk[2:](hires_map)
        cnn_features = self.cnn_flatten(spatial_map)
        scalar_features = self.scalar_mlp(scalar_obs)
        combined = torch.cat((cnn_features, scalar_features), dim=1)

        # scalar_obs: [elixir(1), costs(hand_size),
        # onehots(hand_size*num_card_ids)].
        onehot_start = 1 + self.hand_size
        card_onehots = scalar_obs[:, onehot_start:onehot_start + self.hand_size * self.num_card_ids]
        card_onehots = card_onehots.view(-1, self.hand_size, self.num_card_ids)
        card_embeds = self.card_id_embed(card_onehots)  # (Batch, hand_size, CARD_EMBED_DIM)

        noop = self.noop_embed.view(1, 1, -1).expand(card_embeds.shape[0], 1, -1)
        card_embeds = torch.cat([card_embeds, noop], dim=1)  # (Batch, hand_size+1, CARD_EMBED_DIM)

        return combined, card_embeds, spatial_map, hires_map

    def extract_features(self, obs):
        """The first three return values of extract_features_hires.

        For the many callers that unpack three; placement_given_card then
        rebuilds the hi-res map from obs (identical,
        test_recomputed_hires_equals_the_passed_one).
        """
        combined, card_embeds, spatial_map, _ = self.extract_features_hires(obs)
        return combined, card_embeds, spatial_map

    def hires_features(self, obs):
        """The trunk's activation before pooling: (Batch, 16, 34, 18), the
        full-resolution branch's input.
        """
        spatial_obs = obs[:, :self.spatial_size].view(
            -1, self.channels, self.board_height, self.board_width)
        return self.cnn_trunk[:2](spatial_obs)

    def affordability_mask(self, obs):
        """Which hand slots are playable now, from the elixir and costs in the
        observation. (Batch, hand_size+1) bool; the no-op column is always
        True.

        GameManager::playCard silently ignores an unaffordable card, so without
        this most sampled actions changed nothing and their advantages were
        pure noise. Computed from the stored observation, so rollout and update
        produce the same mask.
        """
        scalar_obs = obs[:, self.spatial_size:]
        # Elixir and costs are both /10 in the observation, so no rescaling is
        # needed.
        elixir = scalar_obs[:, 0:1]
        costs = scalar_obs[:, 1:1 + self.hand_size]
        # cost <= 0 marks an empty slot, never a free card.
        playable = (costs > 0.0) & (costs <= elixir + 1e-6)
        noop = torch.ones(obs.shape[0], 1, dtype=torch.bool, device=obs.device)
        return torch.cat([playable, noop], dim=1)

    def hand_card_ids(self, obs):
        """The card id in each hand slot, from the observation's one-hot. (Batch,
        hand_size) long; -1 for an empty slot.

        Read from the observation the decision was made on: the hand rotates as
        soon as a card is played.
        """
        scalar_obs = obs[:, self.spatial_size:]
        onehot_start = 1 + self.hand_size
        onehots = scalar_obs[:, onehot_start:onehot_start + self.hand_size * self.num_card_ids]
        onehots = onehots.view(-1, self.hand_size, self.num_card_ids)
        ids = onehots.argmax(dim=-1)
        # An empty one-hot would argmax to 0, a real card id (Knight).
        return torch.where(onehots.sum(dim=-1) > 0.5, ids, torch.full_like(ids, -1))

    def elixir_from_obs(self, obs):
        """Current elixir (0..10); the observation stores it /10."""
        return obs[:, self.spatial_size] * 10.0

    def hand_costs_from_obs(self, obs):
        """Hand card costs (0..10), a (Batch, hand_size) tensor, from the scalars
        after the elixir.
        """
        s = self.spatial_size
        return obs[:, s + 1:s + 1 + self.hand_size] * 10.0

    def step_lstm_and_card(self, features, hidden_state, card_mask=None):
        """First half of the recurrent step: advance the LSTM and compute the card
        choice, state value and ability logits, none of which depend on the
        card about to be chosen.

        features: (Batch, lstm_input_dim)
        card_mask: (Batch, hand_size+1) bool from affordability_mask, or None
        (unmasked).
        """
        hx, cx = self.lstm(features, hidden_state)
        # The cycle skip, taken from `features` so it matches what the LSTM was
        # fed.
        head_in = self._head_input(hx, self._split_cycle(features))
        card_logits = self.card_head(head_in)
        if card_mask is not None:
            # -inf, not a large negative: a finite value leaves an illegal
            # action a small probability and an entropy contribution. The no-op
            # is always legal, so no row is all -inf.
            card_logits = card_logits.masked_fill(~card_mask, float("-inf"))
        state_value = self.value_head(head_in)
        # None without Champions.
        ability_slot1_logits = self.ability_slot1_head(hx) if self.ability_slot1_head is not None else None
        ability_slot2_logits = self.ability_slot2_head(hx) if self.ability_slot2_head is not None else None
        return card_logits, ability_slot1_logits, ability_slot2_logits, state_value, (hx, cx)

    def placement_given_card(self, hx, card_embeds, card_idx, obs=None,
                             spatial_map=None, hires_map=None,
                             ctx=None, ctx_hi=None, cycle_feat=None):
        """Placement logits (Batch, placement_cells) conditioned on card_idx: the
        autoregressive step.

        hx: (Batch, 256). card_embeds: (Batch, hand_size+1, CARD_EMBED_DIM).
        card_idx: (Batch,) long in [0, hand_size]; at update time always the
        stored action.

        Joint over all cells rather than p(x)p(y), which cannot express "near
        the left bridge or in the back-right corner" without mass on the mixed
        combinations. placement_cells = 34 rows x 18 = 612. `cell_to_xy`
        converts a cell to coordinates.
        """
        if spatial_map is None:
            raise ValueError(
                "placement_given_card needs the spatial_map from extract_features. "
                "Passing None here would produce different logits from the ones the "
                "rollout produced, and the PPO ratio would break silently -- so this "
                "is an error, not a default.")

        if hires_map is None and obs is None:
            # No fallback to the coarse-only head: a different function from
            # the rollout's would silently break the PPO ratio.
            raise ValueError(
                "placement_given_card needs hires_map, or obs to build it "
                "(see hires_features). None in both would compute a different "
                "head from the one that ran the rollout.")

        batch_idx = torch.arange(card_embeds.shape[0], device=card_embeds.device)
        chosen_embed = card_embeds[batch_idx, card_idx]  # (Batch, CARD_EMBED_DIM)
        # Context broadcast over every cell (added, not concatenated): the
        # situation and card shift the map; local structure stays in the map.
        #
        # ctx/ctx_hi are passed in only by forward_sequence's row compaction,
        # which computes them over the whole batch and slices, because GEMM
        # results depend on batch size on this backend; that keeps the kept
        # rows' logits bit-identical. Conv2d's weight gradients also depend on
        # batch size at some shapes, so compaction cannot make the weights
        # bit-identical, only the logits and the loss.
        #
        # The cycle skip: supplied, else rebuilt from obs, else an error, never
        # a silent zero.
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
            ctx = self.place_ctx(torch.cat((head_in, chosen_embed), dim=-1))  # (B, 32)
        h = spatial_map + ctx.view(-1, 32, 1, 1)
        logit_map = self.place_up(h)                                    # (B, 1, 4*ph, 4*pw)
        logits = logit_map[:, 0, :self.placement_rows, :self.board_width].reshape(
            -1, self.placement_cells)

        # The full-resolution branch, added as a residual; exactly zero at
        # init.
        if hires_map is None:
            hires_map = self.hires_features(obs)
        if ctx_hi is None:
            ctx_hi = self.place_ctx_hi(torch.cat((head_in, chosen_embed), dim=-1))
        h_hi = torch.cat(
            (hires_map,
             ctx_hi.view(-1, HIRES_CTX_DIM, 1, 1).expand(
                 -1, -1, hires_map.shape[2], hires_map.shape[3])), dim=1)
        fine = self.place_hires(h_hi)                                   # (B, 1, H, W)
        logits = logits + fine[:, 0, :self.placement_rows, :self.board_width].reshape(
            -1, self.placement_cells)
        if obs is not None:
            # Per-card legality mask, -inf for the same reason as the card
            # mask. At least one cell is always legal.
            logits = logits.masked_fill(~self.placement_mask(obs, card_idx), float("-inf"))
        return logits

    def forward_sequence(self, feats_seq, card_embeds_seq, spatial_seq, obs_seq,
                         card_mask_seq, card_idx_seq, reset_seq, hidden_state,
                         extra_card_idx_seq=None, hires_seq=None,
                         active_rows=None, with_ability=False):
        """A batched equivalent of looping forward_from_features over timesteps
        (bit-identity tested).

        Only the LSTM is recurrent; the heads run once over all L*B rows
        instead of L times on B.

        feats_seq/obs_seq/card_mask_seq/card_idx_seq/reset_seq: (L, B, ...).
        Returns card_logits (L,B,hand+1), place_logits (L,B,cells), values
        (L,B), aux_card_logits (L,B,C), the final hidden state, and the
        coverage-pass logits (or None); with_ability=True appends the ability
        logits.
        """
        L, B = feats_seq.shape[0], feats_seq.shape[1]
        hx, cx = hidden_state
        hx_steps = []
        for l in range(L):
            hx, cx = self.lstm(feats_seq[l], (hx, cx))
            # Collected before the end-of-episode reset, as in the per-step
            # loop.
            hx_steps.append(hx)
            reset = reset_seq[l].unsqueeze(1)
            hx = hx * reset
            cx = cx * reset

        flat_hx = torch.stack(hx_steps).reshape(L * B, -1)
        # The cycle skip, read off the same features the LSTM consumed.
        flat_cycle = self._split_cycle(feats_seq.reshape(L * B, -1))
        flat_head = self._head_input(flat_hx, flat_cycle)
        card_logits = self.card_head(flat_head)
        mask_flat = card_mask_seq.reshape(L * B, -1)
        card_logits = card_logits.masked_fill(~mask_flat, float("-inf"))
        values = self.value_head(flat_head).squeeze(-1)
        # Next-card logits from flat_hx, not flat_head: the head exists to
        # pressure the recurrence.
        aux = self.aux_card_head(flat_hx)

        # Shared by both placement calls, so conv1 runs once.
        flat_obs = obs_seq.reshape(L * B, -1)
        flat_hires = (self.hires_features(flat_obs) if hires_seq is None
                      else hires_seq.reshape(L * B, *hires_seq.shape[2:]))
        flat_embeds = card_embeds_seq.reshape(L * B, *card_embeds_seq.shape[2:])
        flat_spatial = spatial_seq.reshape(L * B, *spatial_seq.shape[2:])

        # Row compaction: every consumer of the placement outputs is
        # decision-masked, so the head (twice: chosen card and coverage slot)
        # runs only on active rows. active_rows=None keeps the full
        # computation. The fill is zero, not -inf: an all -inf row has nan
        # entropy, and nan * 0 poisons masked sums, while any finite value
        # times 0 is exactly 0.
        def _placement(idx_seq):
            flat_idx = idx_seq.reshape(L * B)
            if active_rows is None:
                return self.placement_given_card(
                    flat_hx, flat_embeds, flat_idx, flat_obs, flat_spatial,
                    hires_map=flat_hires, cycle_feat=flat_cycle)
            out = flat_hx.new_zeros(L * B, self.placement_cells)
            if active_rows.numel() == 0:
                # An all-forced chunk (common for a spent-down agent) would
                # hand Conv2d an empty batch.
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

        # Optional coverage pass: the placement map of another card on the same
        # state, so unplayed cards keep receiving placement gradient (see
        # rl/coverage.py). Same head, no new parameters.
        extra_logits = None
        if extra_card_idx_seq is not None:
            extra_logits = _placement(extra_card_idx_seq).view(L, B, -1)

        out = (card_logits.view(L, B, -1), place_logits.view(L, B, -1),
                values.view(L, B), aux.view(L, B, -1), (hx, cx), extra_logits)
        if not with_ability:
            return out
        # One (L, B, 2) tensor per Champion head, off flat_hx, as at rollout
        # time.
        ability = [h(flat_hx).view(L, B, 2)
                   for h in (self.ability_slot1_head, self.ability_slot2_head)
                   if h is not None]
        return out + (ability,)

    def predict_opp_next_card(self, hx):
        """Logits over card ids for the opponent's next play. (Batch, C).

        Used only by the PPO update's auxiliary loss, whose gradient shapes the
        LSTM and CNN.
        """
        return self.aux_card_head(hx)

    def placement_mask(self, obs, card_idx):
        """Legal board cells for the chosen card. (Batch, placement_cells) bool.

        Ordinary troops get our half; spells and deploy-anywhere troops the
        whole board (row_free_flags, read through the one-hot so it works on a
        batch). Then the engine-derived legality table narrows it to cells
        playCard actually accepts, since a rejected placement is silently a
        no-op with a noise advantage. The no-op returns the permissive row so
        its distribution stays defined.
        """
        batch = obs.shape[0]
        scalar_obs = obs[:, self.spatial_size:]
        onehot_start = 1 + self.hand_size
        onehots = scalar_obs[:, onehot_start:onehot_start + self.hand_size * self.num_card_ids]
        onehots = onehots.view(batch, self.hand_size, self.num_card_ids)
        # (Batch, hand_size): 1.0 where the slot's card ignores the own-half
        # row rule (row_free_flags, not spell_flags, or a Miner is confined to
        # its half).
        slot_is_spell = (onehots * self.row_free_flags.view(1, 1, -1)).sum(dim=-1)
        # The no-op slot is never exempt.
        slot_is_spell = torch.cat(
            [slot_is_spell, torch.zeros(batch, 1, device=obs.device, dtype=slot_is_spell.dtype)], dim=1)
        chosen_is_spell = slot_is_spell.gather(1, card_idx.view(-1, 1)).squeeze(1) > 0.5  # (Batch,)

        rows = torch.arange(self.placement_rows, device=obs.device).view(1, -1, 1)
        own_half = rows < self.own_half_rows                       # (1, rows, 1)
        full_board = torch.ones_like(own_half)
        allowed_rows = torch.where(chosen_is_spell.view(-1, 1, 1), full_board, own_half)
        mask = allowed_rows.expand(batch, self.placement_rows, self.board_width)
        mask = mask.reshape(batch, self.placement_cells)

        # Narrow to what the engine accepts: dead zone and tower footprints
        # included.
        if self._placement_legal is not None:
            table = self._placement_legal.to(obs.device)
            slot_ids = onehots.argmax(dim=-1)                       # (B, hand)
            # An empty slot's all-zero one-hot argmaxes to 0, a real card id.
            slot_empty = onehots.sum(dim=-1) <= 0.0                 # (B, hand)
            slot_ids = torch.where(slot_empty, torch.full_like(slot_ids, -1), slot_ids)
            noop = torch.full((batch, 1), -1, device=obs.device, dtype=slot_ids.dtype)
            slot_ids = torch.cat([slot_ids, noop], dim=1)
            chosen_id = slot_ids.gather(1, card_idx.view(-1, 1)).squeeze(1)
            # The last table row is the permissive fallback for the no-op or an
            # empty slot.
            chosen_id = torch.where(chosen_id < 0,
                                    torch.full_like(chosen_id, table.shape[0] - 1),
                                    chosen_id)
            # Select legality by which of our Princess Towers stand. Liveness
            # is the ally-building channel at each tower's centre cell, which
            # is > 0 exactly while it is alive.
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
        """{(x, y)} that our Princess `tower_slot` (1=LEFT, 2=RIGHT) frees for
        `card_id` when it dies. For diagnostics and tests.
        """
        if self._placement_freed is None:
            return set()
        row = self._placement_freed[tower_slot - 1, card_id]
        return {(c % self.board_width, c // self.board_width)
                for c in torch.nonzero(row, as_tuple=True)[0].tolist()}

    def cell_to_xy(self, cell_idx):
        """Cell index -> the board coordinates the engine accepts: (x, y) float
        tensors of the same shape.

        Row-major: x = column, y = row, exact integers inside the board, so no
        clamp is needed.
        """
        row = torch.div(cell_idx, self.board_width, rounding_mode="floor")
        col = cell_idx % self.board_width
        return col.float(), row.float()

    def forward_from_features(self, features, card_embeds, hidden_state, card_idx,
                              card_mask=None, obs=None, spatial_map=None):
        """Both halves together, for when card_idx is already known.

        In the PPO update pass the stored card_idx and the rollout's card_mask
        (recomputed from the same obs), or the ratio breaks. During rollout
        call step_lstm_and_card and placement_given_card separately.
        """
        (card_logits, ability_slot1_logits, ability_slot2_logits, state_value,
         (hx, cx)) = self.step_lstm_and_card(features, hidden_state, card_mask)
        placement_logits = self.placement_given_card(hx, card_embeds, card_idx, obs, spatial_map)
        return (card_logits, placement_logits, state_value,
                ability_slot1_logits, ability_slot2_logits, (hx, cx))
