"""Where the phase-1 training hour goes.

A measurement instrument, not imported by the training path. Every hook is
applied to a live object at runtime, so profiling cannot change what a run
computes. It never writes a real checkpoint, TensorBoard event or replay;
`--out` is a scratch directory.

The loop's costs are fused inside single pybind calls:

    (1) engine physics            GameManager::step()
    (2) the C++/Python boundary   observation build + std::vector -> Python list
    (3) the model                 MicroRoyaleNet forward / backward
    (4) overhead                  resets, replay logging, GAE, buffer, IPC

`env.step_self_play(...)` is (1) + (2) in one number, so this measures every
boundary crossing with counts, and `tools/audit/engine_profile.cpp` measures
(1) and the C++ half of (2) with no interpreter. The difference is the pybind
marshalling cost; run both.

The ledger tracks inclusive and exclusive time: a nested span subtracts itself
from its parent, so the exclusive column sums to the measured total.

    --mode sync      every env in this process: exact attribution, not real throughput
    --mode async     the real AsyncVectorEnv: true episodes/hour, coarse split only
    --mode torch     the model alone, with a stubbed engine if no .pyd is present
    --mode boundary  the Python half of the observation boundary

Run sync for attribution and async for throughput, and reconcile them.

    python_ai/venv/Scripts/python.exe python_ai/tools/profile_training.py --mode sync  --episodes 60
    python_ai/venv/Scripts/python.exe python_ai/tools/profile_training.py --mode async --episodes 60
    python_ai/venv/Scripts/python.exe python_ai/tools/profile_training.py --mode torch --updates 3
    python_ai/venv/Scripts/python.exe python_ai/tools/profile_training.py --mode sync --episodes 20 --cprofile
"""
import argparse
import dataclasses
import os
import re
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

# Put the repo root on sys.path for a file run; tests/test_package_layout.py
# checks every runnable script does.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# --- the ledger ---
class Ledger:
    """Inclusive + exclusive time and a call count, per key.

    `_stack` holds one accumulator per open span; a closing span adds its
    duration to its parent's accumulator, which the parent subtracts from its
    own self-time. So sum(exclusive) equals the outermost span's wall clock.
    """

    def __init__(self):
        self.incl = defaultdict(float)
        self.excl = defaultdict(float)
        self.count = defaultdict(int)
        self._stack = []
        self.enabled = True

    @contextmanager
    def span(self, key):
        if not self.enabled:
            yield
            return
        self._stack.append(0.0)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            child = self._stack.pop()
            self.incl[key] += dt
            self.excl[key] += dt - child
            self.count[key] += 1
            if self._stack:
                self._stack[-1] += dt

    def add(self, key, seconds, n=1):
        self.incl[key] += seconds
        self.excl[key] += seconds
        self.count[key] += n

    def calibrate(self, n=20000):
        """This instrument's own per-span cost, reported rather than assumed
        negligible.
        """
        sink = Ledger()
        t0 = time.perf_counter()
        for _ in range(n):
            with sink.span("x"):
                pass
        return (time.perf_counter() - t0) / n


# --- the boundary proxy ---
# Category per pybind method. Anything unlisted lands in `<prefix>.info`, so a
# new binding shows up as its own line.
_OBS_METHODS = {"get_observation_for_team", "get_observation", "extract_observation"}
_STEP_METHODS = {"step", "step_self_play", "step_self_play_fast"}
_SNAPSHOT_METHODS = {"snapshot"}
_RESET_METHODS = {"reset", "seed"}


class EngineProxy:
    """Times every call across the pybind boundary, and follows snapshots.

    A teacher rollout works on the object `snapshot()` returns, so that object
    is wrapped under a different prefix, separating "the live match stepped"
    (once per decision) from "a candidate was rolled forward" (K+1 times).
    Attribute lookups are cached.
    """

    __slots__ = ("_obj", "_led", "_prefix", "_cache")

    def __init__(self, obj, led, prefix):
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_led", led)
        object.__setattr__(self, "_prefix", prefix)
        object.__setattr__(self, "_cache", {})

    def _category(self, name):
        p = self._prefix
        if name in _OBS_METHODS:
            return p + ".obs"
        if name in _STEP_METHODS:
            return p + ".step"
        if name in _SNAPSHOT_METHODS:
            return p + ".snapshot"
        if name in _RESET_METHODS:
            return p + ".reset"
        return p + ".info"

    def __getattr__(self, name):
        cache = self._cache
        hit = cache.get(name)
        if hit is not None:
            return hit

        attr = getattr(self._obj, name)
        if not callable(attr):
            return attr

        led = self._led
        key = self._category(name)
        is_snapshot = name in _SNAPSHOT_METHODS

        def timed(*args, **kwargs):
            with led.span(key):
                out = attr(*args, **kwargs)
            if is_snapshot:
                # Everything a rollout does happens on this object.
                return EngineProxy(out, led, "rollout")
            return out

        timed.__name__ = name
        cache[name] = timed
        return timed


# --- reporting ---
def report(led, wall, episodes, label, extra_notes=()):
    print()
    print("=" * 86)
    print(f"  {label}")
    print("=" * 86)
    print(f"  wall clock            {wall:10.2f} s")
    print(f"  episodes completed    {episodes:10d}")
    if episodes:
        print(f"  ms / episode          {1000.0 * wall / episodes:10.1f}")
        print(f"  episodes / hour       {3600.0 * episodes / wall:10.0f}")
    print()
    print(f"  {'category':<28} {'self s':>9} {'self %':>7} "
          f"{'incl s':>9} {'calls':>10} {'us/call':>9}")
    print("  " + "-" * 82)

    rows = sorted(led.excl.items(), key=lambda kv: -kv[1])
    accounted = 0.0
    for key, secs in rows:
        if secs <= 0.0 and led.count[key] == 0:
            continue
        accounted += secs
        n = led.count[key]
        per = 1e6 * led.incl[key] / n if n else 0.0
        print(f"  {key:<28} {secs:9.3f} {100.0 * secs / wall:6.1f}% "
              f"{led.incl[key]:9.3f} {n:10d} {per:9.2f}")
    print("  " + "-" * 82)
    print(f"  {'ACCOUNTED':<28} {accounted:9.3f} {100.0 * accounted / wall:6.1f}%")
    print(f"  {'UNATTRIBUTED':<28} {wall - accounted:9.3f} "
          f"{100.0 * (wall - accounted) / wall:6.1f}%")
    for note in extra_notes:
        print(f"  * {note}")


# --- mode: sync / async: the real trainer, bounded ---
def build_profiled_trainer(led, mode, out_dir, num_envs, update_timestep,
                           max_episodes, teacher_stage):
    """A Phase1Trainer subclass with timers around each phase.

    Subclassed so trainers/train.py is untouched. `collect_rollout` and
    `run_update` may not be overridden (BaseTrainer forbids it), so they are
    wrapped from outside.
    """
    import gymnasium as gym
    import torch

    from python_ai.envs import gym_wrapper
    from python_ai.rl.config import PPOConfig
    from python_ai.trainers import train as train_mod

    # PPOConfig is frozen; `replace` re-runs __post_init__, so a bad
    # update_timestep fails here.
    cfg = dataclasses.replace(PPOConfig(), num_envs=num_envs,
                              update_timestep=update_timestep)

    def _make():
        def _init():
            env = gym_wrapper.MicroRoyaleEnv(
                {"opponent": train_mod.PHASE1_OPPONENT,
                 "teacher_stage": teacher_stage})
            if mode == "sync":
                # Only in-process: an async worker's ledger would never come
                # back.
                env.game = EngineProxy(env.game, led, "live")
            return env
        return _init

    class ProfiledTrainer(train_mod.Phase1Trainer):
        """Phase1Trainer with timers, a scratch checkpoint and a bounded run.

        The redirect must happen in __init__: Phase1Trainer.__init__ sets
        `self.weight_path` as an instance attribute, which would override a
        class attribute here and point the profiler at the live checkpoint.
        """

        def __init__(self, cfg):
            super().__init__(cfg)
            self.weight_path = os.path.join(out_dir, "profile_scratch.pth")
            self.log_dir = os.path.join(out_dir, "tb")

        def build_envs(self):
            if mode == "sync":
                return gym.vector.SyncVectorEnv([_make() for _ in range(num_envs)])
            return gym.vector.AsyncVectorEnv([_make() for _ in range(num_envs)])

        def load_checkpoint(self):
            # Profile from a cold net; the per-step cost does not depend on the
            # weights.
            return False

        def save_checkpoint(self, verbose=True):
            with led.span("overhead.checkpoint"):
                pass

        def record_replay(self):
            with led.span("overhead.replay"):
                super().record_replay()

        def log_update(self, stats):
            with led.span("overhead.logging"):
                super().log_update(stats)

        def should_stop(self):
            return self.episodes_completed >= max_episodes

        def periodic(self, stats):
            with led.span("overhead.periodic"):
                super().periodic(stats)

    return ProfiledTrainer(cfg), torch


def instrument_trainer(trainer, led, torch_mod, mode):
    """Wrap the phases from outside, so no override changes the arithmetic."""
    net = trainer.net

    # The two top-level phases.
    raw_collect = trainer.collect_rollout
    raw_update = trainer.run_update

    def collect():
        with led.span("PHASE.rollout"):
            return raw_collect()

    def update():
        with led.span("PHASE.update"):
            return raw_update()

    trainer.collect_rollout = collect
    trainer.run_update = update

    # The vector env. In sync mode its exclusive time is gym's own plumbing; in
    # async mode the workers' internals are invisible.
    envs = trainer.envs
    raw_step, raw_reset = envs.step, envs.reset

    def env_step(a):
        with led.span("envs.step"):
            return raw_step(a)

    def env_reset(*a, **k):
        with led.span("envs.reset"):
            return raw_reset(*a, **k)

    envs.step, envs.reset = env_step, env_reset

    # The model.
    for name, key in (("extract_features", "torch.fwd.trunk"),
                      ("step_lstm_and_card", "torch.fwd.lstm_card"),
                      ("placement_given_card", "torch.fwd.placement"),
                      ("affordability_mask", "torch.fwd.mask")):
        raw = getattr(net, name)

        def make(raw=raw, key=key):
            def timed(*a, **k):
                with led.span(key):
                    return raw(*a, **k)
            return timed
        setattr(net, name, make())

    raw_upd = trainer.updater.update

    def upd(*a, **k):
        with led.span("torch.ppo_update"):
            return raw_upd(*a, **k)
    trainer.updater.update = upd

    # The advisor coverage target: numpy work per step.
    raw_cov = trainer._draw_coverage

    def cov(*a, **k):
        with led.span("overhead.coverage"):
            return raw_cov(*a, **k)
    trainer._draw_coverage = cov


def run_loop_mode(args, led):
    scratch = args.out
    os.makedirs(scratch, exist_ok=True)

    trainer, torch_mod = build_profiled_trainer(
        led, args.mode, scratch, args.num_envs, args.update_timestep,
        args.episodes, args.teacher_stage)

    print(f"[profile] mode={args.mode}  num_envs={args.num_envs}  "
          f"update_timestep={args.update_timestep}  "
          f"teacher_stage={args.teacher_stage}  target_episodes={args.episodes}")
    print(f"[profile] torch threads: {torch_mod.get_num_threads()}  "
          f"interop: {torch_mod.get_num_interop_threads()}")

    trainer.setup()
    instrument_trainer(trainer, led, torch_mod, args.mode)

    # setup() is startup cost amortised over a run, so it is excluded from the
    # wall clock.
    led.incl.clear()
    led.excl.clear()
    led.count.clear()

    t0 = time.perf_counter()
    while not trainer.should_stop():
        trainer.collect_rollout()
        stats = trainer.run_update()
        trainer.log_update(stats)
        trainer.buffer.clear()
        trainer._hx, trainer._cx = trainer._hx.detach(), trainer._cx.detach()
        trainer.periodic(stats)
    wall = time.perf_counter() - t0

    notes = []
    if args.mode == "async":
        notes.append("ASYNC: everything inside a worker process is invisible "
                     "here. `envs.step` is 8 workers in parallel plus IPC; it "
                     "is NOT the sum of their engine work.")
    else:
        notes.append("SYNC: the 8 envs ran sequentially in this process, so "
                     "ms/episode is HIGHER than a real run. The BREAKDOWN is "
                     "the result; take throughput from --mode async.")
    notes.append(f"instrument overhead ~{1e6 * led.calibrate():.2f} us/span; "
                 f"{sum(led.count.values())} spans recorded")

    report(led, wall, trainer.episodes_completed,
           f"phase-1 training loop -- mode={args.mode}", notes)

    try:
        trainer.envs.close()
    except Exception:
        pass
    return wall, trainer.episodes_completed


# --- mode: torch: the model alone, on a box with no engine ---
_CONST_RE = r"static\s+constexpr\s+(?:int|float)\s+{}\s*=\s*([0-9.]+)"


def _engine_constants_from_header():
    """Read the constants off ClashEnv.h, since a stub cannot ask the bindings. A
    missing name raises rather than falling back to a literal.
    """
    path = os.path.join(_ROOT, "include", "core", "ClashEnv.h")
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        src = fh.read()
    names = ["BOARD_WIDTH", "BOARD_HEIGHT", "NUM_CHANNELS", "HAND_SIZE",
             "NUM_CARD_IDS", "NUM_EXTRA_SCALARS", "MAX_TROOP_HP",
             "MAX_BUILDING_HP"]
    out = {}
    for n in names:
        m = re.search(_CONST_RE.format(n), src)
        if not m:
            raise RuntimeError(
                f"{n} not found in {path}. The stub reads the header so it "
                f"cannot go stale; a rename must be followed here, not "
                f"papered over with a literal.")
        out[n] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
    return out


def install_engine_stub():
    """A constants-only `clash_royale_env`, for --mode torch with no .pyd.

    Enough for models/net.py to build a real MicroRoyaleNet; it cannot simulate
    anything.
    """
    import types

    K = _engine_constants_from_header()
    mod = types.ModuleType("clash_royale_env")

    class _Stub:
        BOARD_WIDTH = K["BOARD_WIDTH"]
        BOARD_HEIGHT = K["BOARD_HEIGHT"]
        NUM_CHANNELS = K["NUM_CHANNELS"]
        HAND_SIZE = K["HAND_SIZE"]
        NUM_CARD_IDS = K["NUM_CARD_IDS"]
        NUM_EXTRA_SCALARS = K["NUM_EXTRA_SCALARS"]
        MAX_TROOP_HP = K["MAX_TROOP_HP"]
        MAX_BUILDING_HP = K["MAX_BUILDING_HP"]

        def __init__(self, *a, **k):
            pass

        # River band [15.5, 17.5): the last whole own-half row is 15.
        def get_own_half_max_y(self):
            return 15.0

        def get_max_placement_x(self):
            return float(self.BOARD_WIDTH - 1)

        def observation_size(self):
            return (self.BOARD_WIDTH * self.BOARD_HEIGHT * self.NUM_CHANNELS
                    + 1 + self.HAND_SIZE
                    + self.HAND_SIZE * self.NUM_CARD_IDS
                    + self.NUM_EXTRA_SCALARS)

        def reset(self):
            return [0.0] * self.observation_size()

        def is_valid_placement(self, card_id, x, y, team=0):
            # Shape-faithful, not rule-faithful: only the cost of building the
            # mask table is measured.
            info = mod.get_card_info(card_id)
            if info["is_spell"]:
                return 0.0 <= x <= self.get_max_placement_x() and 0.0 <= y < self.BOARD_HEIGHT
            return 0.0 <= x <= self.get_max_placement_x() and 0.0 <= y <= 15.0

    _IDS = list(range(1, K["NUM_CARD_IDS"]))

    def get_all_card_ids():
        return _IDS

    def get_card_info(cid):
        return {"cost": 3 + (cid % 3), "is_spell": (cid % 11 == 0),
                "name": f"stub{cid}", "id": cid}

    mod.ClashRoyaleEnv = _Stub
    mod.get_all_card_ids = get_all_card_ids
    mod.get_card_info = get_card_info
    sys.modules["clash_royale_env"] = mod
    return K


def run_torch_mode(args, led):
    """Forward + backward at the shapes the real loop uses.

    The real MicroRoyaleNet at the rollout batch (`num_envs`) and the real BPTT
    chunk, since the manual LSTM loop and the convolutional placement head are
    both shape-sensitive.
    """
    stubbed = False
    try:
        import clash_royale_env  # noqa: F401
        print("[profile] using the REAL clash_royale_env")
    except Exception as exc:
        K = install_engine_stub()
        stubbed = True
        print(f"[profile] no engine ({type(exc).__name__}); installed a "
              f"constants-only stub read from include/core/ClashEnv.h: "
              f"{K['BOARD_WIDTH']}x{K['BOARD_HEIGHT']}x{K['NUM_CHANNELS']}, "
              f"NUM_CARD_IDS={K['NUM_CARD_IDS']}")

    import torch
    from torch.distributions import Categorical

    from python_ai.models.net import MicroRoyaleNet
    from python_ai.rl.config import PPOConfig

    cfg = dataclasses.replace(PPOConfig(), num_envs=args.num_envs)
    B = args.num_envs
    net = MicroRoyaleNet(num_ability_slots=0)
    obs_size = net.observation_size if hasattr(net, "observation_size") else None
    print(f"[profile] torch {torch.__version__}  threads={torch.get_num_threads()}  "
          f"params={sum(p.numel() for p in net.parameters()):,}  stubbed={stubbed}")

    import clash_royale_env as E
    n_obs = E.ClashRoyaleEnv().observation_size()
    obs = torch.randn(B, n_obs)
    hx = torch.zeros(B, net.LSTM_HIDDEN if hasattr(net, "LSTM_HIDDEN") else 256)
    cx = torch.zeros_like(hx)

    # Rollout-shaped forward (batch = num_envs, one step).
    warm = 5
    steps = args.updates * cfg.update_timestep if args.updates else 200
    steps = min(steps, args.torch_steps)
    for i in range(steps + warm):
        led.enabled = i >= warm
        with led.span("torch.rollout_step"):
            with torch.no_grad():
                with led.span("torch.fwd.mask"):
                    mask = net.affordability_mask(obs)
                with led.span("torch.fwd.trunk"):
                    feats, card_embeds, spatial = net.extract_features(obs)
                with led.span("torch.fwd.lstm_card"):
                    card_logits, _, _, value, (hx2, cx2) = net.step_lstm_and_card(
                        feats, (hx, cx), mask)
                idx = Categorical(logits=card_logits).sample()
                with led.span("torch.fwd.placement"):
                    pl = net.placement_given_card(hx2, card_embeds, idx, obs, spatial)
                Categorical(logits=pl).sample()
    led.enabled = True

    # Update-shaped forward+backward (one BPTT chunk), mirroring rl/ppo.py: the
    # trunk runs on a flat (L*B) batch and only the LSTMCell is looped.
    L = cfg.bptt_chunk
    segments = (cfg.update_timestep // cfg.bptt_chunk) * cfg.num_envs
    Bm = max(1, segments // cfg.num_minibatches)
    chunks_per_update = cfg.ppo_epochs * cfg.num_minibatches
    print(f"[profile] BPTT chunk L={L}  minibatch segments B={Bm}  "
          f"flat rows={L * Bm}  chunks per PPO update="
          f"{chunks_per_update} ({cfg.ppo_epochs} epochs x "
          f"{cfg.num_minibatches} minibatches)")

    flat = torch.randn(L * Bm, n_obs)
    rhx = torch.zeros(Bm, hx.shape[1])
    rcx = torch.zeros_like(rhx)
    card_actions = torch.randint(0, net.hand_size + 1, (L, Bm))
    cover_slot = torch.randint(0, net.hand_size + 1, (L, Bm))
    reset_seq = torch.ones(L, Bm)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)

    for i in range(args.torch_bwd + warm):
        led.enabled = i >= warm
        with led.span("torch.update_chunk"):
            opt.zero_grad(set_to_none=True)
            with led.span("torch.bptt.extract_features"):
                (feats, cembeds, spatial, hires) = net.extract_features_hires(flat)
                feats = feats.view(L, Bm, -1)
                hires = hires.view(L, Bm, *hires.shape[1:])
                spatial = spatial.view(L, Bm, *spatial.shape[1:])
                obs_seq = flat.view(L, Bm, -1)
                cembeds = cembeds.view(L, Bm, net.hand_size + 1, -1)
            with led.span("torch.bptt.mask"):
                cmask = net.affordability_mask(flat).view(L, Bm, net.hand_size + 1)
            with led.span("torch.bptt.forward_sequence"):
                (cl, pl, vals, aux, _, cf_pl) = net.forward_sequence(
                    feats, cembeds, spatial, obs_seq, cmask, card_actions,
                    reset_seq, (rhx, rcx), extra_card_idx_seq=cover_slot,
                    hires_seq=hires)
            with led.span("torch.bptt.backward"):
                loss = (cl.float().pow(2).mean() + pl.float().pow(2).mean()
                        + vals.float().pow(2).mean() + aux.float().pow(2).mean()
                        + cf_pl.float().pow(2).mean())
                loss.backward()
            with led.span("torch.bptt.opt_step"):
                opt.step()
    led.enabled = True

    per_chunk = led.incl.get("torch.update_chunk", 0.0) / max(1, args.torch_bwd)
    print(f"\n[profile] one PPO update = {chunks_per_update} chunks x "
          f"{1000 * per_chunk:.1f} ms = {chunks_per_update * per_chunk:.2f} s "
          f"of pure model time")

    total = led.incl.get("torch.rollout_step", 0.0) + led.incl.get("torch.update_chunk", 0.0)
    report(led, total, 0, "MODEL ONLY -- forward at rollout shape, "
                          "forward+backward at BPTT-chunk shape",
           [f"rollout batch = num_envs = {B}; BPTT chunk L={L}, "
            f"minibatch segments={Bm}, flat rows={L * Bm}",
            f"one PPO update = {chunks_per_update} of these chunks",
            "stubbed engine: constants only, NO simulation -- this measures the "
            "model and nothing else." if stubbed else "real engine present",
            "wall clock here is the sum of the two benchmarks, not a training run"])
    return total, 0


def run_boundary_mode(args, led):
    """The Python half of the C++/Python boundary, measurable with no engine.

    `src/bindings.cpp` returns std::vector<float> through pybind11/stl.h, which
    builds a Python list of one PyFloat per element; gym_wrapper then parses it
    back with np.asarray. This measures:

        list -> numpy       what gym_wrapper.step does today
        buffer -> numpy     what a py::array_t binding would cost instead

    The pybind half (building the list) needs the compiled module. The Python
    analogue printed below is an order-of-magnitude analogue, not a bound in
    either direction.
    """
    import numpy as np

    n = 13606
    try:
        import clash_royale_env as E
        n = E.ClashRoyaleEnv([1] * 8, [1] * 8, 100).observation_size()
        print(f"[profile] observation_size from the live engine: {n}")
    except Exception:
        K = _engine_constants_from_header()
        n = (K["BOARD_WIDTH"] * K["BOARD_HEIGHT"] * K["NUM_CHANNELS"]
             + 1 + K["HAND_SIZE"] + K["HAND_SIZE"] * K["NUM_CARD_IDS"]
             + K["NUM_EXTRA_SCALARS"])
        print(f"[profile] no engine; observation_size derived from "
              f"include/core/ClashEnv.h: {n}")

    reps = args.boundary_reps
    src_list = [float(i % 7) * 0.125 for i in range(n)]
    src_arr = np.asarray(src_list, dtype=np.float32)
    raw = src_arr.tobytes()

    t0 = time.perf_counter()
    for _ in range(reps):
        with led.span("boundary.list_to_numpy"):
            np.asarray(src_list, dtype=np.float32)
        with led.span("boundary.build_python_list"):
            [float(v) for v in src_arr]
        with led.span("boundary.frombuffer_zerocopy"):
            np.frombuffer(raw, dtype=np.float32)
        with led.span("boundary.numpy_copy_floor"):
            src_arr.copy()
    wall = time.perf_counter() - t0

    report(led, wall, 0, f"C++/Python BOUNDARY -- one {n}-float observation",
           [f"{reps} repeats of each; min/mean not shown, read us/call",
            "list_to_numpy    = the np.asarray() gym_wrapper.step runs today",
            "build_python_list= a Python-level ANALOGUE of pybind11/stl.h "
            "building the list, NOT a bound on it -- see the docstring",
            "frombuffer       = what a py::array_t<float> binding would cost",
            "numpy_copy_floor = a pure memcpy of the same bytes, the floor"])

    per_today = 1e6 * (led.incl["boundary.list_to_numpy"]
                       + led.incl["boundary.build_python_list"]) / reps
    per_zero = 1e6 * led.incl["boundary.frombuffer_zerocopy"] / reps
    print("")
    print(f"  np.asarray() on the returned list, per observation:"
          f" ~{1e6 * led.incl['boundary.list_to_numpy'] / reps:8.1f} us"
          f"   <- MEASURED, gym_wrapper pays this today")
    print(f"  the same observation as a py::array_t view:       "
          f" ~{per_zero:8.1f} us")
    print(f"  ratio on the half that IS measurable here: "
          f"{1e6 * led.incl['boundary.list_to_numpy'] / reps / max(per_zero, 1e-9):.0f}x")
    print(f"  (the pybind half is additional; CLAUDE.md measures it at "
          f"611 us on the real module)")
    return wall, 0


def _tensors(out):
    if hasattr(out, "shape"):
        return [out]
    acc = []
    for o in out:
        acc.extend(_tensors(o))
    return acc


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode",
                   choices=("sync", "async", "torch", "boundary"),
                   default="sync")
    p.add_argument("--episodes", type=int, default=60,
                   help="stop after this many completed episodes")
    p.add_argument("--updates", type=int, default=0)
    p.add_argument("--num-envs", type=int,
                   default=int(os.environ.get("CLASH_NUM_ENVS", 8)))
    p.add_argument("--update-timestep", type=int, default=500)
    p.add_argument("--teacher-stage", type=int, default=5,
                   help="TEACHER_STAGES rung; higher rungs cost more per decision")
    p.add_argument("--boundary-reps", type=int, default=300)
    p.add_argument("--torch-steps", type=int, default=200)
    p.add_argument("--torch-bwd", type=int, default=20)
    p.add_argument("--cprofile", action="store_true",
                   help="also dump a cProfile function ranking")
    p.add_argument("--out", default=os.path.join(
        os.environ.get("TEMP", "/tmp"), "clash_profile"))
    args = p.parse_args()

    led = Ledger()
    runner = {"torch": run_torch_mode,
              "boundary": run_boundary_mode}.get(args.mode, run_loop_mode)

    if args.cprofile:
        import cProfile
        import pstats
        pr = cProfile.Profile()
        pr.enable()
        runner(args, led)
        pr.disable()
        print("\n" + "=" * 86)
        print("  cProfile -- top 35 by cumulative time")
        print("=" * 86)
        pstats.Stats(pr).sort_stats("cumulative").print_stats(35)
    else:
        runner(args, led)


if __name__ == "__main__":
    main()
