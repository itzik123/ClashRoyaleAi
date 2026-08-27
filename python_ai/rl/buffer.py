"""The rollout buffer: one list per stored quantity, stacked into (T, N, ...).

Both pipelines declared fifteen-to-eighteen bare Python lists, appended to them
in the rollout, `torch.stack`ed each one by hand, and cleared each one by hand
at the end of the update. That is three places where adding a field means
editing three lists in two files, and the failure mode when one is missed is
silent: a buffer that is never cleared grows without bound, and one that is
never stacked raises far from where it was forgotten.

The field set is declared ONCE, at construction, and `add()` refuses a partial
row -- so a field that exists is always the same length as every other.

WHY THE FIELD SET IS A CONSTRUCTOR ARGUMENT rather than fixed: pipeline 2 stores
three extra quantities for correct bootstrapping through a truncation
(`boot_nonterminal`, `trunc_flag`, `trunc_boot`), and the advisor-target fields
only exist when the advisor is enabled. Declaring them makes the difference
between the two pipelines visible in one line instead of implied by which
`append` calls happen to run.
"""
import torch

#: Everything both pipelines always store.
CORE_FIELDS = (
    "obs",
    "card_actions",
    "placement_actions",
    #: 1 on steps where the agent actually had a CHOICE (>=2 legal actions), 0
    #: where the affordability mask left only the forced no-op. On a forced step
    #: the sampled action has probability exactly 1, so its log-prob is exactly
    #: 0, the PPO ratio is a constant 1, and the actor/entropy terms contribute
    #: exactly zero gradient -- including them in the loss DENOMINATOR anyway
    #: would silently scale the actor gradient down by ~3.7x, since only ~27% of
    #: steps have any affordable card. The critic still uses `valid`: the value
    #: function must be learned on every real state, choice or not.
    "decision",
    #: LSTM state as it was ENTERING each timestep -- the starting state for
    #: whichever truncated-BPTT chunk begins there. Captured before the forward
    #: pass and already detached, since the rollout runs under no_grad.
    "hx_in",
    "cx_in",
    "logprobs",
    "values",
    "rewards",
    "masks",
    #: 0 on phantom auto-reset steps. Under gymnasium's NEXT_STEP autoreset an
    #: env whose PREVIOUS step ended the episode does not execute the sampled
    #: action at all -- the worker just calls reset() -- so that step is not a
    #: real transition and must be excluded from every loss term.
    "valid",
    #: Ground-truth opponent elixir: supervision for the auxiliary head only,
    #: never an input.
    "aux_elixir",
    #: The affordable-but-not-necessarily-chosen slot the coverage term scores.
    #: Sampled ONCE at rollout time and buffered, never resampled inside a PPO
    #: epoch: an advisor target has to be computed against the observation the
    #: slot was drawn on, and a fresh draw would pair one card's logits with
    #: another card's target.
    "coverage_slot",
)

#: Pipeline 2 only. Correct-bootstrap GAE bookkeeping: `boot_nonterminal` is 0
#: only on TRUE terminals (bootstrap otherwise), `trunc_flag` marks steps whose
#: next-state value must come from the captured `trunc_boot` rather than from
#: the next (already-reset) episode's V.
TRUNCATION_FIELDS = ("boot_nonterminal", "trunc_flag", "trunc_boot")

#: Present only when the advisor is enabled (`advisors.advisor_target.enabled`).
ADVISOR_FIELDS = ("coverage_target", "coverage_has")


class RolloutBuffer:
    """Append-only per-step storage, stacked to (T, N, ...) for the update."""

    def __init__(self, fields=CORE_FIELDS):
        self.fields = tuple(fields)
        if len(set(self.fields)) != len(self.fields):
            raise ValueError(f"duplicate field in {self.fields}")
        self._data = {name: [] for name in self.fields}

    def add(self, **values):
        """Store one timestep. Every declared field must be supplied.

        Refusing a partial row is the point: a buffer where one list is shorter
        than the others stacks into a silently misaligned batch, and the
        misalignment shows up as a corrupted PPO ratio rather than an error.
        """
        missing = set(self.fields) - set(values)
        extra = set(values) - set(self.fields)
        if missing or extra:
            raise KeyError(
                f"rollout step must supply exactly {sorted(self.fields)}; "
                f"missing {sorted(missing)}, unexpected {sorted(extra)}")
        for name, value in values.items():
            self._data[name].append(value)

    def __len__(self):
        return len(self._data[self.fields[0]])

    def __contains__(self, name):
        return name in self._data

    def stack(self):
        """{field: (T, N, ...) tensor}. Does not clear -- see `clear()`."""
        if not len(self):
            raise RuntimeError("stack() on an empty rollout buffer")
        return {name: torch.stack(values) for name, values in self._data.items()}

    def drain(self):
        """`stack()`, then release the per-step lists.

        `torch.stack` ALLOCATES and copies -- the batch it returns does not view
        the stored tensors -- so from the moment it returns there are two full
        copies of every field alive, and the list half is dead weight nothing
        reads again.

        The update used to run with both resident, because the caller cleared
        only after it returned. At the production shape that is expensive:

            obs 13606 x 8 envs x 4 bytes = 0.435 MB/step
            x 500 steps                  = 217.7 MB
            both copies                  = 435.4 MB

        held across the PPO update, i.e. ~87% of the cycle, against a main
        process measured at 1194 MB private commit. Draining at the stack point
        gives ~218 MB back for exactly the phase that needs it most, and costs
        nothing: the batch already owns its storage.

        Safe precisely BECAUSE stack does not alias, which
        `tests/test_rl_buffer_drain.py` pins as a separate premise rather than
        assuming -- if that ever changed, draining would pull storage out from
        under the update.
        """
        batch = self.stack()
        self.clear()
        return batch

    def clear(self):
        for values in self._data.values():
            values.clear()
