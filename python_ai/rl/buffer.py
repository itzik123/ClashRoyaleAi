"""The rollout buffer: one list per stored quantity, stacked into (T, N, ...).

The field set is declared once, at construction, and `add()` refuses a partial
row, so every field always has the same length. It is a constructor argument
because pipeline 2 stores truncation-bootstrap fields and the advisor fields
exist only when the advisor is on.
"""
import torch

#: Everything both pipelines always store.
CORE_FIELDS = (
    "obs",
    "card_actions",
    "placement_actions",
    #: 1 where the agent had a choice (>= 2 legal actions). A forced step has
    #: log-prob 0 and contributes no gradient, so counting it in the actor's
    #: denominator would shrink the step ~3.7x. The critic uses `valid`.
    "decision",
    #: LSTM state entering each timestep: where a truncated-BPTT chunk starting
    #: there resumes.
    "hx_in",
    "cx_in",
    "logprobs",
    "values",
    "rewards",
    "masks",
    #: 0 on phantom auto-reset steps, whose sampled action never ran; excluded
    #: from every loss.
    "valid",
    #: The opponent's play on this step (-1 for none), as a raw stream;
    #: engine_stats.next_card_labels turns it into labels once per update.
    #: Supervision only, never an input.
    "aux_opp_played",
    #: The affordable slot the coverage term scores, drawn once at rollout
    #: time; the advisor target must match the observation it was drawn on.
    "coverage_slot",
)

#: Pipeline 2 only: `boot_nonterminal` is 0 only on true terminals;
#: `trunc_flag` marks steps whose next-state value is the captured
#: `trunc_boot`.
TRUNCATION_FIELDS = ("boot_nonterminal", "trunc_flag", "trunc_boot")

#: Only when the advisor is enabled.
ADVISOR_FIELDS = ("coverage_target", "coverage_has")

#: Champion ability actions and their readiness mask, one column per Champion
#: head (rl/abilities.py). Readiness is buffered because it is not in the
#: observation.
ABILITY_FIELDS = ("ability_actions", "ability_ready")


class RolloutBuffer:
    """Append-only per-step storage, stacked to (T, N, ...) for the update."""

    def __init__(self, fields=CORE_FIELDS):
        self.fields = tuple(fields)
        if len(set(self.fields)) != len(self.fields):
            raise ValueError(f"duplicate field in {self.fields}")
        self._data = {name: [] for name in self.fields}

    def add(self, **values):
        """Store one timestep; every declared field must be supplied, since a
        short list would stack into a silently misaligned batch.
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
        """{field: (T, N, ...) tensor}. Does not clear."""
        if not len(self):
            raise RuntimeError("stack() on an empty rollout buffer")
        return {name: torch.stack(values) for name, values in self._data.items()}

    def drain(self):
        """`stack()`, then release the per-step lists.

        `torch.stack` copies, so the lists are dead weight during the update
        (~218 MB of observations at the production shape). Safe only because
        stack does not alias; `tests/test_rl_buffer_drain.py` pins that.
        """
        batch = self.stack()
        self.clear()
        return batch

    def clear(self):
        for values in self._data.values():
            values.clear()
