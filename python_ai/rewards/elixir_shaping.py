"""Elixir solvency shaping.

A potential that charges the agent at the moment it dips below the reserve,
instead of many steps later through a tower it could not defend.
Potential-based, so it only redistributes credit: if over-spending were optimal
this would change nothing, and unlike a plain penalty it cannot make doing
nothing pay.
"""
import numpy as np

# Elixir the agent should still hold after acting: the cost of a dedicated
# defensive answer.
SOLVENCY_RESERVE = 4.0

# Numerically equal to W_ELIXIR_OVERFLOW, its mirror image (together they leave
# 4-9 elixir free of charge), but a different quantity. Not imported from
# weights.py: weights imports this module.
W_SOLVENCY = 0.1


def solvency_potential(elixir, reserve=SOLVENCY_RESERVE, w=W_SOLVENCY):
    """Phi(s): 0 at or above the reserve, falling linearly to -w at zero elixir.

    Linear so the agent can prefer 3.9 to 0.1. `elixir` is the current reading,
    never a delta.
    """
    shortfall = np.maximum(0.0, reserve - np.asarray(elixir, dtype=np.float32))
    return (-w * shortfall / reserve).astype(np.float32)


def solvency_shaping(stats, prev_stats, gamma,
                     reserve=SOLVENCY_RESERVE, w=W_SOLVENCY):
    """F = gamma*Phi(s') - Phi(s).

    `gamma` is required for the same reason as in compute_shaping.
    Phi(terminal) is not zeroed; the residue is small (at most -0.019
    measured).
    """
    return (gamma * solvency_potential(stats["team0_elixir_current"], reserve, w)
            - solvency_potential(prev_stats["team0_elixir_current"], reserve, w)
            ).astype(np.float32)


def bankruptcy_rate(elixir, floor=3.0):
    """Share of decisions below `floor` elixir.

    3.0 is a constrained-economy line, not an empty action space (the deck has
    1-cost cards). Kept because the logged baselines were measured against it.
    """
    return float(np.mean(np.asarray(elixir, dtype=np.float32) < floor))
