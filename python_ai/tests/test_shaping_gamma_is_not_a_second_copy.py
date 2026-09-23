"""The potential-based shaping terms carry no copy of gamma.

Policy invariance holds only if `gamma * Phi(s') - Phi(s)` uses the gamma GAE
discounts with. A default gamma is a second copy that stays correct only until
the discount is tuned, then silently changes which policy is optimal.

The fix is absence, not derivation: `rewards/` is a leaf that may not import
`rl/` (test_package_layout.py), so the config cannot be read here. A required
argument is the stronger guarantee anyway: every call site must say which gamma
it means. Production passes `cfg.gamma`; the arithmetic tests pass a fixed
fixture value, since they pin the formula.
"""
import ast
import inspect
import textwrap

from python_ai.rewards import elixir_shaping as elixir_shaping_module
from python_ai.rewards.shaping import compute_shaping
from python_ai.rewards.elixir_shaping import solvency_shaping
from python_ai.rl.config import PPOConfig


def _gamma_param(fn):
    return inspect.signature(fn).parameters["gamma"]


def test_neither_potential_based_term_has_a_gamma_default():
    """The invariant: no default, so no copy to go stale."""
    for label, fn in (("compute_shaping", compute_shaping),
                      ("solvency_shaping", solvency_shaping)):
        p = _gamma_param(fn)
        assert p.default is inspect.Parameter.empty, (
            f"{label}'s gamma defaults to {p.default!r}. That is a second copy "
            "of the discount: any caller omitting the argument computes "
            "gamma*Phi(s') - Phi(s) at a rate that has nothing to do with what "
            "GAE is discounting, which breaks the Ng et al. policy-invariance "
            "these terms are built on -- and breaks it silently, because the "
            "wrong number is still a perfectly plausible one.")


def test_the_reward_layer_still_cannot_reach_the_config():
    """Pins why the fix is absence rather than derivation."""
    for fn in (compute_shaping, solvency_shaping):
        src = inspect.getsource(inspect.getmodule(fn))
        assert "python_ai.rl" not in src, (
            "the rewards layer imports the rl layer; test_package_layout.py "
            "forbids it, and that prohibition is why gamma is a required "
            "argument here rather than a derived default")


def _gamma_default_node(fn):
    """The AST node for `fn`'s `gamma` default, or None.

    Read from source: once bound at import, a literal 0.99 and a derived 0.99
    are indistinguishable.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    fndef = tree.body[0]
    args = fndef.args.args
    defaults = fndef.args.defaults
    # Defaults align to the tail of the positional args; a parameter without
    # one is absent from this mapping, and None is the state we want.
    named = [a.arg for a in args[len(args) - len(defaults):]]
    return dict(zip(named, defaults)).get("gamma")


def test_no_numeric_literal_creeps_back_as_a_gamma_default():
    """Belt and braces over the signature check, read from source, with a message
    naming the literal if one comes back.
    """
    for module_name, fn in (("rewards/shaping.py", compute_shaping),
                            ("rewards/elixir_shaping.py", solvency_shaping)):
        node = _gamma_default_node(fn)
        assert node is None, (
            f"{module_name}: gamma has re-acquired a default. "
            "That is an independent copy of the discount; equality with "
            "PPOConfig.gamma today is a coincidence that the next tuning "
            "change will end silently, and silently is the whole problem -- "
            "the caller keeps computing gamma*Phi(s') - Phi(s) at the OLD "
            "rate and nothing reports it. Bind it to a name instead.")
