"""The potential-based shaping terms must not carry their own copy of gamma.

`compute_shaping` and `solvency_shaping` both take `gamma` and both document,
at length, that policy-invariance holds ONLY if it is the same gamma GAE
discounts with. `base_trainer` passes `cfg.gamma` explicitly and says so. But
both functions also carry a DEFAULT of 0.99 -- a literal, independent of
`PPOConfig.gamma`.

While the two numbers coincide that default is harmless, which is exactly what
makes it dangerous: it is a tripwire armed by whoever next tunes the discount.
`tools/validate_pipeline.py` and roughly ten test call sites already rely on
the default, so the moment `PPOConfig.gamma` moves, those callers silently
compute `F = 0.99*Phi(s') - Phi(s)` against a trainer discounting at the new
rate. That is not a rounding difference -- it is the difference between
policy-invariant shaping and shaping that changes which policy is optimal,
which is the precise failure `rewards/shaping.py`'s own comment warns about:

    "That looks almost identical and is not: Ng et al.'s policy-invariance
     result requires the discounted form, and with gamma<1 the undiscounted
     difference does change which policy is optimal."

This is the sixth instance of CLAUDE.md's no-second-copies rule in this repo,
and the first where the copy is a FUNCTION DEFAULT rather than a module
constant -- which is why none of the earlier greps for restated constants
found it.

THE FIX IS ABSENCE, NOT DERIVATION, and the reason is a layering rule.
Deriving the default from `PPOConfig` was tried first and is ILLEGAL here:
`tests/test_package_layout.py::test_the_reward_layer_imports_no_torch_and_no_trainer`
enforces that `rewards/` is a leaf which may import neither `rl/` nor
`trainers/`, because the shaping terms are pure functions of engine statistics
and keeping the policy layer out of them is what makes them testable with
plain arrays. So the discount cannot be read here at all.

Requiring the argument turns out to be the stronger guarantee anyway. A
derived default still lets a caller stay silent about the discount and be
right by construction; a required one makes "which gamma is this shaping for"
a question every call site must answer out loud. Production callers pass
`cfg.gamma`; the arithmetic unit tests pass their own fixed fixture value,
which is correct rather than a copy -- they are pinning the FORMULA, and a
test that read the live config would change its own expected answers whenever
someone tuned the discount.
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
    """THE INVARIANT. No default means no copy, and no copy cannot go stale.

    A default is what made this a latent defect for as long as it was one: a
    caller could omit the discount, get 0.99, and be silently correct only for
    as long as the trainer also used 0.99.
    """
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
    """Pins WHY the fix is absence rather than derivation, so the next person
    to reach for `from python_ai.rl.config import PPOConfig` in here finds the
    reason before the layout test tells them no."""
    for fn in (compute_shaping, solvency_shaping):
        src = inspect.getsource(inspect.getmodule(fn))
        assert "python_ai.rl" not in src, (
            "the rewards layer imports the rl layer; test_package_layout.py "
            "forbids it, and that prohibition is why gamma is a required "
            "argument here rather than a derived default")


def _gamma_default_node(fn):
    """The AST node for `fn`'s `gamma` default, or None if it has none.

    Read the SOURCE rather than the value, because a default argument is bound
    once at import: by the time the value exists, a literal 0.99 and a derived
    0.99 are indistinguishable, which is precisely the confusion this test
    exists to end.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    fndef = tree.body[0]
    args = fndef.args.args
    defaults = fndef.args.defaults
    # Defaults align to the TAIL of the positional args, so a parameter with
    # no default simply does not appear in this mapping. `None` is the
    # CORRECT answer here and the one the caller asserts on -- gamma having
    # dropped out of the defaults is exactly the state we want.
    named = [a.arg for a in args[len(args) - len(defaults):]]
    return dict(zip(named, defaults)).get("gamma")


def test_no_numeric_literal_creeps_back_as_a_gamma_default():
    """Equality alone is satisfied by a lucky literal, and a lucky literal is
    what this repo has been bitten by six times. Pin the DERIVATION.

    Belt and braces over the signature check above, read from the SOURCE:
    the parameter must carry no default node at all. Kept as a separate test
    because it fails with a different message -- naming the literal that came
    back -- which is the one a future reader will need.
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
