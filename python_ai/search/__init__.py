"""Decision-time lookahead: roll candidate actions forward on `env.snapshot()`
and pick by the critic. Worth +0.4025 win rate against the C++ heuristic at
horizon 12, and NOT distillable into the weights -- see CLAUDE.md."""
