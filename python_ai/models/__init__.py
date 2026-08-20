"""The policy network, its checkpoint IO, and the observation layout.

`net.py` is the single place that knows how an observation is laid out; every
constant in it is derived from the compiled engine rather than copied. Nothing
here imports a trainer -- a probe that wants to read one `.pth` should be able
to do so without dragging a 2,000-line training script into the process."""
