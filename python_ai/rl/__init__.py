"""The PPO algorithm, independent of which opponent the environment supplies.

Everything here is deliberately free of any Clash-specific policy decision: it
takes a net, a vector env and a config, and it does recurrent PPO. The two
pipelines in `trainers/` are what supply the opponent, the curriculum and the
bookkeeping -- that split is what let ~1,600 lines of duplicated loop become one
`BaseTrainer`."""
