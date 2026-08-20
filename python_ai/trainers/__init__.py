"""The two training pipelines, plus the offline trainers that feed them.

`train` (pipeline 1, vs the UtilityTeacher) hands off to `train_selfplay`
(pipeline 2, the PFSP league) by subprocess. Both are thin: they compose
`rl.BaseTrainer` with their own opponent management and their own bookkeeping."""
