# Control experiment launchers

`run_graphmixer.sh` runs the complete Forward-Pair pipeline with the shared
GraphMixer-style sequence encoder. The remaining scripts create deterministic
candidate manifests for the Random-7, exhaustive, and standard-greedy controls.
Each candidate in those manifests is trained and scored with the same
`scripts/pretrain.py` and `scripts/score_proxy.py` interfaces used by the
primary pipeline, which preserves the matched model and update budget.
