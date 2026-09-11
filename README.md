# HRCSR

Core implementation for hierarchical risk-controlled source routing on
cross-domain temporal graphs.

## Contents

- `src/` — temporal graph model, source experts, routing, transfer, sampling,
  and evaluation;
- `configs/` — experiment configurations;
- `experiments/` — runnable experiment entry points;
- `scripts/` — training, aggregation, profiling, and validation utilities;
- `tests/` — automated software tests;
- `results/aggregated/` — selected aggregated experiment results.

## Quick check

```bash
python -m pip install -e '.[test]'
python -m pytest -q
bash experiments/run_smoke.sh
```

The smoke run is only a software check. See the configuration files and
experiment scripts for reproducible training and evaluation commands.
