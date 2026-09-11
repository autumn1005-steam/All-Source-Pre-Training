#!/usr/bin/env python3
from pathlib import Path
import json, sys, time
ROOT=Path("/root/autodl-tmp/TempoFound-OOS-HRCSR-TKDE-v3.1-validation-clean")
sys.path.insert(0,str(ROOT/"src"))
from tempofound.config import ExperimentConfig
from tempofound.transfer import run_transfer
if len(sys.argv) != 4:
    raise SystemExit("usage: run_target_only_dataset.py CONFIG OUTPUT DATASET")
cfg=ExperimentConfig.from_yaml(sys.argv[1]); out=Path(sys.argv[2]); target=sys.argv[3]
processed="/root/autodl-tmp/processed-westb-verified/processed"
out.mkdir(parents=True, exist_ok=True)
for seed in (0,1,2,3,42):
    path=out/target/f"seed{seed}"/"summary.json"
    if path.exists():
        print(f"SKIP target={target} seed={seed}",flush=True); continue
    print(f"START target={target} seed={seed}",flush=True); started=time.monotonic()
    result=run_transfer(cfg,checkpoint_path=None,target=target,seed=seed,processed_dir=processed,mode="scratch",device="cuda:0",evaluation_protocol="ranking",negative_policy="historical",negatives_per_positive=20,evaluation_batch_size=128)
    result["elapsed_seconds"]=time.monotonic()-started
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2,sort_keys=True),encoding="utf-8")
    print(f"FINISH target={target} seed={seed} mrr={result.get('test_mrr')} elapsed={result['elapsed_seconds']:.1f}",flush=True)
print("TARGET_ONLY_DONE",target,flush=True)
