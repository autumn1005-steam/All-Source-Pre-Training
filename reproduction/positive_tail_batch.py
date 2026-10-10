"""Assist the fixed accuracy schedule, with per-result process locks."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess,sys,json,time
ROOT=Path(__file__).resolve().parent
jobs=[dict(dataset=d,mode=m,seed=s) for d in ['aminer','freebase']
      for m in ['simrank','random'] for s in [1,2,3]]
out=ROOT/'positive_extension';logs=out/'tail_job_logs';logs.mkdir(exist_ok=True)
(out/'tail_dispatch.json').write_text(json.dumps(dict(jobs=jobs,workers=2,
    purpose='Assist the same preregistered jobs; exclusive result locks prevent duplicate encoder training',
    latency_claim=False),indent=2))
def run(spec):
    command=[sys.executable,'-u',str(ROOT/'positive_study.py'),'--job']
    for k,v in spec.items():command.extend(['--'+k,str(v)])
    name=f"{spec['dataset']}__{spec['mode']}__seed{spec['seed']}"
    print('TAIL_START',name,flush=True)
    with (logs/(name+'.log')).open('w') as f:
        subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    print('TAIL_FINISH',name,flush=True)
with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,jobs))
print('FIXED_TAIL_COMPLETE',flush=True)
