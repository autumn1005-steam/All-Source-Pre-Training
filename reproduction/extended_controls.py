"""Isolate the anchor's maximum-versus-mean semantic attention summary."""
import json,sys,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent
def run_extended():
 plan=json.loads((ROOT/'study_plan.json').read_text())
 if plan.get('attention_control_status')=='complete':return
 def execute(spec):
  ds,seed,lr=spec;key=f'{ds}__mean_anchor__supplied__k16__lam0.5__lr{lr}__seed{seed}'
  log=ROOT/'job_logs'/(key+'.log')
  with log.open('w') as f:
   subprocess.run([sys.executable,'-u',str(ROOT/'experiment.py'),'--dataset',ds,'--variant','mean_anchor','--seed',str(seed),'--epochs','600','--lrscale',str(lr)],stdout=f,stderr=subprocess.STDOUT,check=True)
  print('ATTENTION_CONTROL',key,flush=True);return key
 specs=[(ds,seed,plan['primary'][ds]['full']['lrscale']) for ds in ['acm','aminer'] for seed in [1,2,3]]
 with ThreadPoolExecutor(max_workers=3) as executor:keys=list(executor.map(execute,specs))
 for ds in ['acm','aminer']:
  plan['ablation'].append({'dataset':ds,'variant':'mean_anchor','keys':[k for k in keys if k.startswith(ds+'__')]})
 plan['attention_control_status']='complete';(ROOT/'study_plan.json').write_text(json.dumps(plan,indent=2))
if __name__=='__main__':run_extended()
