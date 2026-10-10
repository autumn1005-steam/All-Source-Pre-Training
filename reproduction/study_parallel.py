"""Fixed revision study; independent encoders may run concurrently.

GPU latency is measured separately after this study on an idle device.
AdaMEOW's dense pair scorer runs alone to respect its peak memory footprint.
"""
import json, subprocess, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parent
PY=sys.executable
DS=['acm','dblp','aminer','freebase']
def key(ds,var,pos='supplied',lr=1.,seed=0,k=16,lam=.5):
 if var=='ns4gc':return f'{ds}__ns4gc__projected__lr{lr}__seed{seed}'
 if var=='adameow':return f'{ds}__adameow__common__lr{lr}__seed{seed}'
 return f'{ds}__{var}__{pos}__k{k}__lam{lam}__lr{lr}__seed{seed}'
def job(ds,var,seed,epochs,lr=1.,pos='supplied',k=16,lam=.5):
 return dict(dataset=ds,variant=var,seed=seed,epochs=epochs,lrscale=lr,positive=pos,k=k,lam=lam)
def run(spec):
 s=dict(spec);var=s['variant'];name=key(s['dataset'],var,s['positive'],s['lrscale'],s['seed'],s['k'],s['lam'])
 path=ROOT/'results'/(name+'.json')
 if path.exists():
  r=json.loads(path.read_text())
  if r['args']['epochs']==s['epochs'] and all('validation_macro' in v for v in r['classification'].values()):
   print('REUSE',name,flush=True);return name
 script='experiment.py'
 if var in ['ns4gc','adameow']:
  script=var+'_runner.py'
  for p in ['variant','positive','k','lam']:s.pop(p)
 cmd=[PY,'-u',str(ROOT/script)]
 for k,v in s.items():cmd.extend(['--'+k.replace('_','-'),str(v)])
 cmd.append('--force')
 log=ROOT/'job_logs'/(name+'.log');log.parent.mkdir(exist_ok=True)
 print('START',name,flush=True)
 with log.open('w') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('FINISH',name,flush=True);return name
def batch(specs,workers=3):
 with ThreadPoolExecutor(max_workers=workers) as executor:return list(executor.map(run,specs))
def save(plan):
 p=ROOT/'study_plan.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(plan,indent=2));tmp.replace(p)

plan={'selection':'mean validation Macro-F1 over 20/40/60 public label budgets; test metrics excluded from selection',
      'final_encoder_seeds':[1,2,3],'linear_probe_initializations_per_encoder':5,'kmeans_initializations_per_encoder':3,
      'primary':{},'ablation':[],'sensitivity':[],'tuning':[],
      'execution':'three independent jobs concurrently; AdaMEOW runs alone; efficiency measured afterwards on idle GPU'}
tune=[job(ds,var,0,400,lr) for ds in DS for var in ['full','heco'] for lr in [.5,1.,2.]]
plan['tuning']=batch(tune)
for ds in DS:
 plan['primary'][ds]={}
 for var in ['full','heco']:
  trials=[]
  for lr in [.5,1.,2.]:
   r=json.loads((ROOT/'results'/(key(ds,var,lr=lr)+'.json')).read_text())
   trials.append((sum(v['validation_macro'] for v in r['classification'].values())/3,lr))
  plan['primary'][ds][var]={'lrscale':max(trials)[1],'keys':[]}
 plan['primary'][ds]['refined']={'lrscale':plan['primary'][ds]['full']['lrscale'],'k':8,'keys':[]}
 plan['primary'][ds]['ns4gc']={'lrscale':1.,'keys':[]}
 if ds!='freebase':plan['primary'][ds]['adameow']={'lrscale':1.,'keys':[]}
save(plan)
primary=[];mapping=[];ada=[];ada_mapping=[]
for ds in DS:
 for label,spec in plan['primary'][ds].items():
  for seed in [1,2,3]:
   if label=='refined':s=job(ds,'full',seed,600,spec['lrscale'],'refine_supplied',8)
   else:s=job(ds,label,seed,600,spec['lrscale'])
   if label=='adameow':ada.append(s);ada_mapping.append((ds,label))
   else:primary.append(s);mapping.append((ds,label))
for name,(ds,label) in zip(batch(primary),mapping):plan['primary'][ds][label]['keys'].append(name)
save(plan)
ablations=[];mapping=[]
for ds in ['acm','aminer']:
 lr=plan['primary'][ds]['full']['lrscale']
 for var in ['mp_only','mp_aug','mp_ng','mp_sm','no_global','no_cn','layernorm']:
  i=len(plan['ablation']);plan['ablation'].append({'dataset':ds,'variant':var,'keys':[]})
  for seed in [1,2,3]:ablations.append(job(ds,var,seed,600,lr));mapping.append(i)
for ds in DS:
 lr=plan['primary'][ds]['full']['lrscale'];i=len(plan['ablation'])
 plan['ablation'].append({'dataset':ds,'variant':'random_positive_k8','keys':[]})
 for seed in [1,2,3]:ablations.append(job(ds,'full',seed,600,lr,'random_supplied',8));mapping.append(i)
for name,i in zip(batch(ablations),mapping):plan['ablation'][i]['keys'].append(name)
save(plan)
# Fixed loss coefficient sensitivity, without using test outcomes for selection.
sens=[];mapping=[]
for ds in ['acm','aminer']:
 lr=plan['primary'][ds]['full']['lrscale']
 for lam in [.25,.75]:
  i=len(plan['sensitivity']);plan['sensitivity'].append({'dataset':ds,'lam':lam,'keys':[]})
  for seed in [1,2,3]:sens.append(job(ds,'full',seed,600,lr,lam=lam));mapping.append(i)
for name,i in zip(batch(sens),mapping):plan['sensitivity'][i]['keys'].append(name)
save(plan)
for name,(ds,label) in zip(batch(ada,workers=1),ada_mapping):plan['primary'][ds][label]['keys'].append(name)
plan['status']='complete';save(plan);(ROOT/'STUDY_COMPLETE').write_text('complete\n');print('STUDY_COMPLETE',flush=True)
